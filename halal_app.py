#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

from md_utils import clean_and_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("halal-food-bot")

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
LLM_API_KEY = os.environ["GEMINI_API_KEY"]
CHANNEL_ID = "@jyu_halal_ruokalista"
LUNCHES_API = "https://jybar.app.jyu.fi/api/2/lunches"
LLM_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_MODEL = "gemini-2.5-flash"

FISH_FILTER_SCHEMA = {
    "name": "fish_filter_response",
    "strict": "true",
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "allow": {"type": "boolean"},
            },
            "required": ["id", "allow"],
        },
    },
}

CHEFS_CHOICE_SCHEMA = {
    "name": "chefs_choice_response",
    "strict": "true",
    "schema": {
        "type": "object",
        "properties": {
            "dish": {"type": "string"},
            "restaurant": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["dish", "restaurant", "reason"],
    },
}

TRANSLATION_SCHEMA = {
    "name": "translation_response",
    "strict": "true",
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original": {"type": "string"},
                        "translated": {"type": "string"},
                    },
                    "required": ["original", "translated"],
                },
            }
        },
        "required": ["translations"],
    },
}


def has_non_english_chars(text: str) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", text))


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]*>.*?</[^>]*>|<[^>]+>", "", text, flags=re.DOTALL).strip()


def normalize_diet(diet: str) -> str:
    diet_mapping = {
        "VEG": {"VEG", "VEGAN", "VEGAANI", "VEGAANINEN", "VEGETAARINEN", "VEGETARIAN"},
        "L": {"L", "LAKTOOSITON"},
        "G": {"G", "GLUTEENITON"},
        "M": {"M", "MAIDOTON"},
    }

    upper_diet = diet.upper()
    for normalized, variations in diet_mapping.items():
        if any(upper_diet.startswith(v) for v in variations):
            return normalized
    return upper_diet


def is_veg(diets: List[str]) -> bool:
    return any(normalize_diet(d) == "VEG" for d in diets)


def is_valid_price(price: str) -> bool:
    price_pattern = r"\d{1,2},\d{2}"
    return bool(re.fullmatch(price_pattern, price.strip()))


def get_most_common_price(items: List[List[Dict]]) -> Optional[str]:
    all_prices = [
        item.get("price", "").strip()
        for item_group in items
        for item in item_group
        if item.get("price", "").strip()
    ]

    if not all_prices:
        return None

    price_counts = Counter(p.replace(" ", "") for p in all_prices)
    most_common_normalized = price_counts.most_common(1)[0][0]

    return next(p for p in all_prices if p.replace(" ", "") == most_common_normalized)


async def get_location_name(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> str:
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                address = data.get("address", {})
                return address.get("suburb") or address.get("neighbourhood") or ""
    except Exception as e:
        log.error(f"Error fetching location name: {e}")
    return ""


async def fetch_menus(session: aiohttp.ClientSession) -> List[Dict]:
    try:
        async with session.get(LUNCHES_API) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("results", {}).get("en", [])
    except Exception as e:
        log.error(f"Error fetching menus: {e}")
        return []


async def llm_chat_json(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    temperature: float = 0.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": temperature,
    }
    async with session.post(LLM_CHAT_URL, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def translate_dishes(
    session: aiohttp.ClientSession, dishes_by_restaurant: Dict[str, List[str]]
) -> Dict[str, str]:
    """Translate non-English dish names to English."""
    # Collect all dishes from restaurants that have non-English dishes
    all_dishes_to_translate = []

    for restaurant, dishes in dishes_by_restaurant.items():
        has_non_english = any(has_non_english_chars(dish) for dish in dishes)
        if has_non_english:
            all_dishes_to_translate.extend(dishes)

    if not all_dishes_to_translate:
        return {}

    unique_dishes = list(dict.fromkeys(all_dishes_to_translate))

    prompt = f"""
Translate the following dish names to English. If a dish name is already in English, return it exactly as is. Keep translations concise and food-appropriate.

Dishes to translate:
{json.dumps(unique_dishes, ensure_ascii=False)}

Return JSON with original and translated versions for each dish.
"""

    try:
        content = await llm_chat_json(
            session, [{"role": "user", "content": prompt}], TRANSLATION_SCHEMA
        )
        content = strip_html_tags(content).strip()

        obj = json.loads(content)
        translations = {}

        for item in obj.get("translations", []):
            original = item.get("original", "")
            translated = item.get("translated", "")
            if original and translated:
                translations[original] = translated

        return translations
    except Exception as e:
        log.error(f"Error translating dishes: {e}")
        return {}


def build_fish_filter_prompt(candidates: List[Dict[str, Any]]) -> str:
    diets_legend = (
        "(G) Gluten-free, (L) Lactose-free, (VL) Low lactose, "
        "(M) Dairy-free, (Veg) Suitable for vegans."
    )
    return f"""
You are filtering dishes for a Halal-friendly list.

ALLOW only dishes that contain dairy/eggs, fish or seafood and no other meat (e.g. chicken, beef, pork).
If ingredients missing, deduce from name.
Diets legend: {diets_legend}

Input dishes:
{json.dumps(candidates, ensure_ascii=False)}

Output Example:
[
    {{
    "id": "Restaurant1|2",
    "name": "Dish name",
    "allow": false
    }},
    {{
    "id": "Restaurant2|2",
    "name": "Dish name",
    "allow": false
    }}
]
"""


def build_chefs_choice_prompt(lines: List[str]) -> str:
    return f"""
You are selecting the tastiest dish.
Your reason should be concise, minimal, informative, and not exceed 1 sentence.

Pick exactly ONE from this list:
{chr(10).join(lines)}

Return JSON example:
{{
    "dish": "Falafel",
    "restaurant": "Restaurant",
    "reason": "Falafel is a popular Middle Eastern dish made from ground chickpeas or fava beans, seasoned with herbs and spices."
}}
"""


async def filter_fish_only(
    session: aiohttp.ClientSession, candidates: List[Dict]
) -> Dict[str, bool]:
    if not candidates:
        return {}

    prompt = build_fish_filter_prompt(candidates)
    content = await llm_chat_json(
        session, [{"role": "user", "content": prompt}], FISH_FILTER_SCHEMA
    )
    content = strip_html_tags(content)

    try:
        arr = json.loads(content)
        result = {}
        candidate_ids = {c["id"] for c in candidates}

        for obj in arr:
            if isinstance(obj.get("id"), str) and obj["id"] in candidate_ids:
                result[obj["id"]] = bool(obj.get("allow", False))

        return result
    except Exception as e:
        log.error("Failed to parse fish-filter JSON. Content was:\n%s", content)
        log.exception(e)
        return {}


async def get_chefs_choice(
    session: aiohttp.ClientSession, dishes: List[Tuple[str, str]]
) -> str:
    if not dishes:
        return ""

    lines = [f"{name} @ {rest}" for name, rest in dishes]
    prompt = build_chefs_choice_prompt(lines)

    try:
        content = await llm_chat_json(
            session, [{"role": "user", "content": prompt}], CHEFS_CHOICE_SCHEMA, 0.2
        )
        content = strip_html_tags(content).strip()

        obj = json.loads(content)
        dish = obj.get("dish", "").strip()
        rest = obj.get("restaurant", "").strip()
        reason = obj.get("reason", "").strip()

        if dish and rest and reason:
            return f"*{dish}* @ _{rest}_\n💬 _{reason}_"

        return ""
    except Exception as e:
        log.error("Error getting chef's choice: %s", e)
        return ""


async def format_restaurant_menu(
    restaurant: Dict,
    allowed_fish: Set[str],
    session: aiohttp.ClientSession,
    translations: Dict[str, str] = None,
) -> Tuple[Optional[str], List[str]]:
    """Format restaurant menu and return both formatted text and list of dishes."""
    name = restaurant.get("name", "").strip()
    location = restaurant.get("location", {})
    items = restaurant.get("items", [])
    translations = translations or {}

    if not name or not items:
        return None, []

    location_name = ""
    if location.get("lat") and location.get("lon"):
        osm_location = await get_location_name(
            location["lat"], location["lon"], session
        )
        if osm_location:
            location_name = f" ({osm_location})"

    seen_items = set()
    menu_items = []
    dish_names = []
    common_price = get_most_common_price(items)

    for item_group in items:
        for item in item_group:
            item_name = item.get("name", "").strip()
            if not item_name or item_name in seen_items:
                continue

            item_diets = item.get("diets", [])
            item_price = item.get("price", "").strip()

            # Include if veg or if it's an allowed fish dish
            if is_veg(item_diets) or item_name in allowed_fish:
                # For non-veg items, check price matches
                if not is_veg(item_diets):
                    if not common_price or item_price.replace(
                        " ", ""
                    ) != common_price.replace(" ", ""):
                        continue

                seen_items.add(item_name)
                # Use translated name if available
                display_name = translations.get(item_name, item_name)
                menu_items.append(display_name)
                dish_names.append(item_name)  # Keep original for translation detection

    if not menu_items:
        return None, []

    opening_hours = restaurant.get("opening_hours", "")
    price_info = ""
    if common_price and is_valid_price(common_price):
        price_info = f"💶 _{common_price}_"

    time_price_info = ""
    if opening_hours:
        time_price_info = f"⏰ {opening_hours}"
        if price_info:
            time_price_info += f" {price_info}"
    time_price_info = f"{time_price_info}\n" if time_price_info else ""

    menu_text = "\n• ".join(menu_items)

    formatted_menu = f"🍽️ *{name}{location_name}*\n{time_price_info}• {menu_text}\n"
    return formatted_menu, dish_names


async def send_message_chunks(bot: Bot, text: str, dry_run: bool = False) -> None:
    if not text:
        return

    chunks = clean_and_split(text)
    for chunk in chunks:
        try:
            if dry_run:
                log.info(f"[DRY RUN] Would send to Telegram: {chunk}")
            else:
                await bot.send_message(
                    chat_id=CHANNEL_ID, text=chunk, parse_mode="Markdown"
                )
                await asyncio.sleep(0.1)
        except TelegramError as e:
            log.error(f"Error sending message to Telegram: {str(e)}")
            log.error(f"Problematic chunk: {chunk}")


async def build_and_post(dry_run: bool = False) -> None:
    bot = Bot(TELEGRAM_BOT_TOKEN)

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            restaurants = await fetch_menus(session)

            # Build candidates for fish filtering
            candidates = []
            restaurant_prices = {}

            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name:
                    continue

                items = restaurant.get("items", [])
                common_price = get_most_common_price(items)
                restaurant_prices[name] = common_price

                item_index = 0
                for group in items:
                    for item in group:
                        item_name = item.get("name", "").strip()
                        if not item_name:
                            continue

                        diets = item.get("diets", [])
                        price = item.get("price", "").strip()
                        ingredients = item.get("ingredients", "").strip()

                        # Only consider non-veg items with matching price
                        if not is_veg(diets) and common_price:
                            if price.replace(" ", "") == common_price.replace(" ", ""):
                                candidates.append(
                                    {
                                        "id": f"{name}|{item_index}",
                                        "restaurant": name,
                                        "name": item_name,
                                        "diets": diets,
                                        "ingredients": ingredients,
                                    }
                                )
                        item_index += 1

            # Filter fish-only dishes
            id_to_allow = await filter_fish_only(session, candidates)

            # Build allowed fish set by restaurant
            allowed_fish_by_restaurant = defaultdict(set)
            for candidate in candidates:
                if id_to_allow.get(candidate["id"], False):
                    allowed_fish_by_restaurant[candidate["restaurant"]].add(
                        candidate["name"]
                    )

            # First pass: collect all dishes by restaurant for translation detection
            dishes_by_restaurant = {}
            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name:
                    continue

                allowed_fish = allowed_fish_by_restaurant[name]
                _, dish_names = await format_restaurant_menu(
                    restaurant, allowed_fish, session
                )
                if dish_names:
                    dishes_by_restaurant[name] = dish_names

            # Translate dishes if needed
            translations = await translate_dishes(session, dishes_by_restaurant)

            # Second pass: format menus with translations
            menu_parts = []
            all_dishes = []

            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name:
                    continue

                allowed_fish = allowed_fish_by_restaurant[name]
                menu, dish_names = await format_restaurant_menu(
                    restaurant, allowed_fish, session, translations
                )

                if menu:
                    menu_parts.append(menu)
                    menu_parts.append("➖" * 5 + "\n")

                    # Collect dishes for chef's choice (use translated names)
                    items = restaurant.get("items", [])
                    for group in items:
                        for item in group:
                            item_name = item.get("name", "").strip()
                            if item_name and (
                                is_veg(item.get("diets", []))
                                or item_name in allowed_fish
                            ):
                                display_name = translations.get(item_name, item_name)
                                all_dishes.append((display_name, name))

            if menu_parts:
                current_date = datetime.date.today()
                formatted_date = current_date.strftime("%A, %B %d")
                header = f"🌱🐟 *Halal Menu for {formatted_date}*\n\n"
                full_message = header + "".join(menu_parts)

                if not dry_run:
                    chefs_choice = await get_chefs_choice(session, all_dishes)
                    if chefs_choice:
                        full_message += "\n\n👨‍🍳 " + chefs_choice

                await send_message_chunks(bot, full_message, dry_run)
                log.info("Successfully posted Halal menu summary to channel")
            else:
                log.warning("No Halal options available for today")

    except Exception as e:
        log.error(f"Error in build_and_post: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post Halal-friendly daily menus to Telegram"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending messages to Telegram",
    )

    args = parser.parse_args()
    log.info(f"Starting Halal Food Bot (dry run: {args.dry_run})")
    asyncio.run(build_and_post(dry_run=args.dry_run))
