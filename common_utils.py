#!/usr/bin/env python3
import asyncio
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

# Configuration
LUNCHES_API = "https://jybar.app.jyu.fi/api/2/lunches"
LLM_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_MODEL = "gemini-2.5-flash"
SKIP_RESTAURANTS = ["tilia", "normaalikoulu", "kvarkki", "bistro"]

# Environment variables
TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Schemas
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

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator with exponential backoff."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    continue
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    return decorator


# Restaurant and filtering utilities
def normalize_diet(diet: str) -> str:
    """Normalize diet names to handle different variations."""
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


def should_skip_restaurant(restaurant_name: str) -> bool:
    """Check if restaurant should be skipped based on SKIP_RESTAURANTS list."""
    if not restaurant_name:
        return True

    name_lower = restaurant_name.lower()
    return any(skip_name in name_lower for skip_name in SKIP_RESTAURANTS)


def is_veg(diets: List[str]) -> bool:
    """Check if dish is vegetarian."""
    return any(normalize_diet(d) == "VEG" for d in diets)


def extract_prices(price_string: str) -> List[str]:
    """Extract all valid prices from a price string and sort them cheapest to most expensive."""
    price_pattern = r"\d{1,2}[\.,]\d{2}"
    matches = re.findall(price_pattern, price_string)
    # Sort prices numerically (cheapest first)
    matches.sort(key=lambda p: float(p.replace(',', '.')))
    return matches


def get_common_price(items: List[List[Dict]]) -> List[str]:
    """Find the most common price list across all dishes.
    Returns the price list tuple that appears most frequently."""
    price_list_counts = Counter()

    for item_group in items:
        for item in item_group:
            price_str = item.get("price", "").strip()
            if price_str:
                prices = extract_prices(price_str)
                if prices:
                    # Convert to tuple for counting
                    price_tuple = tuple(prices)
                    price_list_counts[price_tuple] += 1

    if not price_list_counts:
        return []

    # Find the most common price list
    most_common_tuple = price_list_counts.most_common(1)[0][0]
    
    # Convert back to list
    return list(most_common_tuple)


# API functions
@retry_with_backoff()
async def fetch_menus(session: aiohttp.ClientSession) -> List[Dict]:
    """Fetch menus from the JYU API."""
    async with session.get(LUNCHES_API) as response:
        response.raise_for_status()
        data = await response.json()
        return data.get("results", {}).get("en", [])


@retry_with_backoff(max_retries=2)
async def get_location_name(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> str:
    """Get location name from coordinates using OpenStreetMap."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            address = data.get("address", {})
            return address.get("suburb") or address.get("neighbourhood") or ""
    return ""


@retry_with_backoff()
async def llm_chat_json(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    temperature: float = 0.0,
) -> str:
    """Send a chat request to LLM with JSON schema response format."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
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


# AI-powered features
def build_fish_filter_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Build prompt for filtering fish dishes."""
    diets_legend = "(G) Gluten-free, (L) Lactose-free, (VL) Low lactose, (M) Dairy-free, (Veg) Suitable for vegans."
    return f"""
You are filtering dishes for a Halal-friendly list.

ALLOW only dishes that contain dairy/eggs, fish or seafood and no other meat (e.g. chicken, beef, pork).
If ingredients missing, deduce from name.
Diets legend: {diets_legend}

Input dishes:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

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
    """Build prompt for selecting chef's choice."""
    return """
You are selecting the tastiest dish.
Provide a minimal description of what the dish is for those who don't know it.

Pick exactly ONE from this list:
{}

Return JSON example:
{{
    "dish": "Falafel",
    "restaurant": "Restaurant",
    "reason": "A popular Middle Eastern dish made from ground chickpeas, seasoned with herbs and spices, then deep-fried to golden perfection."
}}
""".format("\n".join(lines))


async def filter_fish_only(
    session: aiohttp.ClientSession, candidates: List[Dict]
) -> Dict[str, bool]:
    """Filter dishes to only include fish/seafood using AI."""
    if not candidates:
        return {}

    prompt = build_fish_filter_prompt(candidates)
    content = await llm_chat_json(
        session, [{"role": "user", "content": prompt}], FISH_FILTER_SCHEMA
    )

    arr = json.loads(content)
    result = {}
    candidate_ids = {c["id"] for c in candidates}

    for obj in arr:
        if isinstance(obj.get("id"), str) and obj["id"] in candidate_ids:
            result[obj["id"]] = bool(obj.get("allow", False))

    return result


async def get_chefs_choice(
    session: aiohttp.ClientSession, dishes: List[Tuple[str, str]]
) -> str:
    """Get chef's choice recommendation from AI."""
    if not dishes:
        return ""

    lines = [f"{name} @ {rest}" for name, rest in dishes]
    prompt = build_chefs_choice_prompt(lines)

    try:
        content = await llm_chat_json(
            session, [{"role": "user", "content": prompt}], CHEFS_CHOICE_SCHEMA, 0.2
        )

        obj = json.loads(content)
        dish = obj.get("dish", "").strip()
        rest = obj.get("restaurant", "").strip()
        reason = obj.get("reason", "").strip()

        if dish and rest and reason:
            return f"*{dish}* @ _{rest}_\n💬 _{reason}_"
    except Exception as e:
        logger.error("Error getting chef's choice: %s", e)

    return ""


def has_non_english_chars(text: str) -> bool:
    """Check if text contains non-English characters."""
    return bool(re.search(r"[^\x00-\x7F]", text))


async def translate_dishes(
    session: aiohttp.ClientSession, dishes_by_restaurant: Dict[str, List[str]]
) -> Dict[str, str]:
    """Translate non-English dish names to English."""
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

        obj = json.loads(content)
        translations = {}

        for item in obj.get("translations", []):
            original = item.get("original", "")
            translated = item.get("translated", "")
            if original and translated:
                translations[original] = translated

        return translations
    except Exception as e:
        logger.error(f"Error translating dishes: {e}")
        return {}


# Formatting and sending functions
async def format_restaurant_menu(
    restaurant: Dict,
    filter_func,
    session: aiohttp.ClientSession,
    translations: Dict[str, str] = None,
) -> Tuple[Optional[str], List[str]]:
    """Format a restaurant's menu items and return both formatted text and list of dishes."""
    name = restaurant.get("name", "").strip()
    location = restaurant.get("location", {})
    items = restaurant.get("items", [])
    translations = translations or {}

    if not name or not items or should_skip_restaurant(name):
        return None, []

    # Get location name
    location_name = ""
    if location.get("lat") and location.get("lon"):
        osm_location = await get_location_name(
            location["lat"], location["lon"], session
        )
        if osm_location:
            location_name = f" ({osm_location})"

    # Process menu items
    seen_items = set()
    menu_items = []
    dish_names = []
    common_price = get_common_price(items)

    for item_group in items:
        for item in item_group:
            item_name = item.get("name", "").strip()
            if not item_name or item_name in seen_items:
                continue

            item_diets = item.get("diets", [])
            item_price = item.get("price", "").strip()

            # Filter by price if common price exists
            if common_price:
                item_prices = extract_prices(item_price)
                if item_prices and item_prices != common_price:
                    continue

            # Apply custom filter function
            if filter_func(item, item_diets, item_name):
                seen_items.add(item_name)
                display_name = translations.get(item_name, item_name)
                menu_items.append(display_name)
                dish_names.append(item_name)

    if not menu_items:
        return None, []

    # Format opening hours and price
    opening_hours = restaurant.get("opening_hours", "")
    price_info = ""
    if common_price:
        price_info = f"💶 _{' / '.join(common_price)}_"

    time_price_info = ""
    if opening_hours:
        time_price_info = f"⏰ {opening_hours}"
        if price_info:
            time_price_info += f" {price_info}"
    time_price_info = f"{time_price_info}\n" if time_price_info else ""

    # Format menu text
    menu_text = "\n• ".join(menu_items)
    formatted_menu = f"🍽️ *{name}{location_name}*\n{time_price_info}• {menu_text}\n"

    return formatted_menu, dish_names


async def send_message_chunks(
    bot: Bot, channel_id: str, text: str, dry_run: bool = False
) -> None:
    """Safely send message in chunks to Telegram."""
    if not text:
        return

    chunks = clean_and_split(text)
    for chunk in chunks:
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would send to Telegram: {chunk}")
            else:
                await bot.send_message(
                    chat_id=channel_id, text=chunk, parse_mode="Markdown"
                )
                await asyncio.sleep(0.1)
        except TelegramError as e:
            logger.error(f"Error sending message to Telegram: {str(e)}")
            if "retry after" in str(e).lower():
                retry_after = int("".join(filter(str.isdigit, str(e))))
                await asyncio.sleep(min(retry_after, 30))
                try:
                    await bot.send_message(
                        chat_id=channel_id, text=chunk, parse_mode="Markdown"
                    )
                except TelegramError:
                    logger.error(f"Failed to send chunk after retry: {chunk}")
            else:
                logger.error(f"Problematic chunk: {chunk}")


# High-level processing functions
async def process_restaurants_for_diet(
    session: aiohttp.ClientSession, diets: Set[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Process restaurants for specific dietary requirements."""
    restaurants = await fetch_menus(session)
    menu_parts = []
    all_dishes = []

    for restaurant in restaurants:
        # Create filter function for this diet
        def diet_filter(item, item_diets, item_name):
            return all(
                normalize_diet(diet) in {normalize_diet(d) for d in item_diets}
                for diet in diets
            )

        menu, dish_names = await format_restaurant_menu(
            restaurant, diet_filter, session
        )

        if menu:
            menu_parts.append(menu)
            menu_parts.append("➖" * 5 + "\n")

            # Add dishes for chef's choice
            for dish_name in dish_names:
                all_dishes.append((dish_name, restaurant.get("name", "")))

    return menu_parts, all_dishes


async def process_restaurants_for_halal(
    session: aiohttp.ClientSession,
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, Set[str]]]:
    """Process restaurants for halal requirements (veg + fish)."""
    restaurants = await fetch_menus(session)

    # Collect candidates for fish filtering
    candidates = []
    valid_restaurants = []

    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        valid_restaurants.append(restaurant)
        items = restaurant.get("items", [])
        common_price = get_common_price(items)

        item_index = 0
        for group in items:
            for item in group:
                item_name = item.get("name", "").strip()
                if not item_name:
                    continue

                diets = item.get("diets", [])
                price = item.get("price", "").strip()

                if not is_veg(diets) and (
                    not common_price
                    or extract_prices(price) == common_price
                ):
                    candidates.append(
                        {
                            "id": f"{name}|{item_index}",
                            "restaurant": name,
                            "name": item_name,
                            "diets": diets,
                            "ingredients": item.get("ingredients", "").strip(),
                        }
                    )
                item_index += 1

    # Filter fish dishes
    id_to_allow = await filter_fish_only(session, candidates)
    allowed_fish_by_restaurant = defaultdict(set)
    for candidate in candidates:
        if id_to_allow.get(candidate["id"], False):
            allowed_fish_by_restaurant[candidate["restaurant"]].add(candidate["name"])

    # Process restaurants with allowed fish
    menu_parts = []
    all_dishes = []

    for restaurant in valid_restaurants:
        name = restaurant.get("name", "").strip()
        allowed_fish = allowed_fish_by_restaurant[name]

        # Create filter function for halal
        def halal_filter(item, item_diets, item_name):
            return is_veg(item_diets) or item_name in allowed_fish

        menu, dish_names = await format_restaurant_menu(
            restaurant, halal_filter, session
        )

        if dish_names:
            # Collect for translation
            if menu:
                menu_parts.append(menu)
                menu_parts.append("➖" * 5 + "\n")

            # Add dishes for chef's choice
            for dish_name in dish_names:
                all_dishes.append((dish_name, name))

    return menu_parts, all_dishes, allowed_fish_by_restaurant
