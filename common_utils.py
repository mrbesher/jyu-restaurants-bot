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
from retry_utils import retry_with_backoff

# Configuration
WEEKLY_API = "https://jybar.app.jyu.fi/api/2/lunches/weekly"
LLM_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_MODEL = "gemini-2.5-flash"
SKIP_RESTAURANTS = ["tilia", "normaalikoulu", "kvarkki", "bistro"]
NO_PRICE_LIMIT_EXCEPTIONS = ["Ilokivi"]  # Restaurants that can show all groups even without prices

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
logger.setLevel(logging.INFO)




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
    matches.sort(key=lambda p: float(p.replace(",", ".")))
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
async def fetch_menus_with_offset(session: aiohttp.ClientSession, day_offset: int = 0) -> List[Dict]:
    """Fetch menus from the weekly API for a specific day offset.

    Args:
        session: aiohttp session
        day_offset: Days relative to today (0=today, 1=tomorrow, -1=yesterday)

    Returns:
        List of restaurants in the same format as fetch_menus()
    """
    # Calculate target date
    from datetime import datetime, timedelta
    target_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y%m%d")
    logger.debug(f"Fetching menus for date offset {day_offset} (date: {target_date})")

    # Always fetch from weekly API
    async with session.get(WEEKLY_API) as response:
        logger.debug("Weekly API response status: %d", response.status)
        response.raise_for_status()
        data = await response.json()

        # Filter and convert data
        filtered_restaurants = []
        results = data.get("results", {}).get("en", [])

        for restaurant in results:
            # Find the lunch for our target date
            target_lunch = None
            for lunch in restaurant.get("lunches", []):
                if lunch.get("date") == target_date:
                    target_lunch = lunch
                    break

            if target_lunch:
                # Convert weekly format to daily format
                converted = {
                    "name": restaurant.get("title", ""),
                    "restaurant_id": restaurant.get("restaurant_id", ""),
                    "url": restaurant.get("url", ""),
                    "time": restaurant.get("time", ""),
                    "location": restaurant.get("location", {}),
                    "lang": restaurant.get("lang", "en"),
                    "items": []
                }

                # Convert items from weekly format to daily format
                for item in target_lunch.get("items", []):
                    # Weekly API has items[].comp[] structure
                    components = item.get("comp", [])
                    if components:
                        converted["items"].append(components)

                if converted["items"]:  # Only add if there are menu items
                    filtered_restaurants.append(converted)

        logger.debug(f"Found {len(filtered_restaurants)} restaurants with menus for {target_date}")

        # If no restaurants have menus for the requested date, return empty list
        if not filtered_restaurants:
            logger.warning(f"No restaurants found with menus for {target_date}")

        return filtered_restaurants


@retry_with_backoff()
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
You are selecting the tastiest dish group.
Provide a minimal description of what the dish is for those who don't know it.

Pick exactly ONE from this list:
{}

Return JSON example:
{{
    "dish": "Falafel group",
    "restaurant": "Restaurant",
    "reason": "A popular Middle Eastern dish made from ground chickpeas, seasoned with herbs and spices, then deep-fried to golden perfection."
}}
""".format("\n".join(lines))


def build_halal_chefs_choice_prompt(lines: List[str]) -> str:
    """Build halal-specific prompt for selecting chef's choice with prioritized cuisines."""
    return """
You are selecting the tastiest dish group for today's halal-friendly recommendation.

PRIORITY RULES:
1. HIGH PRIORITY: Pizza dishes - always prefer these if available
2. HIGH PRIORITY: Fish and seafood dishes - these are also highly preferred
3. HIGH PRIORITY: Middle Eastern dishes (falafel, hummus, shawarma, kebab, etc.)
4. Choose from other dishes only if no options from above are available

SELECTION GUIDELINES:
- Select exactly ONE MAIN DISH GROUP from the list below
- If there's a compatible side dish (salad, fries, rice, etc.) that pairs well with your chosen main dish, you can mention it
- Focus on dishes that would appeal to most people
- Consider both taste and visual appeal

Provide a minimal description of what the dish is for those who don't know it.

Available dishes:
{}

Return JSON example:
{{
    "dish": "Pizza Margherita group",
    "restaurant": "Restaurant",
    "reason": "Classic Italian pizza with fresh mozzarella, tomatoes, and basil on a crispy thin crust. A crowd favorite that never disappoints."
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


async def get_halal_chefs_choice(
    session: aiohttp.ClientSession, dishes: List[Tuple[str, str]]
) -> str:
    """Get halal chef's choice recommendation from AI with prioritized cuisines."""
    if not dishes:
        return ""

    lines = [f"{name} @ {rest}" for name, rest in dishes]
    prompt = build_halal_chefs_choice_prompt(lines)

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
        logger.error("Error getting halal chef's choice: %s", e)

    return ""


def has_non_english_chars(text: str) -> bool:
    """Check if text contains non-English characters."""
    result = bool(re.search(r"[^\x00-\x7F]", text))
    logger.debug("Checking '%s' for non-English chars: %s", text, result)
    return result


async def translate_dishes(
    session: aiohttp.ClientSession, dishes_by_restaurant: Dict[str, List[str]]
) -> Dict[str, str]:
    """Translate non-English dish names to English."""
    all_dishes_to_translate = []

    for restaurant, dishes in dishes_by_restaurant.items():
        has_non_english = any(has_non_english_chars(dish) for dish in dishes)
        if has_non_english:
            logger.info(
                f"Restaurant {restaurant} has non-English dishes: {[d for d in dishes if has_non_english_chars(d)]}"
            )
            all_dishes_to_translate.extend(dishes)

    if not all_dishes_to_translate:
        logger.info("No dishes need translation")
        return {}

    unique_dishes = list(dict.fromkeys(all_dishes_to_translate))
    logger.info(f"Translating {len(unique_dishes)} unique dishes: {unique_dishes}")

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
        logger.info(f"LLM response: {content}")

        obj = json.loads(content)
        translations = {}

        for item in obj.get("translations", []):
            original = item.get("original", "")
            translated = item.get("translated", "")
            if original and translated:
                translations[original] = translated
                logger.info(f"Translation: '{original}' -> '{translated}'")

        logger.info(f"Total translations: {len(translations)}")
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
) -> Tuple[Optional[str], List[Tuple[str, List[str]]]]:
    """Format a restaurant's menu items and return both formatted text and list of groups with dishes."""
    name = restaurant.get("name", "").strip()
    logger.debug("Formatting menu for restaurant: %s", name)
    location = restaurant.get("location", {})
    items = restaurant.get("items", [])
    translations = translations or {}
    logger.debug("Restaurant %s has %d item groups", name, len(items))

    if not name or not items or should_skip_restaurant(name):
        logger.debug("Skipping restaurant %s (no name, no items, or in skip list)", name)
        return None, []

    # Get location name
    location_name = ""
    if location.get("lat") and location.get("lon"):
        osm_location = await get_location_name(
            location["lat"], location["lon"], session
        )
        if osm_location:
            location_name = f" ({osm_location})"

    # Check if restaurant has prices and limit groups if needed
    common_price = get_common_price(items)
    if not common_price and name not in NO_PRICE_LIMIT_EXCEPTIONS:
        items = items[:4]
        logger.debug("No prices found, limiting to first 4 groups for %s", name)

    # Process menu items by groups
    seen_items = set()
    menu_groups = []
    group_data = []
    group_index = 0

    for item_group in items:
        logger.debug("Processing item group with %d items", len(item_group))
        group_items = []
        group_dishes = []

        # Check if this group has any items that pass the filter
        group_has_items = False

        for item in item_group:
            item_name = item.get("name", "").strip()
            if not item_name or item_name in seen_items:
                continue

            item_diets = item.get("diets", [])
            item_price = item.get("price", "").strip()
            logger.debug("Processing item: '%s', diets: %s, price: %s", item_name, item_diets, item_price)

            # Filter by price if common price exists
            if common_price:
                item_prices = extract_prices(item_price)
                if item_prices and item_prices != common_price:
                    continue

            # Apply custom filter function
            if filter_func(item, item_diets, item_name):
                seen_items.add(item_name)
                display_name = translations.get(item_name, item_name)
                if display_name != item_name:
                    logger.debug("Applying translation: '%s' -> '%s'", item_name, display_name)
                group_items.append(display_name)
                group_dishes.append(item_name)
                group_has_items = True

        # Add the group if it has items
        if group_has_items:
            group_text = " + ".join(group_items)
            menu_groups.append(group_text)
            group_data.append((group_items[0] if group_items else "", group_dishes))
            group_index += 1

    if not menu_groups:
        logger.debug("No menu items for restaurant %s after filtering", name)
        return None, []

    # Format opening hours and price
    opening_hours = restaurant.get("time", "")
    price_info = ""
    if common_price:
        price_info = f"💶 _{' / '.join(common_price)}_"

    time_price_info = ""
    if opening_hours:
        time_price_info = f"⏰ {opening_hours}"
    if price_info:
        if time_price_info:
            time_price_info += f" {price_info}"
        else:
            time_price_info = price_info
    time_price_info = f"{time_price_info}\n" if time_price_info else ""

    # Format menu text with groups
    menu_text = "\n• ".join(menu_groups)
    formatted_menu = f"🍽️ *{name}{location_name}*\n{time_price_info}• {menu_text}\n"
    logger.debug("Formatted menu for %s with %d groups", name, len(menu_groups))

    return formatted_menu, group_data


@retry_with_backoff()
async def send_single_message(bot: Bot, channel_id: str, chunk: str) -> None:
    """Send a single message chunk to Telegram with retry."""
    await bot.send_message(
        chat_id=channel_id, text=chunk, parse_mode="Markdown"
    )


async def send_message_chunks(
    bot: Bot, channel_id: str, text: str, dry_run: bool = False
) -> None:
    """Safely send message in chunks to Telegram."""
    if not text:
        return

    chunks = clean_and_split(text)
    for chunk in chunks:
        if dry_run:
            logger.info(f"[DRY RUN] Would send to Telegram: {chunk}")
        else:
            try:
                await send_single_message(bot, channel_id, chunk)
                await asyncio.sleep(0.1)
            except TelegramError as e:
                logger.error(f"Failed to send chunk after all retries: {chunk}")
                logger.error(f"Final error: {str(e)}")


# High-level processing functions
async def process_restaurants_for_diet(
    session: aiohttp.ClientSession, diets: Set[str], day_offset: int = 0
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Process restaurants for specific dietary requirements."""
    restaurants = await fetch_menus_with_offset(session, day_offset)
    menu_parts = []
    all_dishes = []
    dishes_by_restaurant = {}

    # First pass: collect dishes and check which need translation
    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        # Create filter function for this diet
        def diet_filter(item, item_diets, item_name):
            return all(
                normalize_diet(diet) in {normalize_diet(d) for d in item_diets}
                for diet in diets
            )

        menu, group_data = await format_restaurant_menu(
            restaurant, diet_filter, session
        )

        if group_data:
            # Collect dishes for translation
            all_dishes_in_restaurant = []
            for _, dishes in group_data:
                all_dishes_in_restaurant.extend(dishes)
            dishes_by_restaurant[name] = all_dishes_in_restaurant

            # Collect for menu parts (will be rebuilt with translations)
            if menu:
                menu_parts.append(menu)
                menu_parts.append("➖" * 5 + "\n")

            # Add groups for chef's choice
            for first_dish, dishes in group_data:
                group_name = f"{first_dish} group"
                all_dishes.append((group_name, restaurant.get("name", "")))

    # Get translations for non-English dishes
    translations = await translate_dishes(session, dishes_by_restaurant)

    # Update dish names with translations
    all_dishes = [
        (translations.get(dish_name, dish_name), restaurant)
        for dish_name, restaurant in all_dishes
    ]

    # Rebuild menu with translations
    menu_parts_translated = []
    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        def diet_filter(item, item_diets, item_name):
            return all(
                normalize_diet(diet) in {normalize_diet(d) for d in item_diets}
                for diet in diets
            )

        menu, _ = await format_restaurant_menu(
            restaurant, diet_filter, session, translations
        )

        if menu:
            menu_parts_translated.append(menu)
            menu_parts_translated.append("➖" * 5 + "\n")

    return menu_parts_translated, all_dishes


async def process_restaurants_for_halal(
    session: aiohttp.ClientSession, day_offset: int = 0
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, Set[str]]]:
    """Process restaurants for halal requirements (veg + fish)."""
    restaurants = await fetch_menus_with_offset(session, day_offset)

    # Collect candidates for fish filtering at group level
    candidates = []
    valid_restaurants = []
    group_candidates = []  # Track which groups contain non-veg items

    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        valid_restaurants.append(restaurant)
        items = restaurant.get("items", [])
        common_price = get_common_price(items)

        group_index = 0
        for group in items:
            group_has_non_veg = False
            group_items = []
            item_index = 0

            for item in group:
                item_name = item.get("name", "").strip()
                if not item_name:
                    continue

                diets = item.get("diets", [])
                price = item.get("price", "").strip()

                group_items.append({
                    "name": item_name,
                    "diets": diets,
                    "price": price
                })

                if not is_veg(diets) and (
                    not common_price or extract_prices(price) == common_price
                ):
                    group_has_non_veg = True
                    candidates.append(
                        {
                            "id": f"{name}|{group_index}|{item_index}",
                            "restaurant": name,
                            "name": item_name,
                            "diets": diets,
                            "ingredients": item.get("ingredients", "").strip(),
                        }
                    )
                item_index += 1

            # Track if this group has any non-veg items
            if group_has_non_veg:
                group_candidates.append({
                    "restaurant": name,
                    "group_index": group_index,
                    "items": group_items
                })

            group_index += 1

    # Filter fish dishes
    id_to_allow = await filter_fish_only(session, candidates)
    allowed_fish_by_restaurant = defaultdict(set)
    allowed_groups_by_restaurant = defaultdict(set)

    for candidate in candidates:
        if id_to_allow.get(candidate["id"], False):
            allowed_fish_by_restaurant[candidate["restaurant"]].add(candidate["name"])
            # Extract group index from id
            group_idx = int(candidate["id"].split("|")[1])
            allowed_groups_by_restaurant[candidate["restaurant"]].add(group_idx)

    # First pass: collect dishes for translation
    dishes_by_restaurant = {}
    for restaurant in valid_restaurants:
        name = restaurant.get("name", "").strip()
        allowed_fish = allowed_fish_by_restaurant[name]
        allowed_groups = allowed_groups_by_restaurant[name]

        # Create filter function for halal
        def halal_filter(item, item_diets, item_name):
            return is_veg(item_diets) or item_name in allowed_fish

        _, group_data = await format_restaurant_menu(
            restaurant, halal_filter, session
        )

        if group_data:
            # Collect dishes for translation
            all_dishes_in_restaurant = []
            for _, dishes in group_data:
                all_dishes_in_restaurant.extend(dishes)
            dishes_by_restaurant[name] = all_dishes_in_restaurant

    # Get translations
    translations = await translate_dishes(session, dishes_by_restaurant)

    # Process restaurants with allowed fish and translations
    menu_parts = []
    all_dishes = []

    for restaurant in valid_restaurants:
        name = restaurant.get("name", "").strip()
        allowed_fish = allowed_fish_by_restaurant[name]

        # Create filter function for halal
        def halal_filter(item, item_diets, item_name):
            return is_veg(item_diets) or item_name in allowed_fish

        menu, group_data = await format_restaurant_menu(
            restaurant, halal_filter, session, translations
        )

        if group_data:
            # Collect for translation
            if menu:
                menu_parts.append(menu)
                menu_parts.append("➖" * 5 + "\n")

            # Add groups for chef's choice (with translations)
            for first_dish, dishes in group_data:
                translated_first = translations.get(first_dish, first_dish)
                all_dishes.append((f"{translated_first} group", name))

    return menu_parts, all_dishes, allowed_fish_by_restaurant
