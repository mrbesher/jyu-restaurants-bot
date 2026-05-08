#!/usr/bin/env python3
import asyncio
import datetime
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

from md_utils import clean_and_split
from retry_utils import retry_with_backoff

WEEKLY_API = "https://jybar.app.jyu.fi/api/2/lunches/weekly"
LLM_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_MODEL = "gemini-3-flash-preview"
SKIP_RESTAURANTS = {"tilia", "normaalikoulu", "kvarkki", "bistro"}
NO_PRICE_LIMIT_EXCEPTIONS = {"Ilokivi"}

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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

WEEKLY_PICKS_SCHEMA = {
    "name": "weekly_picks_response",
    "strict": "true",
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "day": {"type": "string"},
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dish": {"type": "string"},
                            "restaurant": {"type": "string"},
                            "kind": {"type": "string"},
                        },
                        "required": ["dish", "restaurant", "kind"],
                    },
                },
            },
            "required": ["day", "picks"],
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


class FileCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load cache %s: %s", self.path, e)
                self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Failed to save cache %s: %s", self.path, e)


_DIET_MAPPING = {
    "VEG": {"VEG", "VEGAN", "VEGAANI", "VEGAANINEN", "VEGETAARINEN", "VEGETARIAN"},
    "L": {"L", "LAKTOOSITON"},
    "G": {"G", "GLUTEENITON"},
    "M": {"M", "MAIDOTON"},
}


def normalize_diet(diet: str) -> str:
    upper = diet.upper()
    for normalized, variations in _DIET_MAPPING.items():
        if any(upper.startswith(v) for v in variations):
            return normalized
    return upper


def should_skip_restaurant(restaurant_name: str) -> bool:
    if not restaurant_name:
        return True
    name_lower = restaurant_name.lower()
    return any(skip_name in name_lower for skip_name in SKIP_RESTAURANTS)


def is_veg(diets: list[str]) -> bool:
    return any(normalize_diet(d) == "VEG" for d in diets)


def extract_prices(price_string: str) -> list[str]:
    matches = re.findall(r"\d{1,2}[.,]\d{2}", price_string)
    matches.sort(key=lambda p: float(p.replace(",", ".")))
    return matches


def get_common_price(items: list[list[dict]]) -> list[str] | None:
    counts = Counter()
    for group in items:
        for item in group:
            price_str = item.get("price", "").strip()
            if price_str:
                prices = tuple(extract_prices(price_str))
                if prices:
                    counts[prices] += 1
    if not counts:
        return None
    return list(counts.most_common(1)[0][0])


@retry_with_backoff()
async def fetch_menus_with_offset(
    session: aiohttp.ClientSession, day_offset: int = 0
) -> list[dict]:
    from datetime import datetime, timedelta

    target_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y%m%d")
    logger.debug("Fetching menus for date offset %d (%s)", day_offset, target_date)

    async with session.get(WEEKLY_API) as response:
        response.raise_for_status()
        data = await response.json()

    results = data.get("results", {}).get("en", [])
    filtered = []

    for restaurant in results:
        target_lunch = None
        for lunch in restaurant.get("lunches", []):
            if lunch.get("date") == target_date:
                target_lunch = lunch
                break

        if not target_lunch:
            continue

        converted = {
            "name": restaurant.get("title", ""),
            "restaurant_id": restaurant.get("restaurant_id", ""),
            "url": restaurant.get("url", ""),
            "time": restaurant.get("time", ""),
            "location": restaurant.get("location", {}),
            "lang": restaurant.get("lang", "en"),
            "items": [],
        }

        for item in target_lunch.get("items", []):
            components = item.get("comp", [])
            if components:
                converted["items"].append(components)

        if converted["items"]:
            filtered.append(converted)

    logger.debug("Found %d restaurants with menus for %s", len(filtered), target_date)
    return filtered


_LOCATION_OVERRIDES: dict[str, str] = {}


@retry_with_backoff()
async def get_location_name(
    restaurant_name: str,
    location: dict[str, Any],
    session: aiohttp.ClientSession,
    cache: FileCache,
) -> str:
    override = _LOCATION_OVERRIDES.get(restaurant_name.lower())
    if override is not None:
        return override

    lat = location.get("lat")
    lon = location.get("lon")
    if not lat or not lon:
        return ""

    key = f"{lat},{lon}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            address = data.get("address", {})
            result = address.get("suburb") or address.get("neighbourhood") or ""
            cache.set(key, result)
            return result

    cache.set(key, "")
    return ""


@retry_with_backoff()
async def llm_chat_json(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    temperature: float = 0.0,
) -> str:
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
        "reasoning_effort": "low",
    }

    async with session.post(LLM_CHAT_URL, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


def build_fish_filter_prompt(candidates: list[dict[str, Any]]) -> str:
    diets_legend = (
        "(G) Gluten-free, (L) Lactose-free, (VL) Low lactose, "
        "(M) Dairy-free, (Veg) Suitable for vegans."
    )
    return f"""You are filtering dishes for a Halal-friendly list.

ALLOW only dishes that contain dairy/eggs, fish or seafood and no other meat (e.g. chicken, beef, pork).
If ingredients missing, deduce from name.
Diets legend: {diets_legend}

Input dishes:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Output Example:
[
    {{"id": "Restaurant1|2", "name": "Dish name", "allow": false}},
    {{"id": "Restaurant2|2", "name": "Dish name", "allow": false}}
]
"""


def build_chefs_choice_prompt(lines: list[str]) -> str:
    dishes = "\n".join(lines)
    return f"""You are selecting the tastiest dish.
Each item represents a grouping of dishes (e.g., same dish with different sides), referred to by the main dish name.

Pick exactly ONE from this list:
{dishes}

Return JSON:
- "reason": very minimal short description of the dish (2-5 words)

Example:
{{
    "dish": "Falafel",
    "restaurant": "Restaurant",
    "reason": "Crispy chickpea patties with herbs"
}}
"""


def build_halal_chefs_choice_prompt(lines: list[str]) -> str:
    dishes = "\n".join(lines)
    return f"""You are selecting the tastiest dish for today's halal-friendly recommendation.
Each item represents a grouping of dishes (e.g., same dish with different sides), referred to by the main dish name.

PRIORITY RULES:
1. HIGH PRIORITY: Pizza dishes - always prefer these if available
2. HIGH PRIORITY: Fish and seafood dishes - these are also highly preferred
3. HIGH PRIORITY: Middle Eastern dishes (falafel, hummus, shawarma, kebab, etc.)
4. Choose from other dishes only if no options from above are available

SELECTION GUIDELINES:
- Select exactly ONE MAIN DISH from the list below
- If there's a compatible side dish (salad, fries, rice, etc.) that pairs well with your chosen main dish, you can mention it
- Focus on dishes that would appeal to most people
- Consider both taste and visual appeal

Available dishes:
{dishes}

Return JSON:
- "reason": very minimal short description of the dish (2-5 words)

Example:
{{
    "dish": "Pizza Margherita",
    "restaurant": "Restaurant",
    "reason": "Classic tomato and mozzarella pizza"
}}
"""


def build_weekly_picks_prompt(per_day: dict[str, list[dict]]) -> str:
    payload = json.dumps(per_day, ensure_ascii=False, indent=2)
    return f"""You are picking the top 2 halal-friendly highlights for each upcoming weekday.

All dishes provided are already halal-friendly (vegetarian, fish, or seafood pizzas — no haram meat).

RULES:
1. PIZZA WINS ALWAYS. If a day has any pizza, prefer pizza picks over fish.
2. Pick at most 2 dishes per day. Use 1 if only 1 candidate is available.
3. Skip days with no candidates.
4. Translate Finnish dish names to natural English. If already English, return as-is.
5. "kind" must be either "pizza" or "fish" matching the input kind for the chosen dish.

Dishes per day:
{payload}

Return a JSON array of objects, one per non-empty day, each with "day" (echo the day key) and "picks".
"""


def build_translation_prompt(dishes: list[str]) -> str:
    return f"""Translate the following dish names to English. If a dish name is already in English, return it exactly as is. Keep translations concise and food-appropriate.

Dishes to translate:
{json.dumps(dishes, ensure_ascii=False)}

Return JSON with original and translated versions for each dish.
"""


_FISH_KEYWORDS = (
    "fish", "salmon", "tuna", "shrimp", "prawn", "cod", "haddock", "halibut",
    "mackerel", "trout", "herring", "perch", "pike", "whitefish",
    "seafood", "calamari", "squid", "mussel", "oyster", "crab",
    "lobster", "anchovy", "sardine", "plaice", "pangasius", "tilapia",
    "kala", "lohi", "tonnikala", "katkarapu", "siika", "ahven", "hauki",
    "kuha", "silakka", "silli", "muikku", "made", "kirjolohi", "nieriä",
    "merenelävät",
)


def _looks_like_fish(item_name: str, ingredients: str) -> bool:
    text = f"{item_name} {ingredients}".lower()
    return any(kw in text for kw in _FISH_KEYWORDS)


_PIZZA_KEYWORDS = ("pizza", "pitsa")


def _looks_like_pizza(item_name: str) -> bool:
    name_lower = item_name.lower()
    return any(kw in name_lower for kw in _PIZZA_KEYWORDS)


async def filter_fish_only(
    session: aiohttp.ClientSession, candidates: list[dict[str, Any]]
) -> dict[str, bool]:
    if not candidates:
        return {}

    prompt = build_fish_filter_prompt(candidates)
    content = await llm_chat_json(
        session, [{"role": "user", "content": prompt}], FISH_FILTER_SCHEMA
    )

    result = {}
    candidate_ids = {c["id"] for c in candidates}

    try:
        for obj in json.loads(content):
            item_id = obj.get("id")
            if isinstance(item_id, str) and item_id in candidate_ids:
                result[item_id] = bool(obj.get("allow", False))
    except Exception as e:
        logger.error("Fish filter LLM returned invalid JSON: %s", e)

    return result


async def _get_chefs_choice(
    session: aiohttp.ClientSession,
    dishes: list[tuple[str, str]],
    prompt_builder,
    temperature: float = 0.2,
) -> str:
    if not dishes:
        return ""

    lines = [f"{name} @ {rest}" for name, rest in dishes]
    prompt = prompt_builder(lines)

    try:
        content = await llm_chat_json(
            session,
            [{"role": "user", "content": prompt}],
            CHEFS_CHOICE_SCHEMA,
            temperature,
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


async def get_chefs_choice(
    session: aiohttp.ClientSession, dishes: list[tuple[str, str]]
) -> str:
    return await _get_chefs_choice(session, dishes, build_chefs_choice_prompt)


async def get_halal_chefs_choice(
    session: aiohttp.ClientSession, dishes: list[tuple[str, str]]
) -> str:
    return await _get_chefs_choice(session, dishes, build_halal_chefs_choice_prompt)


def has_non_english_chars(text: str) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", text))


async def translate_dishes(
    session: aiohttp.ClientSession,
    dishes_by_restaurant: dict[str, list[str]],
    cache: FileCache,
) -> dict[str, str]:
    all_dishes = list(dict.fromkeys(
        d for dishes in dishes_by_restaurant.values() for d in dishes
    ))

    translations: dict[str, str] = {}
    uncached: list[str] = []
    cache_hits = 0

    for dish in all_dishes:
        cached = cache.get(dish)
        if cached is not None:
            translations[dish] = cached
            cache_hits += 1
        elif not has_non_english_chars(dish):
            translations[dish] = dish
        else:
            uncached.append(dish)

    if not uncached:
        logger.info(
            "All %d dishes resolved (%d cached, %d English)",
            len(translations), cache_hits, len(translations) - cache_hits,
        )
        return translations

    logger.info("Translating %d uncached dishes", len(uncached))

    prompt = build_translation_prompt(uncached)
    try:
        content = await llm_chat_json(
            session, [{"role": "user", "content": prompt}], TRANSLATION_SCHEMA
        )
        obj = json.loads(content)

        for item in obj.get("translations", []):
            original = item.get("original", "")
            translated = item.get("translated", "")
            if original and translated:
                translations[original] = translated
                cache.set(original, translated)
                logger.info("Translation: '%s' -> '%s'", original, translated)
    except Exception as e:
        logger.error("Error translating dishes: %s", e)

    return translations


def collect_filtered_dishes(
    restaurant: dict, filter_func
) -> tuple[list[str] | None, list[tuple[str, list[str]]]]:
    name = restaurant.get("name", "").strip()
    items = restaurant.get("items", [])

    if not name or not items or should_skip_restaurant(name):
        return None, []

    common_price = get_common_price(items)
    if not common_price and name not in NO_PRICE_LIMIT_EXCEPTIONS:
        items = items[:4]

    seen_items: set[str] = set()
    group_data: list[tuple[str, list[str]]] = []

    for item_group in items:
        group_dishes: list[str] = []

        for item in item_group:
            item_name = item.get("name", "").strip()
            if not item_name or item_name in seen_items:
                continue

            item_price = item.get("price", "").strip()
            if common_price:
                item_prices = extract_prices(item_price)
                if item_prices and item_prices != common_price:
                    continue

            if filter_func(item):
                seen_items.add(item_name)
                group_dishes.append(item_name)

        if group_dishes:
            group_data.append((group_dishes[0], group_dishes))

    return common_price, group_data


def format_restaurant_menu(
    restaurant: dict,
    common_price: list[str] | None,
    groups: list[tuple[str, list[str]]],
    location_name: str = "",
    translations: dict[str, str] | None = None,
) -> str | None:
    if not groups:
        return None

    name = restaurant.get("name", "").strip()
    translations = translations or {}

    menu_groups = []
    for _, dish_names in groups:
        display_names = [translations.get(d, d) for d in dish_names]
        menu_groups.append(" + ".join(display_names))

    opening_hours = restaurant.get("time", "")
    price_info = f"💶 _{' / '.join(common_price)}_" if common_price else ""

    time_price_info = ""
    if opening_hours:
        time_price_info = f"⏰ {opening_hours}"
    if price_info:
        time_price_info += f" {price_info}" if time_price_info else price_info
    time_price_info = f"{time_price_info}\n" if time_price_info else ""

    menu_text = "\n• ".join(menu_groups)
    return f"🍽️ *{name}{location_name}*\n{time_price_info}• {menu_text}\n"


@retry_with_backoff()
async def send_single_message(bot: Bot, channel_id: str, chunk: str) -> None:
    await bot.send_message(chat_id=channel_id, text=chunk, parse_mode="Markdown")


async def send_message_chunks(
    bot: Bot, channel_id: str, text: str, dry_run: bool = False
) -> None:
    if not text:
        return

    chunks = clean_and_split(text)
    for chunk in chunks:
        if dry_run:
            logger.info("[DRY RUN] Would send: %s", chunk)
            continue
        try:
            await send_single_message(bot, channel_id, chunk)
            await asyncio.sleep(0.1)
        except TelegramError as e:
            logger.error("Failed to send chunk after all retries: %s", chunk)
            logger.error("Final error: %s", e)


async def process_restaurants_for_diet(
    session: aiohttp.ClientSession, diets: set[str], day_offset: int = 0
) -> tuple[list[str], list[tuple[str, str]]]:
    restaurants = await fetch_menus_with_offset(session, day_offset)

    normalized_diets = {normalize_diet(d) for d in diets}

    def diet_filter(item: dict) -> bool:
        item_diets = item.get("diets", [])
        normalized_item_diets = {normalize_diet(d) for d in item_diets}
        return normalized_diets <= normalized_item_diets

    valid_restaurants: list[dict] = []
    common_prices: dict[str, list[str] | None] = {}
    all_group_data: dict[str, list[tuple[str, list[str]]]] = {}
    dishes_by_restaurant: dict[str, list[str]] = {}
    all_dishes: list[tuple[str, str]] = []

    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        valid_restaurants.append(restaurant)
        common_price, group_data = collect_filtered_dishes(restaurant, diet_filter)

        if group_data:
            common_prices[name] = common_price
            all_group_data[name] = group_data
            dishes_by_restaurant[name] = [
                d for _, dishes in group_data for d in dishes
            ]
            for first_dish, _ in group_data:
                all_dishes.append((first_dish, name))

    displayed_names = set(all_group_data.keys())
    restaurants_to_display = [
        r for r in valid_restaurants if r.get("name", "").strip() in displayed_names
    ]

    if not restaurants_to_display:
        return [], []

    location_cache = FileCache(Path("location_cache.json"))
    location_tasks = [
        get_location_name(r.get("name", "").strip(), r.get("location", {}), session, location_cache)
        for r in restaurants_to_display
    ]
    location_results = await asyncio.gather(*location_tasks)
    location_by_name = {
        r.get("name", "").strip(): loc
        for r, loc in zip(restaurants_to_display, location_results, strict=True)
    }
    location_cache.save()

    translation_cache = FileCache(Path("translation_cache.json"))
    translations = await translate_dishes(
        session, dishes_by_restaurant, translation_cache
    )
    translation_cache.save()

    all_dishes = [
        (translations.get(dish_name, dish_name), restaurant)
        for dish_name, restaurant in all_dishes
    ]

    menu_parts: list[str] = []
    for restaurant in restaurants_to_display:
        name = restaurant.get("name", "").strip()
        group_data = all_group_data[name]
        loc = location_by_name.get(name, "")
        location_str = f" ({loc})" if loc else ""

        menu = format_restaurant_menu(
            restaurant,
            common_prices.get(name),
            group_data,
            location_str,
            translations,
        )
        if menu:
            menu_parts.append(menu)
            menu_parts.append("➖" * 5 + "\n")

    return menu_parts, all_dishes


async def process_restaurants_for_halal(
    session: aiohttp.ClientSession, day_offset: int = 0
) -> tuple[list[str], list[tuple[str, str]], dict[str, set[str]]]:
    restaurants = await fetch_menus_with_offset(session, day_offset)

    candidates: list[dict] = []
    valid_restaurants: list[dict] = []
    allowed_fish_by_restaurant: dict[str, set[str]] = defaultdict(set)

    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        valid_restaurants.append(restaurant)
        items = restaurant.get("items", [])
        common_price = get_common_price(items)

        for group_index, group in enumerate(items):
            for item_index, item in enumerate(group):
                item_name = item.get("name", "").strip()
                if not item_name:
                    continue

                diets = item.get("diets", [])
                price = item.get("price", "").strip()

                if is_veg(diets):
                    continue

                if common_price and extract_prices(price) != common_price:
                    continue

                ingredients = item.get("ingredients", "").strip()
                if _looks_like_fish(item_name, ingredients):
                    allowed_fish_by_restaurant[name].add(item_name)
                    continue

                candidates.append(
                    {
                        "id": f"{name}|{group_index}|{item_index}",
                        "restaurant": name,
                        "name": item_name,
                        "diets": diets,
                        "ingredients": ingredients,
                    }
                )

    id_to_allow = await filter_fish_only(session, candidates)

    for candidate in candidates:
        if id_to_allow.get(candidate["id"], False):
            allowed_fish_by_restaurant[candidate["restaurant"]].add(candidate["name"])

    def make_halal_filter(restaurant_name: str):
        allowed = allowed_fish_by_restaurant.get(restaurant_name, set())

        def halal_filter(item: dict) -> bool:
            return is_veg(item.get("diets", [])) or item.get("name", "").strip() in allowed

        return halal_filter

    common_prices: dict[str, list[str] | None] = {}
    all_group_data: dict[str, list[tuple[str, list[str]]]] = {}
    dishes_by_restaurant: dict[str, list[str]] = {}
    all_dishes: list[tuple[str, str]] = []

    for restaurant in valid_restaurants:
        name = restaurant.get("name", "").strip()
        filt = make_halal_filter(name)
        common_price, group_data = collect_filtered_dishes(restaurant, filt)

        if group_data:
            common_prices[name] = common_price
            all_group_data[name] = group_data
            dishes_by_restaurant[name] = [
                d for _, dishes in group_data for d in dishes
            ]
            for first_dish, _ in group_data:
                all_dishes.append((first_dish, name))

    displayed_names = set(all_group_data.keys())
    restaurants_to_display = [
        r for r in valid_restaurants if r.get("name", "").strip() in displayed_names
    ]

    if not restaurants_to_display:
        return [], [], dict(allowed_fish_by_restaurant)

    location_cache = FileCache(Path("location_cache.json"))
    location_tasks = [
        get_location_name(r.get("name", "").strip(), r.get("location", {}), session, location_cache)
        for r in restaurants_to_display
    ]
    location_results = await asyncio.gather(*location_tasks, return_exceptions=True)
    location_by_name = {
        r.get("name", "").strip(): (loc if isinstance(loc, str) else "")
        for r, loc in zip(restaurants_to_display, location_results, strict=True)
    }
    location_cache.save()

    translation_cache = FileCache(Path("translation_cache.json"))
    translations = await translate_dishes(
        session, dishes_by_restaurant, translation_cache
    )
    translation_cache.save()

    all_dishes = [
        (translations.get(dish_name, dish_name), restaurant)
        for dish_name, restaurant in all_dishes
    ]

    menu_parts: list[str] = []
    for restaurant in restaurants_to_display:
        name = restaurant.get("name", "").strip()
        group_data = all_group_data[name]
        loc = location_by_name.get(name, "")
        location_str = f" ({loc})" if loc else ""

        menu = format_restaurant_menu(
            restaurant,
            common_prices.get(name),
            group_data,
            location_str,
            translations,
        )
        if menu:
            menu_parts.append(menu)
            menu_parts.append("➖" * 5 + "\n")

    return menu_parts, all_dishes, dict(allowed_fish_by_restaurant)


def _gather_pizza_fish_for_day(
    restaurants: list[dict], day_offset: int
) -> tuple[list[dict], list[dict]]:
    pre_allowed: list[dict] = []
    llm_candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for restaurant in restaurants:
        name = restaurant.get("name", "").strip()
        if not name or should_skip_restaurant(name):
            continue

        items = restaurant.get("items", [])
        common_price = get_common_price(items)

        for gi, group in enumerate(items):
            for ii, item in enumerate(group):
                item_name = item.get("name", "").strip()
                if not item_name:
                    continue

                key = (name, item_name)
                if key in seen:
                    continue

                price = item.get("price", "").strip()
                if common_price and extract_prices(price) and extract_prices(price) != common_price:
                    continue

                ingredients = item.get("ingredients", "").strip()
                is_pizza = _looks_like_pizza(item_name)
                is_fish = _looks_like_fish(item_name, ingredients)
                if not (is_pizza or is_fish):
                    continue

                seen.add(key)
                kind = "pizza" if is_pizza else "fish"
                diets = item.get("diets", [])

                if is_veg(diets) or is_fish:
                    pre_allowed.append(
                        {"restaurant": name, "name": item_name, "kind": kind}
                    )
                else:
                    llm_candidates.append({
                        "id": f"D{day_offset}|{name}|{gi}|{ii}",
                        "day_offset": day_offset,
                        "restaurant": name,
                        "name": item_name,
                        "diets": diets,
                        "ingredients": ingredients,
                        "kind": kind,
                    })

    return pre_allowed, llm_candidates


def _format_weekly_picks_message(parsed: list[dict]) -> str:
    lines = ["🍕🐟 *This Week: Pizza & Fish Halal Highlights*"]
    has_any = False
    for day_obj in parsed:
        picks = day_obj.get("picks") or []
        if not picks:
            continue
        has_any = True
        lines.append(f"\n*{day_obj.get('day', '')}*")
        for pick in picks:
            emoji = "🍕" if pick.get("kind") == "pizza" else "🐟"
            dish = pick.get("dish", "").strip()
            rest = pick.get("restaurant", "").strip()
            if dish and rest:
                lines.append(f"{emoji} *{dish}* @ _{rest}_")
    return "\n".join(lines) if has_any else ""


async def process_weekly_pizza_fish_picks(
    session: aiohttp.ClientSession,
) -> str:
    today = datetime.date.today()
    per_day_allowed: dict[int, list[dict]] = defaultdict(list)
    pooled_llm_candidates: list[dict] = []

    for offset in range(5):
        restaurants = await fetch_menus_with_offset(session, offset)
        pre_allowed, llm_candidates = _gather_pizza_fish_for_day(restaurants, offset)
        per_day_allowed[offset].extend(pre_allowed)
        pooled_llm_candidates.extend(llm_candidates)

    if pooled_llm_candidates:
        filter_input = [
            {k: c[k] for k in ("id", "restaurant", "name", "diets", "ingredients")}
            for c in pooled_llm_candidates
        ]
        id_to_allow = await filter_fish_only(session, filter_input)
        for c in pooled_llm_candidates:
            if id_to_allow.get(c["id"], False):
                per_day_allowed[c["day_offset"]].append(
                    {"restaurant": c["restaurant"], "name": c["name"], "kind": c["kind"]}
                )

    ranking_input: dict[str, list[dict]] = {}
    for offset, dishes in sorted(per_day_allowed.items()):
        if not dishes:
            continue
        day_label = (today + datetime.timedelta(days=offset)).strftime("%A, %B %d")
        ranking_input[day_label] = dishes

    if not ranking_input:
        return ""

    prompt = build_weekly_picks_prompt(ranking_input)
    try:
        content = await llm_chat_json(
            session,
            [{"role": "user", "content": prompt}],
            WEEKLY_PICKS_SCHEMA,
            temperature=0.2,
        )
        parsed = json.loads(content)
    except Exception as e:
        logger.error("Weekly picks LLM call failed: %s", e)
        return ""

    return _format_weekly_picks_message(parsed)
