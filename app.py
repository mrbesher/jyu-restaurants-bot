import argparse
import asyncio
import datetime
import logging
import os
import re
from typing import Dict, List, Optional, Set

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

from md_utils import clean_and_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHANNEL_ID = "@jyu_yliopiston_ravintolat"
LUNCHES_API = "https://jybar.app.jyu.fi/api/2/lunches"


async def get_location_name(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> str:
    """Get location name from coordinates using OpenStreetMap."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                address = data.get("address", {})
                return address.get("suburb") or address.get("neighbourhood") or ""
    except Exception as e:
        logger.error(f"Error fetching location name: {e}")
    return ""


def normalize_diet(diet: str) -> str:
    """Normalize diet names to handle different variations."""
    diet_mapping = {
        "VEG": {"VEG", "VEGAN", "VEGAANI", "VEGAANINEN"},
        "L": {"L", "LAKTOOSITON"},
        "G": {"G", "GLUTEENITON"},
        "M": {"M", "MAIDOTON"},
    }

    upper_diet = diet.upper()
    for normalized, variations in diet_mapping.items():
        if upper_diet in variations:
            return normalized
    return upper_diet


async def fetch_menus(session: aiohttp.ClientSession) -> List[Dict]:
    """Fetch menus from the JYU API."""
    try:
        async with session.get(LUNCHES_API) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("results", {}).get("en", [])
    except Exception as e:
        logger.error(f"Error fetching menus: {e}")
        return []


def is_valid_price(price: str) -> bool:
    """Check if the price matches the expected format (e.g., 2,95 or 10,50)."""
    price_pattern = r"\d{1,2},\d{2}"
    return bool(re.findall(price_pattern, price.strip()))


def get_most_common_price(items: List[List[Dict]]) -> Optional[str]:
    """Find the most common price in the menu items."""
    price_counts = {}
    for item_group in items:
        for item in item_group:
            price = item.get("price", "").strip()
            if price:
                price_counts[price] = price_counts.get(price, 0) + 1

    if not price_counts:
        return None

    return max(price_counts.items(), key=lambda x: x[1])[0]


def format_menu_item(item: Dict, seen_items: set, common_price: str) -> Optional[str]:
    """Format a single menu item and track duplicates."""
    name = item.get("name", "").strip()
    if not name or name in seen_items:
        return None

    price = item.get("price", "").strip()
    # normalize prices (remove all spaces)
    if price and price.replace(" ", "") != common_price.replace(" ", ""):
        return None

    seen_items.add(name)
    return name


async def format_restaurant_menu(
    restaurant: Dict, diets: Set[str], session: aiohttp.ClientSession
) -> Optional[str]:
    """Format a restaurant's menu items."""
    name = restaurant.get("name", "").strip()
    location = restaurant.get("location", {})
    items = restaurant.get("items", [])

    if not name or not items:
        return None

    location_name = ""
    if location.get("lat") and location.get("lon"):
        osm_location = await get_location_name(
            location["lat"], location["lon"], session
        )
        if osm_location:
            location_name = f" ({osm_location})"

    seen_items = set()
    menu_items = []
    common_price = get_most_common_price(items)

    for item_group in items:
        for item in item_group:
            item_diets = {normalize_diet(d) for d in item.get("diets", [])}
            if all(normalize_diet(diet) in item_diets for diet in diets):
                formatted_item = format_menu_item(item, seen_items, common_price)
                if formatted_item:
                    menu_items.append(formatted_item)

    if not menu_items:
        return None

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

    return f"🍽️ *{name}{location_name}*\n{time_price_info}• {menu_text}\n"


async def send_message_chunks(bot: Bot, text: str, dry_run: bool = False) -> None:
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
                    chat_id=CHANNEL_ID, text=chunk, parse_mode="Markdown"
                )
                await asyncio.sleep(0.1)
        except TelegramError as e:
            logger.error(f"Error sending message to Telegram: {str(e)}")
            logger.error(f"Problematic chunk: {chunk}")


async def get_chefs_choice(diet_menus: str) -> str:
    """Use Groq API to analyze menus and select the best option."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""{diet_menus}\n\nAnalyze the menu options and select the most appealing restaurant dish from those. Return ONLY "*dish name* @ _restaurant name_". After that write "💬" and one short sentence explaining what the main dish you chose is for those who don't know it. Make the comment italic using ONLY underscores. Don't add any other explanation or commentary. AGAIN the format is: *dish name* @ _restaurant name_\n💬 _short explanation_"""

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "qwen/qwen3-32b",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error getting chef's choice: {e}")
        return ""


async def post_daily_menus(diets: List[str], dry_run: bool = False):
    """Fetch and post restaurant menus to the Telegram channel."""
    bot = Bot(TELEGRAM_BOT_TOKEN)
    diets_set = set(diets)

    try:
        async with aiohttp.ClientSession() as session:
            restaurants = await fetch_menus(session)

            menu_parts = []
            all_menus = []

            for restaurant in restaurants:
                menu = await format_restaurant_menu(restaurant, diets_set, session)
                if menu:
                    menu_parts.append(menu)
                    menu_parts.append("➖" * 5 + "\n")
                    all_menus.append(menu)

            if menu_parts:
                diet_str = " & ".join(diets)
                current_date = datetime.date.today()
                formatted_date = current_date.strftime("%A, %B %d")
                header = f"🌱 *{diet_str} Menu for {formatted_date}*\n\n"
                full_message = header + "".join(menu_parts)

                if not dry_run:
                    chefs_choice = await get_chefs_choice("\n\n".join(all_menus))
                    if chefs_choice:
                        full_message += "\n\n👨‍🍳 " + chefs_choice

                await send_message_chunks(bot, full_message, dry_run)
                logger.info(f"Successfully posted {diet_str} menu summary to channel")
            else:
                logger.warning(f"No {' & '.join(diets)} options available for today")

    except Exception as e:
        logger.error(f"Error in post_daily_menus: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post restaurant menus to Telegram channel"
    )
    parser.add_argument(
        "--diets",
        type=str,
        nargs="+",
        default=["Veg"],
        help="Diet filters to apply (e.g., Veg L G). All specified diets must be satisfied.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending messages to Telegram",
    )

    args = parser.parse_args()

    logger.info(f"Posting {' & '.join(args.diets)} menus (dry run: {args.dry_run})")
    asyncio.run(post_daily_menus(args.diets, args.dry_run))
