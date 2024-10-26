import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple

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


class Restaurant(NamedTuple):
    id: str
    name: str
    location: str
    costNumber: str


RESTAURANTS = [
    Restaurant("207659", "Maija", "Mattilanniemi", "1402"),
    Restaurant("207735", "Piato", "Mattilanniemi", "1408"),
    Restaurant("207412", "Tilia", "Seminaarimäki", "1413"),
    Restaurant("207272", "Lozzi", "Seminaarimäki", "1401"),
    Restaurant("207354", "Belvedere", "Seminaarimäki", "1404"),
    Restaurant("207483", "Cafe Syke", "Seminaarimäki", "1405"),
    Restaurant("207190", "Uno", "Ruusupuisto", "1414"),
    Restaurant("207103", "Ylistö", "Ylistönrinne", "1403"),
    Restaurant("207038", "Cafe Kvarkki", "Ylistönrinne", "140301"),
    Restaurant("206838", "Rentukka", "Kortepohja", "1416"),
    Restaurant("206878", "Amanda", "Normaalikoulu", "1411"),
]


async def fetch_lunch_time(session: aiohttp.ClientSession, restaurant: Restaurant, date: str) -> str:
    """Fetch lunch time from the alternative API."""
    if not restaurant.costNumber:
        return ""

    url = "https://www.semma.fi/modules/json/json/Index"
    params = {"costNumber": restaurant.costNumber, "language": "en"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            for menu_day in data.get("MenusForDays", []):
                menu_date = datetime.strptime(menu_day["Date"][:10], "%Y-%m-%d").date()
                if menu_date == target_date:
                    return menu_day.get("LunchTime", "")
    except (aiohttp.ClientError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error fetching lunch time for {restaurant.name}: {str(e)}")
    
    return ""

async def fetch_menu(
    session: aiohttp.ClientSession,
    restaurant: Restaurant,
    date: str
) -> Dict[str, Any]:
    """Fetch menu data from a restaurant using the new Semma API."""
    url = "https://www.semma.fi/api/restaurant/menu/day"
    params = {
        "date": date,
        "language": "en",
        "onlyPublishedMenu": "true",
        "restaurantPageId": restaurant.id
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if not data or "LunchMenu" not in data or not data["LunchMenu"]:
                return None
            
            lunch_menu = data["LunchMenu"]
            
            lunch_time = await fetch_lunch_time(session, restaurant, date)
            
            return {
                "restaurant_name": f"{restaurant.name} ({restaurant.location})",
                "menu": {
                    "lunch_time": lunch_time,
                    "set_menus": [
                        create_set_menu(menu) for menu in lunch_menu["SetMenus"]
                    ]
                }
            }
    except (aiohttp.ClientError, json.JSONDecodeError) as e:
        logger.error(f"Error fetching menu for {restaurant.name}: {str(e)}")
        return None


def create_set_menu(menu: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": menu["Name"].strip(),
        "price": menu["Price"],
        "components": [create_meal_component(meal) for meal in menu["Meals"]],
    }


def create_meal_component(meal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": meal["Name"].strip(),
        "is_vegan": any("veg" in diet.lower() for diet in meal.get("Diets", [])),
    }


def format_menu_message(menu_data: Dict[str, Any]) -> str:
    """Format menu data into readable messages, returning vegan only lunch options."""
    if not menu_data or not menu_data.get("menu"):
        return ""

    restaurant_name = menu_data["restaurant_name"]
    today_menu = menu_data["menu"]

    if not today_menu or not today_menu["set_menus"]:
        return ""

    vegan_parts = []
    if any(menu["set_menus"] for menu in [today_menu]):
        vegan_parts = []
        if today_menu["lunch_time"]:
            vegan_parts.append(f"⏰ {today_menu['lunch_time']}\n")

        for menu in today_menu["set_menus"]:
            if not menu.get("components") or "lunch" not in menu["name"].lower():
                continue

            # Get vegan components
            vegan_components = [
                comp["name"] for comp in menu["components"] if comp["is_vegan"]
            ]

            if vegan_components:
                menu_name = menu["name"]
                price = menu["price"]
                price_str = f"\n_({price})_" if price else ""
                components_str = "\n• ".join(vegan_components)
                vegan_parts.append(f"*{menu_name}*{price_str}\n• {components_str}\n")

    return f"🍽️ *{restaurant_name}*\n" + "\n".join(vegan_parts) if vegan_parts else ""


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


async def get_chefs_choice(vegan_menus: str) -> str:
    """Use Groq API to analyze vegan menus and select the best option."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""{vegan_menus}\n\nAnalyze the menu options and select the most appealing restaurant menu from those. Return ONLY "🍽️ restaurant name" followed by its menu EXACTLY as formatted in the input. After the menu write "💬 Comment:" and one sentence about the main dish from the menu you chose for those who don't know it. Make the comment italic using underscores. Don't add any other explanation or commentary."""

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.4,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                choice = result["choices"][0]["message"]["content"]
                return choice
    except Exception as e:
        logger.error(f"Error getting chef's choice: {str(e)}")
        return ""


async def post_daily_menus(day_offset: int = 0, dry_run: bool = False):
    """Fetch and post vegan restaurant menus to the Telegram channel."""
    bot = Bot(TELEGRAM_BOT_TOKEN)
    target_date = datetime.now() + timedelta(days=day_offset)
    formatted_date_api = target_date.strftime("%Y-%m-%d")
    formatted_date_display = target_date.strftime("%A, %B %d")

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_menu(session, restaurant, formatted_date_api)
                for restaurant in RESTAURANTS
            ]
            menus = await asyncio.gather(*tasks)

            vegan_message_parts = []
            all_vegan_menus = []

            for menu_data in menus:
                if menu_data:
                    vegan_menu = format_menu_message(menu_data)
                    if vegan_menu:
                        vegan_message_parts.append(vegan_menu)
                        vegan_message_parts.append("➖" * 5 + "\n")
                        all_vegan_menus.append(vegan_menu)

            if vegan_message_parts:
                vegan_header = f"🌱 *{formatted_date_display}*\n\n"
                vegan_full_message = vegan_header + "".join(vegan_message_parts)

                chef_menu = "\n\n".join(all_vegan_menus)
                chef_menu = re.sub(r"\s*[_\*]*\(.*\)[_*]*[^\S\r\n]*", "", chef_menu)

                if dry_run:
                    logger.info(f"[DRY RUN] Sending to chef for analysis: {chef_menu}")

                chefs_choice = await get_chefs_choice(chef_menu)
                if chefs_choice:
                    vegan_full_message += (
                        "\n\n👨‍🍳 *Chef's Choice of the Day*\n" + chefs_choice
                    )

                await send_message_chunks(bot, vegan_full_message, dry_run)
                logger.info("Successfully posted vegan menu summary to channel")
            else:
                logger.warning(
                    f"No vegan options available for the specified date ({formatted_date_display})"
                )

    except Exception as e:
        logger.error(f"Error in post_daily_menus: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post restaurant menus to Telegram channel"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="Number of days to offset from today (0 for today, 1 for tomorrow, etc.)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending messages to Telegram",
    )

    args = parser.parse_args()

    logger.info(
        f"Posting menus for {args.days} days from now (dry run: {args.dry_run})"
    )
    asyncio.run(post_daily_menus(args.days, args.dry_run))
