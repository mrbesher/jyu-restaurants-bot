import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHANNEL_ID = "@jyu_yliopiston_ravintolat"
CHANNEL_ID = "@jyu_restaurant_test"


RESTAURANT_URLS = [
    "https://www.semma.fi/modules/json/json/Index?costNumber=1402&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1408&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1413&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1401&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1404&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1405&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1414&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1403&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=140301&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1416&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1409&language=en",
    "https://www.semma.fi/modules/json/json/Index?costNumber=1411&language=en",
]


async def fetch_menu(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch menu data from a restaurant URL."""
    try:
        async with session.get(url, headers={"Accept": "application/json"}) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, json.JSONDecodeError) as e:
        logger.error(f"Error fetching menu from {url}: {str(e)}")
        return None


def is_vegan_component(component: str) -> bool:
    """Check if a component is vegan using regex."""
    vegan_pattern = r"\([^()]*\b[Vv]eg\b[^()]*\)"
    return bool(re.search(vegan_pattern, component))


def split_message_safely(message: str, max_length: int = 4000) -> List[str]:
    """Split message into chunks, ensuring Markdown entities are not broken."""
    if len(message) <= max_length:
        return [message]

    chunks = []
    current_chunk = ""
    lines = message.split("\n")

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.rstrip())

    return chunks


def format_menu_message(menu_data: Dict[str, Any]) -> str:
    """Format menu data into readable messages, returning vegan only lunch options."""
    if not menu_data or not menu_data.get("MenusForDays"):
        return "", ""

    restaurant_name = menu_data["RestaurantName"]
    today_menu = menu_data["MenusForDays"][0]

    if not today_menu or not today_menu["SetMenus"]:
        return ""

    vegan_components = []
    for menu in today_menu["SetMenus"]:
        if not menu["Components"]:
            continue

        components = [comp.replace("*", "☆") for comp in menu["Components"]]
        menu_vegan_components = [
            comp for comp in components if is_vegan_component(comp)
        ]
        menu_vegan_components = [
            re.sub(r"[^\S\r\n]*[_\*]*\(.*\)[_*]*[^\S\r\n]*", "", comp)
            for comp in menu_vegan_components
        ]
        if menu_vegan_components:
            menu_name: str = menu["Name"]
            if "lunch" not in menu_name.lower():
                continue
            price = menu["Price"]
            price_str = f"\n_({price})_"
            components_str = "\n• ".join(menu_vegan_components)
            vegan_components.append(f"*{menu_name}*{price_str}\n• {components_str}\n")

    vegan_parts = []
    if vegan_components:
        vegan_parts = [f"🍽️ *{restaurant_name}*"]
        if today_menu["LunchTime"]:
            vegan_parts.append(f"⏰ {today_menu['LunchTime']}\n")
        vegan_parts.extend(vegan_components)

    return "\n".join(vegan_parts) if vegan_parts else ""


async def send_message_chunks(bot: Bot, text: str, dry_run: bool = False) -> None:
    """Safely send message in chunks to Telegram."""
    if not text:
        return

    chunks = split_message_safely(text)
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

    prompt = f"""{vegan_menus}\n\nAnalyze the vegan menu options and select the most appealing restaurant menu from those. Return ONLY "🍽️ restaurant name" followed by its menu EXACTLY as formatted in the input. After the menu write "💬 Comment:" and one sentence about the main dish from the menu you chose for those who don't know it. Make the comment italic using underscores. Don't add any other explanation or commentary."""

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

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_menu(session, url) for url in RESTAURANT_URLS]
            menus = await asyncio.gather(*tasks)

            vegan_message_parts = []
            formatted_date = target_date.strftime("%A, %B %d")

            all_vegan_menus = []
            for menu_data in menus:
                if menu_data:
                    vegan_menu = format_menu_message(menu_data)
                    if vegan_menu:
                        vegan_message_parts.append(vegan_menu)
                        vegan_message_parts.append("➖" * 5 + "\n")
                        all_vegan_menus.append(vegan_menu)

            if vegan_message_parts:
                vegan_header = f"🌱 *{formatted_date}*\n\n"
                vegan_full_message = vegan_header + "".join(vegan_message_parts)


                chef_menu = "\n\n".join(all_vegan_menus)
                chef_menu = re.sub(r"\s*[_\*]*\(.*\)[_*]*[^\S\r\n]*", "", chef_menu)

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Sending to chef for analysis: {chef_menu}"
                    )
                
                chefs_choice = await get_chefs_choice(chef_menu)
                if chefs_choice:
                    vegan_full_message += (
                        "\n\n👨‍🍳 *Chef's Choice of the Day*\n" + chefs_choice
                    )

                await send_message_chunks(bot, vegan_full_message, dry_run)
                logger.info("Successfully posted vegan menu summary to channel")
            else:
                logger.warning(
                    f"No vegan options available for the specified date ({formatted_date})"
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
