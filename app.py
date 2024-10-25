import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
import re
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
    vegan_pattern = r'\([^()]*\b[Vv]eg\b[^()]*\)'
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


def format_menu_message(
    menu_data: Dict[str, Any], default_price: str = None
) -> Tuple[str, str]:
    """Format menu data into readable messages, returning both regular and vegan only messages."""
    if not menu_data or not menu_data.get("MenusForDays"):
        return "", ""

    restaurant_name = menu_data["RestaurantName"]
    today_menu = menu_data["MenusForDays"][0]

    if not today_menu or not today_menu["SetMenus"]:
        return "", ""

    regular_parts = [f"🍽 *{restaurant_name}*"]
    if today_menu["LunchTime"]:
        regular_parts.append(f"⏰ {today_menu['LunchTime']}\n")

    for menu in today_menu["SetMenus"]:
        if not menu["Components"]:
            continue

        menu_name = menu["Name"]
        price = menu["Price"]
        components = [comp.replace("*", "\\*") for comp in menu["Components"]]
        components_str = "\n- ".join(components)

        price_str = f" ({price})" if price != default_price else ""
        regular_parts.append(f"*{menu_name}*{price_str}\n- {components_str}\n")

    vegan_components = []
    for menu in today_menu["SetMenus"]:
        if not menu["Components"]:
            continue

        menu_vegan_components = [
            comp for comp in components if is_vegan_component(comp)
        ]
        if menu_vegan_components:
            menu_name = menu["Name"]
            price = menu["Price"]
            price_str = f" ({price})" if price != default_price else ""
            components_str = "\n- ".join(menu_vegan_components)
            vegan_components.append(f"*{menu_name}*{price_str}\n- {components_str}\n")

    vegan_parts = []
    if vegan_components:
        vegan_parts = [f"🌱 *{restaurant_name}*"]
        if today_menu["LunchTime"]:
            vegan_parts.append(f"⏰ {today_menu['LunchTime']}\n")
        vegan_parts.extend(vegan_components)

    return "\n".join(regular_parts) if regular_parts else "", "\n".join(
        vegan_parts
    ) if vegan_parts else ""


async def send_message_chunks(bot: Bot, text: str) -> None:
    """Safely send message in chunks to Telegram."""
    if not text:
        return

    chunks = split_message_safely(text)
    for chunk in chunks:
        try:
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

    prompt = f"""Analyze these vegan menu options and select the most appealing restaurant menu from those. Return ONLY the restaurant name followed by its menu EXACTLY as formatted in the input. Don't add any explanation or commentary:

    {vegan_menus}"""

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.1-70b-versatile",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                choice = result["choices"][0]["message"]["content"]
                
                # remove unescaped asterisks
                choice = re.sub(r'(?<!\\)\*', '', choice)
                
                return choice
    except Exception as e:
        logger.error(f"Error getting chef's choice: {str(e)}")
        return ""


async def post_daily_menus(day_offset: int = 0):
    """Fetch and post vegan restaurant menus to the Telegram channel."""
    bot = Bot(TELEGRAM_BOT_TOKEN)
    target_date = datetime.now() + timedelta(days=day_offset)

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_menu(session, url) for url in RESTAURANT_URLS]
            menus = await asyncio.gather(*tasks)

            all_prices = []
            for menu_data in menus:
                if menu_data and menu_data.get("MenusForDays"):
                    today_menu = menu_data["MenusForDays"][0]
                    if today_menu and today_menu["SetMenus"]:
                        all_prices.extend(
                            menu["Price"]
                            for menu in today_menu["SetMenus"]
                            if menu.get("Price")
                        )

            default_price = (
                max(set(all_prices), key=all_prices.count) if all_prices else None
            )

            vegan_message_parts = []
            formatted_date = target_date.strftime("%A, %B %d")

            price_header = (
                f"🗓 *{formatted_date}*\n💰 *Price: {default_price}*\n"
                if default_price
                else f"🗓 *{formatted_date}*\n"
            )
            vegan_message_parts.append(price_header)

            default_price_menus = []
            for menu_data in menus:
                if menu_data:
                    _, vegan_menu = format_menu_message(menu_data, default_price)
                    if vegan_menu:
                        vegan_message_parts.append(vegan_menu)
                        vegan_message_parts.append("-" * 3 + "\n")
                        
                        # if the menu doesn't contain a price (it's the default price)
                        if not re.search(r'\(.*\d+,\d+.*\)', vegan_menu):
                            default_price_menus.append(vegan_menu)

            if vegan_message_parts:
                vegan_header = f"🌱 *Vegan options for {formatted_date}*\n\n"
                vegan_full_message = vegan_header + "\n".join(vegan_message_parts)

                chefs_choice = await get_chefs_choice("\n\n".join(default_price_menus))
                if chefs_choice:
                    vegan_full_message += (
                        "\n\n👨‍🍳 *Chef's Choice of the Day*\n" + chefs_choice
                    )

                await send_message_chunks(bot, vegan_full_message)
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

    args = parser.parse_args()

    logger.info(f"Posting menus for {args.days} days from now")
    asyncio.run(post_daily_menus(args.days))
