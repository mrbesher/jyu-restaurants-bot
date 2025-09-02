#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import logging
import os

import aiohttp
from common_utils import retry_with_backoff

from common_utils import (
    process_restaurants_for_halal,
    translate_dishes,
    get_chefs_choice,
    send_message_chunks,
    Bot,
    is_veg,
    format_restaurant_menu,
    fetch_menus,
)

CHANNEL_ID = os.environ["CHANNEL_ID"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("halal-food-bot")


@retry_with_backoff()
async def build_and_post(dry_run: bool = False) -> None:
    """Build and post halal-friendly menu to Telegram."""
    bot = Bot(os.environ["BOT_TOKEN"])

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60)
    ) as session:
        # Process restaurants for halal requirements
        menu_parts, all_dishes, allowed_fish_by_restaurant = await process_restaurants_for_halal(session)

        if menu_parts:
            # Get translations for non-English dishes
            dishes_by_restaurant = {}
            for restaurant in allowed_fish_by_restaurant:
                dishes_by_restaurant[restaurant] = list(allowed_fish_by_restaurant[restaurant])
            
            translations = await translate_dishes(session, dishes_by_restaurant)
            
            # Update dish names with translations
            all_dishes = [
                (translations.get(dish_name, dish_name), restaurant)
                for dish_name, restaurant in all_dishes
            ]

            # Format the message
            current_date = datetime.date.today()
            formatted_date = current_date.strftime("%A, %B %d")
            header = f"🌱🐟 *Halal Menu for {formatted_date}*\n\n"

            # Rebuild menu with translations
            menu_parts_translated = []
            restaurants = await fetch_menus(session)
            
            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                allowed_fish = allowed_fish_by_restaurant.get(name, set())
                
                def halal_filter(item, item_diets, item_name):
                    return is_veg(item_diets) or item_name in allowed_fish
                
                menu, _ = await format_restaurant_menu(
                    restaurant, halal_filter, session, translations
                )
                
                if menu:
                    menu_parts_translated.append(menu)
                    menu_parts_translated.append("➖" * 5 + "\n")

            full_message = header + "".join(menu_parts_translated)

            if not dry_run:
                chefs_choice = await get_chefs_choice(session, all_dishes)
                if chefs_choice:
                    full_message += "\n\n👨‍🍳 " + chefs_choice

            await send_message_chunks(bot, CHANNEL_ID, full_message, dry_run)
            logger.info("Successfully posted Halal menu summary to channel")
        else:
            logger.warning("No Halal options available for today")


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
    logger.info(f"Starting Halal Food Bot (dry run: {args.dry_run})")
    asyncio.run(build_and_post(dry_run=args.dry_run))