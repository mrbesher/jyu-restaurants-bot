import argparse
import asyncio
import datetime
import logging
import os
from typing import List

import aiohttp

from common_utils import (
    process_restaurants_for_diet,
    get_chefs_choice,
    send_message_chunks,
    translate_dishes,
    Bot,
    logger,
)

CHANNEL_ID = os.environ["CHANNEL_ID"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def post_daily_menus(diets: List[str], dry_run: bool = False):
    """Fetch and post restaurant menus to the Telegram channel."""
    bot = Bot(os.environ["BOT_TOKEN"])
    diets_set = set(diets)

    try:
        async with aiohttp.ClientSession() as session:
            menu_parts, all_dishes = await process_restaurants_for_diet(session, diets_set)

            if menu_parts:
                diet_str = " & ".join(diets)
                current_date = datetime.date.today()
                formatted_date = current_date.strftime("%A, %B %d")
                header = f"🌱 *{diet_str} Menu for {formatted_date}*\n\n"
                full_message = header + "".join(menu_parts)

                if not dry_run:
                    chefs_choice = await get_chefs_choice(session, all_dishes)
                    if chefs_choice:
                        full_message += "\n\n👨‍🍳 " + chefs_choice

                await send_message_chunks(bot, CHANNEL_ID, full_message, dry_run)
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