#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import logging
import os

import aiohttp
from retry_utils import retry_with_backoff

from common_utils import (
    process_restaurants_for_halal,
    get_halal_chefs_choice,
    send_message_chunks,
    Bot,
)

CHANNEL_ID = os.environ["CHANNEL_ID"]

# Suppress telegram library debug logging that may leak tokens
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("halal-food-bot")


@retry_with_backoff()
async def build_and_post(dry_run: bool = False, day_offset: int = 0) -> None:
    """Build and post halal-friendly menu to Telegram."""
    bot = Bot(os.environ["BOT_TOKEN"])

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Process restaurants for halal requirements
        menu_parts, all_dishes, _ = await process_restaurants_for_halal(session, day_offset)

        if menu_parts:
            # Format the message
            current_date = datetime.date.today() + datetime.timedelta(days=day_offset)

            # Format date as "Day Name, Month Day"
            formatted_date = current_date.strftime("%A, %B %d")
            header = f"🌱🐟 *Halal Menu for {formatted_date}*\n\n"

            # menu_parts already contains translations from process_restaurants_for_halal
            full_message = header + "".join(menu_parts)

            if not dry_run:
                chefs_choice = await get_halal_chefs_choice(session, all_dishes)
                if chefs_choice:
                    full_message += "\n\n👨‍🍳 " + chefs_choice

            await send_message_chunks(bot, CHANNEL_ID, full_message, dry_run)
            logger.info("Successfully posted Halal menu summary to channel")
        else:
            if day_offset == 0:
                logger.warning("No Halal options available for today")
            else:
                logger.warning(f"No Halal options available for day offset {day_offset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post Halal-friendly daily menus to Telegram"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending messages to Telegram",
    )
    parser.add_argument(
        "--day-offset",
        type=int,
        default=0,
        help="Day offset relative to today (0=today, 1=tomorrow, -1=yesterday, etc.)",
    )

    args = parser.parse_args()
    logger.info(f"Starting Halal Food Bot (dry run: {args.dry_run}, day offset: {args.day_offset})")
    asyncio.run(build_and_post(dry_run=args.dry_run, day_offset=args.day_offset))