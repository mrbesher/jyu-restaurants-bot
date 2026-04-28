#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import logging
import os

import aiohttp

from common_utils import (
    process_restaurants_for_halal,
    get_halal_chefs_choice,
    send_message_chunks,
    Bot,
)

CHANNEL_ID = os.environ["CHANNEL_ID"]

logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("halal-food-bot")


async def build_and_post(dry_run: bool = False, day_offset: int = 0) -> None:
    bot = Bot(os.environ["BOT_TOKEN"])

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        menu_parts, all_dishes, _ = await process_restaurants_for_halal(session, day_offset)

        if menu_parts:
            current_date = datetime.date.today() + datetime.timedelta(days=day_offset)
            formatted_date = current_date.strftime("%A, %B %d")
            header = f"🌱🐟 *Halal Menu for {formatted_date}*\n\n"
            full_message = header + "".join(menu_parts)

            if not dry_run:
                chefs_choice = await get_halal_chefs_choice(session, all_dishes)
                if chefs_choice:
                    full_message += "\n\n👨‍🍳 " + chefs_choice

            await send_message_chunks(bot, CHANNEL_ID, full_message, dry_run)
            logger.info("Successfully posted Halal menu summary to channel")
        else:
            date_label = "today" if day_offset == 0 else f"day offset {day_offset}"
            logger.warning("No Halal options available for %s", date_label)


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
    logger.info(
        "Starting Halal Food Bot (dry run: %s, day offset: %s)",
        args.dry_run,
        args.day_offset,
    )
    asyncio.run(build_and_post(dry_run=args.dry_run, day_offset=args.day_offset))
