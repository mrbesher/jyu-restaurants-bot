#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os

import aiohttp

from common_utils import (
    Bot,
    process_weekly_pizza_fish_picks,
    send_message_chunks,
)

CHANNEL_ID = os.environ["CHANNEL_ID"]

logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weekly-bot")


async def build_and_post(dry_run: bool = False) -> None:
    bot = Bot(os.environ["BOT_TOKEN"])
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        message = await process_weekly_pizza_fish_picks(session)
        if message:
            await send_message_chunks(bot, CHANNEL_ID, message, dry_run)
            logger.info("Successfully posted weekly Pizza & Fish picks")
        else:
            logger.warning("No pizza or fish dishes found for the upcoming week")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post weekly Pizza & Fish halal highlights to Telegram"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending messages to Telegram",
    )
    args = parser.parse_args()
    logger.info("Starting Weekly Pizza & Fish Bot (dry run: %s)", args.dry_run)
    asyncio.run(build_and_post(dry_run=args.dry_run))
