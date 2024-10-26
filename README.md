# Semma Restaurants' Vegan Menu Bot

Sends vegan options with an LLM recommendation from Semma's restaurants and posts daily to a [Telegram channel](https://t.me/jyu_yliopiston_ravintolat).

## What it does
- Grabs menus from Semma's restaurants (semma.fi)
- Filters out the vegan lunch options
- Uses an LLM to pick the best option
- Posts everything to Telegram

## Setup
1. Set these environment variables:
   - `BOT_TOKEN`: Your Telegram bot token
   - `GROQ_API_KEY`: Your Groq API key
2. Install requirements
3. Run `app.py`

Optional flags:
- `--days N`: Post menu N days ahead
- `--dry-run`: Test without posting

> I am not affiliated with Semma, University of Jyväskylä, JYY or Compass Group. I just like their food 🥗
