# JYU Restaurant Bot

Sends daily menu options from student restaurants with student discounts in Jyväskylä and posts to a [Telegram channel](https://t.me/jyu_yliopiston_ravintolat).

## What it does
- Grabs menus from student restaurants
- Filters options based on diets (e.g., Veg, L, G)
- Uses an LLM to pick the most appealing option
- Supports fetching menus for future days

## Setup
1. Set these environment variables:
   - `BOT_TOKEN`: Your Telegram bot token
   - `GROQ_API_KEY`: Your Groq API key
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `CHANNEL_ID`: Target Telegram channel ID
2. Install requirements: `uv sync`
3. Run the bot: `uv run python app.py --diets Veg L G`

### Usage
```bash
# Today's menu (default)
uv run python app.py --diets Veg L G

# Tomorrow's menu
uv run python app.py --diets Veg L G --day-offset 1

# Menu for 3 days from now
uv run python app.py --diets Veg L G --day-offset 3

# Halal-friendly menu
uv run python halal_app.py --day-offset 1

# Dry run (no posting)
uv run python app.py --dry-run
```

> I am not affiliated with Semma, University of Jyväskylä, JYY or Compass Group. I just like their food! 🥗