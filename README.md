# JYU Restaurant Bot

Sends daily menu options from student restaurants with student discounts in Jyväskylä and posts to a [Telegram channel](https://t.me/jyu_yliopiston_ravintolat).

## What it does
- Grabs menus from student restaurants
- Filters options based on diets (e.g., Veg, L, G)
- Uses an LLM to pick the most appealing option

## Setup
1. Set these environment variables:
   - `BOT_TOKEN`: Your Telegram bot token
   - `GROQ_API_KEY`: Your Groq API key
2. Install requirements
3. `python app.py --diets Veg L G`

> I am not affiliated with Semma, University of Jyväskylä, JYY or Compass Group. I just like their food! 🥗