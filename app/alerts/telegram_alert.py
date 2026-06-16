import asyncio

from telegram import Bot
from dotenv import load_dotenv

from app.config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

load_dotenv()

bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_alert(message):

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=message        
    )

def send_telegram(message):

    try:

        asyncio.run(
            send_alert(message)
        )

    except Exception as e:

        print(
            f"Telegram Error: {e}"
        )