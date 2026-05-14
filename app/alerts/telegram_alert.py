import os

from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("8523771506:AAHaxz7QaXolwaoL0gDg0eh_yVrCzaQNl1g")
CHAT_ID = os.getenv("8822251742")

#bot = Bot(token=BOT_TOKEN)
bot = Bot(token="8523771506:AAHaxz7QaXolwaoL0gDg0eh_yVrCzaQNl1g")

async def send_alert(message):

    await bot.send_message(
        chat_id="8822251742",
        text=message        
    )