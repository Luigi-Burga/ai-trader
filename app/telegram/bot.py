"""AI Trader - Telegram Analysis Bot V1.0."""
from __future__ import annotations
import logging, os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from app.telegram.handlers import analyze_command, help_command, start, unknown_command
VERSION="1.0"

def _token() -> str:
    load_dotenv(); token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token: raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
    return token

def build_application() -> Application:
    app=Application.builder().token(_token()).build()
    app.add_handler(CommandHandler("start", start)); app.add_handler(CommandHandler("help", help_command)); app.add_handler(CommandHandler("analyze", analyze_command)); app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    return app

def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO").upper(), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger(__name__).info("AI Trader Telegram Bot V%s starting...", VERSION)
    build_application().run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__": main()
