"""AI Trader - Telegram Handlers V1.0."""
from __future__ import annotations
import asyncio, logging, os
from typing import Optional
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from app.services.analysis_service import analyze_ticker
from app.telegram.formatter import format_analysis, format_error
LOGGER = logging.getLogger(__name__)

def _allowed_user_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    allowed=set()
    for item in raw.split(","):
        try:
            if item.strip(): allowed.add(int(item.strip()))
        except ValueError: LOGGER.warning("Ignoring invalid Telegram user ID value.")
    return allowed

def _authorized(update: Update) -> bool:
    user=update.effective_user
    return bool(user and _allowed_user_ids() and int(user.id) in _allowed_user_ids())

def _extract_ticker(text: Optional[str]) -> Optional[str]:
    if not text: return None
    parts=text.strip().split()
    if not parts: return None
    ticker=parts[0].strip().upper()
    return ticker if ticker.replace(".","").replace("-","").isalnum() and len(ticker)<=15 else None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update): return await update.effective_message.reply_text("Access denied.")
    await update.effective_message.reply_text("AI Trader V1\n\nUse:\n/analyze GDXU\n\nThe command runs the validated AI Trader pipeline.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update): return await update.effective_message.reply_text("Access denied.")
    await update.effective_message.reply_text("<b>AI Trader Telegram V1</b>\n\n/analyze GDXU — analyze one ticker\n/start — usage\n/help — commands", parse_mode=ParseMode.HTML)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update): return await update.effective_message.reply_text("Access denied.")
    ticker=_extract_ticker(" ".join(context.args) if context.args else None)
    if not ticker: return await update.effective_message.reply_text("Usage: /analyze GDXU")
    waiting=await update.effective_message.reply_text(f"🔎 Analyzing <b>{ticker}</b>...", parse_mode=ParseMode.HTML)
    try:
        result=await asyncio.to_thread(analyze_ticker, ticker)
        await waiting.edit_text(format_analysis(result), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as exc:
        LOGGER.exception("Telegram analysis failed for %s", ticker)
        await waiting.edit_text(format_error(f"Analysis failed for {ticker}: {type(exc).__name__}: {exc}"), parse_mode=ParseMode.HTML)

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update): return await update.effective_message.reply_text("Access denied.")
    await update.effective_message.reply_text("Unknown command. Use /help.")
