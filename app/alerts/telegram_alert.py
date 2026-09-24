"""
Telegram Alert V3.0
-------------------

Optimized Telegram notification transport for AI Trader.

Compatibility contract preserved from V2.0:
    send_telegram(message) -> bool
    telegram_configured() -> bool

Key optimization:
    - Reuses ONE Bot instance and ONE asyncio event loop per process.
    - The loop lives in a dedicated worker thread.
    - Each send is submitted to that loop with run_coroutine_threadsafe().
    - The Bot HTTP/session is therefore reused instead of creating a new
      Bot/context/event loop for every message.

This keeps the existing synchronous callers unchanged:
    send_telegram("message")

The implementation is lazy: no Telegram connection is created until the
first send_telegram() call.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from concurrent.futures import Future
from typing import Optional, Tuple

from telegram import Bot
from telegram.error import TelegramError

VERSION = "3.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config() -> Tuple[str, str]:
    """
    Load Telegram credentials using the same precedence as V2.0:
      1. .env / app.config.settings
      2. environment variables

    Never prints the token or chat ID.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # python-dotenv is optional; environment variables can still work.
        pass

    token: Optional[str] = None
    chat_id: Optional[str] = None

    try:
        from app.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

        token = TELEGRAM_BOT_TOKEN
        chat_id = TELEGRAM_CHAT_ID
    except Exception:
        pass

    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError(
            "Telegram configuration missing: "
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required"
        )

    return str(token), str(chat_id)


# ---------------------------------------------------------------------------
# Reusable Telegram transport
# ---------------------------------------------------------------------------

class _TelegramDispatcher:
    """
    Owns a dedicated asyncio event loop and a single reusable Bot.

    Public callers remain synchronous. The dispatcher serializes Telegram
    sends through the same event loop and Bot instance.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot: Optional[Bot] = None
        self._chat_id: Optional[str] = None

        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._init_error: Optional[BaseException] = None

        self._state_lock = threading.Lock()

    def _start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._ready.clear()
            self._shutdown.clear()
            self._init_error = None

            self._thread = threading.Thread(
                target=self._thread_main,
                name="ai-trader-telegram",
                daemon=True,
            )
            self._thread.start()

        # Wait until the loop and Bot are initialized.
        self._ready.wait()

        if self._init_error is not None:
            error = self._init_error
            self.close()
            raise RuntimeError(f"Telegram dispatcher initialization failed: {error}")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._loop = loop

        try:
            token, chat_id = _load_config()
            self._chat_id = chat_id

            # Bot is created once and initialized once on the same loop that
            # will perform all sends.
            bot = Bot(token=token)
            self._bot = bot

            loop.run_until_complete(bot.initialize())
            self._ready.set()

            loop.run_forever()

        except BaseException as exc:
            self._init_error = exc
            self._ready.set()

        finally:
            try:
                if self._bot is not None:
                    loop.run_until_complete(self._bot.shutdown())
            except Exception:
                pass

            self._bot = None
            self._chat_id = None

            try:
                loop.close()
            except Exception:
                pass

            self._loop = None

    async def _send_async(self, message: str) -> bool:
        if self._bot is None or self._chat_id is None:
            raise RuntimeError("Telegram dispatcher is not initialized")

        await self._bot.send_message(
            chat_id=self._chat_id,
            text=message,
        )
        return True

    def send(self, message: str) -> bool:
        if not isinstance(message, str):
            message = str(message)

        self._start()

        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("Telegram event loop is not running")

        future: Future = asyncio.run_coroutine_threadsafe(
            self._send_async(message),
            loop,
        )

        # Preserve V2.0 synchronous semantics: send_telegram() does not
        # return until the Telegram operation has completed.
        return bool(future.result())

    def close(self) -> None:
        thread = self._thread
        loop = self._loop

        if thread is None:
            return

        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

        if thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=5.0)

        with self._state_lock:
            self._thread = None
            self._loop = None
            self._bot = None
            self._chat_id = None


_DISPATCHER = _TelegramDispatcher()


# ---------------------------------------------------------------------------
# Public V2.0-compatible API
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> bool:
    """
    Send a Telegram message.

    V2.0 contract preserved:
        - synchronous API
        - returns True on success
        - returns False on any send/configuration error
        - does not expose credentials

    V3.0 reuses the Bot/session across calls.
    """
    print(f"TELEGRAM SEND: {message}")

    try:
        return _DISPATCHER.send(message)
    except Exception as exc:
        print(f"Telegram Error: {type(exc).__name__}: {exc}")
        return False


def telegram_configured() -> bool:
    """
    Return whether Telegram credentials are configured.

    Does not start the dispatcher or open a network connection.
    """
    try:
        _load_config()
        return True
    except Exception:
        return False


def close_telegram() -> None:
    """
    Optional explicit shutdown hook.

    Existing callers do not need to call this. The dispatcher is also
    registered with atexit for normal process termination.
    """
    _DISPATCHER.close()


atexit.register(close_telegram)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = "AI Trader Telegram Alert V3.0 smoke test"

    print(f"Telegram Alert V{VERSION}")
    print(f"Configured: {telegram_configured()}")
    print(f"Send result: {send_telegram(message)}")
