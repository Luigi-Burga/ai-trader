"""
Telegram Alert V2
======================

Objetivo:
    Enviar alertas de Telegram de forma segura desde el flujo síncrono
    de AI Trader, sin reutilizar un objeto Bot entre distintos event loops.

Diseño:
    - El Bot se crea dentro de cada ejecución asíncrona.
    - asyncio.run() crea y cierra el event loop de cada envío.
    - El Bot se crea, utiliza y cierra dentro del mismo event loop.
    - No mantiene estado asíncrono global entre llamadas.

API pública:
    send_telegram(message) -> bool

Requisitos:
    pip install python-telegram-bot python-dotenv
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from telegram import Bot


VERSION = "2.0"
MODULE_NAME = "Telegram Alert V2"


def _load_config() -> tuple[str, str]:
    """
    Carga las credenciales usando el mismo esquema de configuración
    utilizado por la aplicación, con fallback a variables de entorno.

    No imprime el token.
    """
    load_dotenv()

    token: Optional[str] = None
    chat_id: Optional[str] = None

    # Intentar primero la configuración actual de AI Trader.
    try:
        from app.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

        token = TELEGRAM_BOT_TOKEN
        chat_id = TELEGRAM_CHAT_ID
    except Exception:
        pass

    # Fallback a variables de entorno.
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN no está configurado."
        )

    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID no está configurado."
        )

    return str(token), str(chat_id)


async def _send_alert_async(message: str) -> None:
    """
    Envía un único mensaje.

    IMPORTANTE:
    El Bot se crea DENTRO del event loop que asyncio.run() acaba de crear.
    Por lo tanto, ningún Bot es reutilizado entre diferentes event loops.
    """
    token, chat_id = _load_config()

    # El Bot vive exclusivamente dentro de este event loop.
    async with Bot(token=token) as bot:
        await bot.send_message(
            chat_id=chat_id,
            text=message,
        )


def send_telegram(message: str) -> bool:
    """
    API síncrona compatible con la función original.

    Cada llamada:
        send_telegram()
            -> asyncio.run()
                -> crea Bot
                -> send_message()
                -> cierra Bot
            -> cierra event loop

    Devuelve True si el envío fue correcto.
    """
    print(f"TELEGRAM SEND: {message}")

    try:
        asyncio.run(_send_alert_async(message))
        print("Telegram: OK")
        return True

    except Exception as exc:
        print(
            f"Telegram Error: {type(exc).__name__}: {exc}"
        )
        return False



def telegram_configured() -> bool:
    """
    Comprueba si TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID están configurados.

    No expone ni imprime el token.
    """
    try:
        _load_config()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Smoke test opcional:
    # python telegram_alert_v2.py "mensaje"
    import sys

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        raise SystemExit(0 if send_telegram(message) else 1)

    print(f"Telegram Alert V2 v{VERSION}")
    print(
        "Uso: python telegram_alert_v2.py "
        '"mensaje de prueba"'
    )
