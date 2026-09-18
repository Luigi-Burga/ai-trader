"""
Fundamental Alert Cache V2

Purpose
-------
Daily deduplication for fundamental Telegram alerts.

Behavior
--------
- An alert is considered "already sent" only when it was successfully
  marked for the current local calendar date.
- mark_sent() stores the ISO date (YYYY-MM-DD), replacing the old
  permanent boolean=True behavior.
- Legacy boolean entries are treated as stale and therefore do not block
  a new alert. This allows the new daily cache to recover from the old
  permanent cache format.
- Symbols are normalized to uppercase.
- Writes are performed atomically to reduce the risk of a partially
  written JSON cache.
- The public API remains compatible with the existing main integration:
      load_cache()
      save_cache(data)
      already_sent(symbol)
      mark_sent(symbol)

Version: 2.0
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


VERSION = "2.0"
MODULE_NAME = "Fundamental Alert Cache V2"

# Keep the same physical cache location used by V1.
CACHE_FILE = Path(__file__).resolve().parent / "fundamental_alerts.json"


def _today() -> str:
    """Return today's local date in ISO format."""
    return date.today().isoformat()


def _normalize_symbol(symbol: str) -> str:
    """Normalize ticker symbols to uppercase and remove surrounding spaces."""
    return str(symbol).strip().upper()


def load_cache() -> dict[str, Any]:
    """
    Load the alert cache.

    Missing, empty, malformed, or non-dictionary cache files are treated
    as an empty cache so a corrupt cache does not permanently suppress
    alerts.
    """
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def save_cache(data: dict[str, Any]) -> bool:
    """
    Save the cache atomically.

    Returns True on success and False if the cache cannot be written.
    """
    if not isinstance(data, dict):
        raise TypeError("cache data must be a dictionary")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None

    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{CACHE_FILE.name}.",
            suffix=".tmp",
            dir=str(CACHE_FILE.parent),
            text=True,
        )

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_name, CACHE_FILE)
        temp_name = None
        return True

    except OSError:
        return False

    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def already_sent(symbol: str) -> bool:
    """
    Return True only when this symbol was successfully marked today.

    Supported V2 format:
        "NVDA": "2026-09-18"

    Legacy V1 format:
        "NVDA": true

    Legacy boolean entries are intentionally treated as stale. They do
    not block today's alert, because V1 had no date information.
    """
    key = _normalize_symbol(symbol)
    cache = load_cache()
    value = cache.get(key)

    if isinstance(value, str):
        return value == _today()

    # V1 stored True permanently. There is no reliable date in that format,
    # so it must not suppress the new daily alert system.
    return False


def mark_sent(symbol: str) -> bool:
    """
    Mark a symbol as successfully alerted today.

    The caller should invoke this ONLY after send_telegram() returns True.
    """
    key = _normalize_symbol(symbol)
    cache = load_cache()
    cache[key] = _today()
    return save_cache(cache)


def clear_symbol(symbol: str) -> bool:
    """Remove one symbol from the cache."""
    key = _normalize_symbol(symbol)
    cache = load_cache()

    if key in cache:
        del cache[key]

    return save_cache(cache)


def clear_cache() -> bool:
    """Clear the entire alert cache."""
    return save_cache({})


if __name__ == "__main__":
    print(f"{MODULE_NAME} V{VERSION}")
    print(f"Cache file : {CACHE_FILE}")
    print(f"Today      : {_today()}")

    cache = load_cache()
    print(f"Entries    : {len(cache)}")

    for symbol, value in sorted(cache.items()):
        status = "SENT_TODAY" if already_sent(symbol) else "AVAILABLE"
        print(f"{symbol}: {value!r} -> {status}")
