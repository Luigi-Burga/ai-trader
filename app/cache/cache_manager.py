import json
import os

from datetime import (
    datetime,
    timedelta
)

CACHE_FILE = "app/cache/fundamentals.json"


def load_cache():

    if not os.path.exists(CACHE_FILE):

        return {}

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read().strip()

            if not content:

                return {}

            return json.loads(content)

    except Exception as e:

        print(
            f"Cache load error: {e}"
        )

        return {}


def save_cache(data):

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def get_cached_fundamental(
    symbol,
    max_hours=24
):

    cache = load_cache()

    if symbol not in cache:

        return None

    timestamp = datetime.fromisoformat(
        cache[symbol]["timestamp"]
    )

    age = datetime.now() - timestamp

    if age > timedelta(hours=max_hours):

        return None

    return cache[symbol]["data"]


def update_cache(
    symbol,
    data
):

    cache = load_cache()

    cache[symbol] = {

        "timestamp":
            datetime.now().isoformat(),

        "data":
            data
    }

    save_cache(cache)