import json
import os

CACHE_FILE = "app/cache/fundamental_alerts.json"


def load_cache():

    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(data):

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)


def already_sent(symbol):

    cache = load_cache()

    return cache.get(symbol, False)


def mark_sent(symbol):

    cache = load_cache()

    cache[symbol] = True

    save_cache(cache)