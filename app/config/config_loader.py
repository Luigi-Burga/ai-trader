import json

WATCHLIST_FILE = "app/config/watchlist.json"
PORTFOLIO_FILE = "app/config/portfolio.json"

def load_watchlist():

    with open(
        WATCHLIST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["stocks"]


def load_portfolio():

    with open(
        PORTFOLIO_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["positions"]


def save_portfolio(positions):

    data = {
        "positions": positions
    }

    with open(
        PORTFOLIO_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )