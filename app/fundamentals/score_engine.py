import yfinance as yf

from app.cache.cache_manager import (
    get_cached_fundamental,
    update_cache
)

from app.fundamentals.revenue import revenue_score
from app.fundamentals.margins import margin_score
from app.fundamentals.debt import debt_score


def is_etf(info):

    quote_type = info.get("quoteType", "")

    return quote_type.upper() == "ETF"


def get_rating(score):

    if score >= 45:
        return "🟢 STRONG BUY"

    elif score >= 35:
        return "🟢 BUY"

    elif score >= 25:
        return "🟡 HOLD"

    elif score >= 15:
        return "🟠 REDUCE"

    return "🔴 SELL"


def calculate_fundamental_score(symbol):

    try:

        #
        # Check cache first
        #
        cached = get_cached_fundamental(symbol)

        if cached:
            print(f"{symbol} => FUNDAMENTAL CACHE")
            return cached

        print(f"{symbol} => FUNDAMENTAL DOWNLOAD")

        stock = yf.Ticker(symbol)

        info = stock.info

        #
        # ETF detection
        #
        if is_etf(info):

            result = {
                "symbol": symbol,
                "type": "ETF",
                "revenue": 0,
                "margins": 0,
                "debt": 0,
                "total": 0,
                "rating": "ETF"
            }

            update_cache(symbol, result)

            return result

        #
        # Company scoring
        #
        revenue = revenue_score(info)

        margins = margin_score(info)

        debt = debt_score(info)

        total = revenue + margins + debt

        rating = get_rating(total)

        result = {
            "symbol": symbol,
            "type": "STOCK",
            "revenue": revenue,
            "margins": margins,
            "debt": debt,
            "total": total,
            "rating": rating
        }

        #
        # Save cache
        #
        update_cache(symbol, result)

        return result

    except Exception as e:

        print(
            f"Fundamental score error "
            f"{symbol}: {e}"
        )

        return {
            "symbol": symbol,
            "type": "UNKNOWN",
            "revenue": 0,
            "margins": 0,
            "debt": 0,
            "total": 0,
            "rating": "ERROR"
        }
    
def build_fundamental_message(result):

    return (
        f"📊 FUNDAMENTAL ANALYSIS\n\n"
        f"Ticker: {result['symbol']}\n"
        f"Type: {result['type']}\n\n"
        f"Revenue Growth : {result['revenue']}/25\n"
        f"Margins        : {result['margins']}/15\n"
        f"Debt           : {result['debt']}/15\n\n"
        f"Total Score    : {result['total']}/55\n"
        f"Rating         : {result['rating']}"
    ) 