import yfinance as yf

from app.fundamentals.revenue import revenue_score
from app.fundamentals.margins import margin_score
from app.fundamentals.debt import debt_score


def calculate_fundamental_score(symbol):

    try:

        stock = yf.Ticker(symbol)
        info = stock.info

        revenue = revenue_score(info)
        margins = margin_score(info)
        debt = debt_score(info)

        total_score = revenue + margins + debt

        return {
            "symbol": symbol,
            "revenue": revenue,
            "margins": margins,
            "debt": debt,
            "total": total_score
        }

    except Exception as e:

        print(f"{symbol}: {e}")

        return None
    
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