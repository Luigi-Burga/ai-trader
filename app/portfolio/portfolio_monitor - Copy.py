import yfinance as yf

from app.portfolio.target_alert import (
    evaluate_target_alert
)

from app.portfolio.multi_level_alerts import (
    evaluate_multi_level_alerts
)

from app.portfolio.trailing_stop import (
    evaluate_trailing_stop
)


def calculate_position_status(
    current_price,
    buy_price,
    target_profit
):

    profit_percent = (
        (current_price - buy_price)
        / buy_price
    ) * 100

    target_price = buy_price * (
        1 + target_profit / 100
    )

    return {

        "profit_percent": round(
            profit_percent,
            2
        ),

        "target_price": round(
            target_price,
            2
        ),

        "target_hit":
            current_price >= target_price
    }


def monitor_position(position):

    symbol = position["ticker"]

    buy_price = float(
        position["buy_price"]
    )

    target_profit = float(
        position["target_profit"]
    )

    ticker = yf.Ticker(symbol)

    df = ticker.history(period="1d")

    if df.empty:

        print(
            f"{symbol} -> "
            f"No market data"
        )

        return

    current_price = float(
        df["Close"].iloc[-1]
    )

    print(
    f"{symbol} | "
    f"Close={df['Close'].iloc[-1]:.2f} | "
    f"High={df['High'].iloc[-1]:.2f}"
)

    status = calculate_position_status(
        current_price,
        buy_price,
        target_profit
    )

    print(
        f"{symbol} | "
        f"Price={current_price:.2f} | "
        f"Profit={status['profit_percent']}%"
    )

    alerts = evaluate_multi_level_alerts(
        position,
        current_price
    )

    for alert in alerts:

        print(
            f"{symbol} "
            f"Level {alert['level']}%"
        )

    target = evaluate_target_alert(
        position,
        current_price
    )

    if target["triggered"]:

        print(
            f"{symbol} "
            f"TARGET REACHED"
        )

    trailing = evaluate_trailing_stop(
        position,
        current_price
    )

    if trailing["status"] == "SELL":

        print(
            f"{symbol} "
            f"TRAILING STOP SELL"
        )