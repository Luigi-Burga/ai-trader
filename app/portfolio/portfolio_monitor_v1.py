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

from app.alerts.telegram_alert import (
    send_telegram
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

    #
    # Use intraday data
    #
    df = ticker.history(
        period="1d",
        interval="1m"
    )

    if df.empty:

        print(
            f"{symbol} -> "
            f"No market data"
        )

        return

    current_price = float(
        df["Close"].iloc[-1]
    )

    #
    # IMPORTANT:
    # Use intraday HIGH
    #
    highest_today = float(
        df["High"].max()
    )

    status = calculate_position_status(
        current_price,
        buy_price,
        target_profit
    )

    print(
        f"{symbol} | "
        f"Close={current_price:.2f} | "
        f"High={highest_today:.2f} | "
        f"Profit={status['profit_percent']}%"
    )

    #
    # MULTI LEVEL ALERTS
    #
    alerts = evaluate_multi_level_alerts(
        position,
        highest_today
    )

    for alert in alerts:

        message = (
            f"🚨 {symbol}\n"
            f"Profit Alert: {alert['level']}%\n"
            f"Price: {highest_today:.2f}"
        )

        print(message)

        send_telegram(message)

    #
    # TARGET ALERT
    #
    target = evaluate_target_alert(
        position,
        highest_today
    )

    if target["triggered"]:

        message = (
            f"🎯 TARGET REACHED\n\n"
            f"Ticker: {symbol}\n"
            f"Price Actual: {current_price:.2f}\n"
            f"Price Alto: {highest_today:.2f}\n"
            f"Target: {target['target_price']:.2f}"
        )

        print(message)

        send_telegram(message)

    #
    # TRAILING STOP
    #
    trailing = evaluate_trailing_stop(
        position,
        highest_today
    )

    #
    # Save highest price reached
    #
    position["highest_price"] = max(
        position["highest_price"],
        highest_today
    )

    if trailing["status"] == "SELL":

        message = (
            f"🔴 TRAILING STOP SELL\n\n"
            f"Ticker: {symbol}\n"
            f"Price: {highest_today:.2f}\n"
            f"Stop: {trailing['trailing_price']:.2f}"
        )

        print(message)

        send_telegram(message)