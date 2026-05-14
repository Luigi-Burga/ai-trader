import asyncio
from datetime import datetime

from app.data.market_data import (
    get_stock_data,
    get_current_price
)

from app.alerts.telegram_alert import send_alert

from app.portfolio.portfolio_monitor import (
    calculate_position_status
)

from app.portfolio.trailing_stop import (
    evaluate_trailing_stop
)

from app.scanners.watchlist_scanner import (
    analyze_buy_opportunity
)

from app.utils.market_hours import (
    market_is_open
)

# ====================================
# WATCHLIST
# Stocks you want to buy
# ====================================

WATCHLIST = [

    "NVDA",
    "PLTR",
    "PLTU",
    "SMH",
    "VOO"
]

# ====================================
# PORTFOLIO
# Stocks already owned
# ====================================

SALELIST = [

    {
        "ticker": "BITX",
        "number_of_shares": 51,
        "buy_price": 48.26,
        "target_profit": 5,
        "trailing_stop": 0,
        "highest_price": 48.26
    },

    {
        "ticker": "FNGU",
        "number_of_shares": 106,
        "buy_price": 33.83,
        "target_profit": 5,
        "trailing_stop": 0,
        "highest_price": 33.83
    },

    {
        "ticker": "PLTR",
        "number_of_shares": 30,
        "buy_price": 129.88,
        "target_profit": 10,
        "trailing_stop": 0,
        "highest_price": 129.88
    },

    {
        "ticker": "PLTU",
        "number_of_shares": 100,
        "buy_price": 33.42,
        "target_profit": 10,
        "trailing_stop": 0,
        "highest_price": 33.42
    }
]

# ====================================
# MAIN MONITOR LOOP
# ====================================

async def monitor_market():

    print("===================================")
    print("AI TRADING AGENT STARTED")
    print("===================================")

    while True:

        try:

            # ====================================
            # MARKET HOURS VALIDATION
            # ====================================

            if not market_is_open():

                print(
                    f"[{datetime.now()}] "
                    f"Market closed. Sleeping..."
                )

                await asyncio.sleep(300)

                continue

            print("\n===================================")
            print(f"Market Scan: {datetime.now()}")
            print("===================================\n")

            # ====================================
            # WATCHLIST SCANNER
            # ====================================

            print("========== WATCHLIST ==========\n")

            for ticker in WATCHLIST:

                try:

                    df = get_stock_data(ticker)

                    result = analyze_buy_opportunity(df)

                    print(
                        f"{ticker} => "
                        f"{result['signal']} | "
                        f"RSI: {result['rsi']} | "
                        f"Confidence: {result['confidence']}"
                    )

                    # ====================================
                    # BUY ALERT
                    # ====================================

                    if result["signal"] == "BUY":

                        message = f"""
🚀 BUY OPPORTUNITY 🚀

Ticker: {ticker}

Signal: BUY

Confidence: {result['confidence']}

RSI: {result['rsi']}

Reasons:
{', '.join(result['reasons'])}

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                except Exception as e:

                    print(f"{ticker} WATCHLIST ERROR => {e}")

            # ====================================
            # PORTFOLIO MONITOR
            # ====================================

            print("\n========== PORTFOLIO ==========\n")

            for stock in SALELIST:

                try:

                    ticker = stock["ticker"]

                    df = get_stock_data(ticker)

                    current_price = get_current_price(df)

                    # ====================================
                    # POSITION STATUS
                    # ====================================

                    result = calculate_position_status(
                        current_price=current_price,
                        buy_price=stock["buy_price"],
                        target_profit=stock["target_profit"]
                    )

                    # ====================================
                    # TRAILING STOP ENGINE
                    # ====================================

                    trailing_result = evaluate_trailing_stop(
                        stock,
                        current_price
                    )

                    # ====================================
                    # UPDATE HIGHEST PRICE
                    # ====================================

                    stock["highest_price"] = (
                        trailing_result["highest_price"]
                    )

                    print(
                        f"{ticker} | "
                        f"BUY: ${result['buy_price']} | "
                        f"CURRENT: ${result['current_price']} | "
                        f"P/L: {result['profit_percent']}% | "
                        f"P/L US$: {round(result['current_price'] * stock['number_of_shares'] - result['buy_price'] * stock['number_of_shares'], 2)} | "
                       # f"HIGHEST: ${round(stock['highest_price'], 2)}"
                    )

                    # ====================================
                    # TARGET REACHED
                    # ====================================

                    if result["target_hit"]:

                        print(
                            f"{ticker} target reached."
                        )

                    # ====================================
                    # TRAILING STOP SELL ALERT
                    # ====================================

                    if trailing_result["status"] == "SELL":

                        message = f"""
🚨 TRAILING STOP SELL 🚨

Ticker: {ticker}

Buy Price:
${result['buy_price']}

Current Price:
${result['current_price']}

Profit:
{result['profit_percent']}%

Highest Price:
${round(trailing_result['highest_price'], 2)}

Trailing Stop:
${round(trailing_result['trailing_price'], 2)}

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                except Exception as e:

                    print(f"{ticker} PORTFOLIO ERROR => {e}")

            # ====================================
            # WAIT BEFORE NEXT SCAN
            # ====================================

            print("\nNext scan in 30 seconds...\n")

            await asyncio.sleep(60)

        except Exception as e:

            print(f"MAIN LOOP ERROR => {e}")

            await asyncio.sleep(30)

# ====================================
# APPLICATION ENTRYPOINT
# ====================================

if __name__ == "__main__":

    try:

        asyncio.run(monitor_market())

    except Exception as e:

        print(f"FATAL ERROR => {e}")