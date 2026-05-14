import asyncio

from datetime import datetime
#from app.utils.market_hours import market_is_open

from app.data.market_data import (
    get_stock_data,
    get_current_price
)

from app.alerts.telegram_alert import send_alert

from app.portfolio.portfolio_monitor import (
    calculate_position_status
)

from app.scanners.watchlist_scanner import (
    analyze_buy_opportunity
)

# =========================
# STOCKS TO BUY
# =========================

WATCHLIST = [

    "NVDA",
    "MU",
    "CCJ",
    "SMH",
    "RIOT",
    "CORZ",
    "WULF"
]

# =========================
# STOCKS OWNED
# =========================

SALELIST = [

    {
        "ticker": "BITX",
        "buy_price": 48.26,
        "target_profit": 11
    },

    {
        "ticker": "FNGU",
        "buy_price": 33.83,
        "target_profit": 5
    },

    {
        "ticker": "PLTR",
        "buy_price": 129.88,
        "target_profit": 10
    },

    {
        "ticker": "PLTU",
        "buy_price": 33.42,
        "target_profit": 10
    }
]

async def monitor_market():
 
    while True:

      #  if not market_is_open():
      #      print("Market is closed.")
      #      await asyncio.sleep(300)
      #      continue

        print("\n===================================")
        print(f"Market Scan: {datetime.now()}")
        print("===================================\n")

        # ====================================
        # BUY OPPORTUNITY SCANNER
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

                if result["signal"] == "BUY":

                    message = f"""
🚀 BUY OPPORTUNITY 🚀

Ticker: {ticker}

Signal: BUY
Confidence: {result['confidence']}

RSI: {result['rsi']}

Reasons:
{', '.join(result['reasons'])}

Time: {datetime.now()}
"""

                    await send_alert(message)

            except Exception as e:

                print(f"{ticker} ERROR => {e}")

        # ====================================
        # PORTFOLIO MONITOR
        # ====================================

        print("\n========== PORTFOLIO ==========\n")

        for stock in SALELIST:

            try:

                ticker = stock["ticker"]

                df = get_stock_data(ticker)

                current_price = get_current_price(df)

                result = calculate_position_status(
                    current_price=current_price,
                    buy_price=stock["buy_price"],
                    target_profit=stock["target_profit"]
                )

                print(
                    f"{ticker} | "
                    f"BUY: {result['buy_price']} | "
                    f"CURRENT: {result['current_price']} | "
                    f"P/L: {result['profit_percent']}%"
                )

                if result["target_hit"]:

                    message = f"""
🚨 SELL TARGET HIT 🚨

Ticker: {ticker}

Buy Price: ${result['buy_price']}
Current Price: ${result['current_price']}

Profit: {result['profit_percent']}%

Target Price: ${result['target_price']}

Time: {datetime.now()}
"""

                    await send_alert(message)

            except Exception as e:

                print(f"{ticker} ERROR => {e}")

        await asyncio.sleep(60)

if __name__ == "__main__":

    asyncio.run(monitor_market())