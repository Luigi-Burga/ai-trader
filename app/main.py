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

from app.portfolio.multi_level_alerts import (
    evaluate_multi_level_alerts
)

from app.scanners.watchlist_scanner import (
    analyze_buy_opportunity
)

from app.scanners.entry_price import (
    evaluate_entry_price
)

from app.utils.market_hours import (
    market_is_open
)

# ====================================
# WATCHLIST
# Stocks you want to buy
# ====================================

WATCHLIST = [

    {
        "ticker": "NVDA",
        "buy_target": 190
    },

    {
        "ticker": "TQQQ",
        "buy_target": 60
    },

    {
        "ticker": "AMZN",
        "buy_target": 230
    },

    {
        "ticker": "PLTR",
        "buy_target": 125
    },

    {
        "ticker": "CRWD",
        "buy_target": 450
    },

    {
        "ticker": "CIBR",
        "buy_target": 70
    },

    {
        "ticker": "UPRO",
        "buy_target": 115
    },

    {
        "ticker": "SMH",
        "buy_target": 450
    },

    {
        "ticker": "VOO",
        "buy_target": 600
    },

    {
        "ticker": "CCJ",
        "buy_target": 100
    }
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
        "highest_price": 48.26,
        "alerts_sent": []
    },

    {
        "ticker": "FNGU",
        "number_of_shares": 106,
        "buy_price": 33.83,
        "target_profit": 3,
        "trailing_stop": 0,
        "highest_price": 33.83,
        "alerts_sent": []
    },

    {
        "ticker": "GDXU",
        "number_of_shares": 30,
        "buy_price": 161.41,
        "target_profit": 10,
        "trailing_stop": 0,
        "highest_price": 161.41,
        "alerts_sent": []
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


            # ==========================================
            # WATCHLIST MONITOR
            # ==========================================

            print("========== WATCHLIST ==========\n")

            for stock in WATCHLIST:

                try:

                    ticker = stock["ticker"]

                    buy_target = stock["buy_target"]

                    # ==========================================
                    # GET MARKET DATA
                    # ==========================================

                    df = get_stock_data(ticker)

                    current_price = get_current_price(df)

                    # ==========================================
                    # TECHNICAL ANALYSIS
                    # ==========================================

                    technical_result = (
                        analyze_buy_opportunity(df)
                    )

                    # ==========================================
                    # ENTRY PRICE ANALYSIS
                    # ==========================================

                    entry_result = evaluate_entry_price(
                        current_price,
                        buy_target
                    )

                    # ==========================================
                    # CONSOLE OUTPUT
                    # ==========================================

                    print(
                        f"{ticker} | "
                        f"CURRENT: ${current_price} | "
                        f"TARGET: ${buy_target} | "
                        f"SIGNAL: {entry_result['signal']} | "
                        f"RSI: {technical_result['rsi']} | "
                        f"CONFIDENCE: "
                        f"{technical_result['confidence']}"
                    )

                    # ==========================================
                    # BUY ALERT
                    # ==========================================

                    if (
                        entry_result["signal"] == "BUY_NOW"
                        and technical_result["signal"] == "BUY"
                    ):

                        message = f"""
🚀 BUY OPPORTUNITY 🚀

Ticker:
{ticker}

Current Price:
${current_price}

Buy Target:
${buy_target}

Distance To Target:
{entry_result['difference_percent']}%

RSI:
{technical_result['rsi']}

Confidence:
{technical_result['confidence']}

Reasons:
{', '.join(technical_result['reasons'])}

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                    # ==========================================
                    # NEAR BUY ZONE ALERT
                    # ==========================================

                    elif (
                        entry_result["signal"]
                        == "NEAR_BUY_ZONE"
                    ):

                        message = f"""
🟡 NEAR BUY ZONE 🟡

Ticker:
{ticker}

Current Price:
${current_price}

Buy Target:
${buy_target}

Distance:
{entry_result['difference_percent']}%

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                except Exception as e:

                    print(
                        f"{ticker} WATCHLIST ERROR => {e}"
                    )

            # ==========================================
            # PORTFOLIO MONITOR
            # ==========================================

            print("\n========== PORTFOLIO ==========\n")

            for stock in SALELIST:

                try:

                    ticker = stock["ticker"]

                    # ==========================================
                    # GET MARKET DATA
                    # ==========================================

                    df = get_stock_data(ticker)

                    current_price = get_current_price(df)

                    # ==========================================
                    # POSITION STATUS
                    # ==========================================

                    result = calculate_position_status(
                        current_price=current_price,

                        buy_price=stock["buy_price"],

                        target_profit=stock[
                            "target_profit"
                        ]
                    )

                    # ==========================================
                    # MULTI-LEVEL ALERTS
                    # ==========================================

                    multi_alerts = (
                        evaluate_multi_level_alerts(
                            stock,
                            current_price
                        )
                    )

                    # ==========================================
                    # PROCESS ALERTS
                    # ==========================================

                    for alert in multi_alerts:

                        level = alert["level"]

                        if level == 65:

                            emoji = "🟡"

                            title = "EARLY WARNING"

                        elif level == 75:

                            emoji = "🟠"

                            title = (
                                "IMPORTANT TARGET APPROACH"
                            )

                        elif level == 90:

                            emoji = "🔴"

                            title = "PREPARE SELL"

                        elif level == 100:

                            emoji = "🚀"

                            title = "TARGET REACHED"

                        else:

                            emoji = "📈"

                            title = "TARGET ALERT"

                        message = f"""
{emoji} {title} {emoji}

Ticker:
{ticker}

Buy Price:
${result['buy_price']}

Current Price:
${result['current_price']}

Profit:
{result['profit_percent']}%

Alert Level:
{level}%

Alert Price:
${alert['alert_price']}

Target Price:
${alert['target_price']}

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                    # ==========================================
                    # TRAILING STOP ENGINE
                    # ==========================================

                    trailing_result = (
                        evaluate_trailing_stop(
                            stock,
                            current_price
                        )
                    )

                    # ==========================================
                    # UPDATE HIGHEST PRICE
                    # ==========================================

                    stock["highest_price"] = (
                        trailing_result[
                            "highest_price"
                        ]
                    )

                    # ==========================================
                    # CONSOLE OUTPUT
                    # ==========================================

                    print(
                        f"{ticker} | "
                        f"BUY: ${result['buy_price']} | "
                        f"CURRENT: ${result['current_price']} | "
                        f"P/L: "
                        f"{result['profit_percent']}% | "
                        f"HIGHEST: "
                        f"${round(stock['highest_price'], 2)}"
                    )

                    # ==========================================
                    # SELL ALERT
                    # ==========================================

                    if (
                        trailing_result["status"]
                        == "SELL"
                    ):

                        message = f"""
🚨 TRAILING STOP SELL 🚨

Ticker:
{ticker}

Buy Price:
${result['buy_price']}

Current Price:
${result['current_price']}

Profit:
{result['profit_percent']}%

Highest Price:
${round(
    trailing_result['highest_price'],
    2
)}

Trailing Stop:
${round(
    trailing_result['trailing_price'],
    2
)}

Time:
{datetime.now()}
"""

                        print(message)

                        await send_alert(message)

                except Exception as e:

                    print(
                        f"{ticker} PORTFOLIO ERROR => {e}"
                    )

            # ==========================================
            # WAIT BEFORE NEXT SCAN
            # ==========================================

            print("\nNext scan in 60 seconds...\n")

            await asyncio.sleep(60)

        except Exception as e:

            print(f"MAIN LOOP ERROR => {e}")

            await asyncio.sleep(60)

# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================

if __name__ == "__main__":

    try:

        asyncio.run(monitor_market())

    except Exception as e:

        print(f"FATAL ERROR => {e}")