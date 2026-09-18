from datetime import datetime

from app.config.config_loader import (
    load_portfolio,
    load_watchlist,
    save_portfolio
)

from app.portfolio.portfolio_monitor import monitor_position

# WATCHLIST SCANNER V2.1
# Validated pipeline:
# watchlist -> Orchestrator V1.3 -> NLE/CCE -> Decision Gate V1.2 -> Signal Engine V1.1
from watchlist_scanner_v2_1 import (
    scan_buy_opportunity
)

from app.fundamentals.score_engine import (
    calculate_fundamental_score,
    build_fundamental_message
)

from app.cache.fundamental_alert_cache import (
    already_sent,
    mark_sent
)

from app.alerts.telegram_alert import (
    send_telegram
)

from app.utils.market_hours import (
    is_market_open
)


def main():

    if is_market_open():
        print(
            "Market closed. "
            "Skipping scan."
        )
        return

    print("\n")
    print("===================================")
    print(f"Market Scan: {datetime.now()}")
    print("===================================")

    #
    # PORTFOLIO MONITOR
    #
    print("\n===== PORTFOLIO MONITOR =====")

    portfolio = load_portfolio()

    for stock in portfolio:

        try:
            monitor_position(stock)

        except Exception as e:
            print(
                f"Portfolio Error "
                f"{stock.get('symbol', 'UNKNOWN')} "
                f": {e}"
            )

    save_portfolio(portfolio)

    #
    # WATCHLIST MONITOR
    #
    print("\n===== WATCHLIST MONITOR V2.1 =====")

    watchlist = load_watchlist()

    for stock in watchlist:

        try:
            result = scan_buy_opportunity(stock)

            # Scanner V2.1 already performs:
            # Orchestrator -> NLE/CCE -> Decision Gate -> Signal Engine.
            # Keep the result available for future notifications/auditing.
            final_signal = str(
                result.get("final_signal")
                or result.get("signal")
                or "ERROR"
            ).upper()

            print(
                f"{stock.get('ticker', 'UNKNOWN')} | "
                f"FINAL={final_signal}"
            )

        except Exception as e:
            print(
                f"Watchlist Error "
                f"{stock.get('ticker', 'UNKNOWN')} "
                f": {e}"
            )

    #
    # FUNDAMENTAL ANALYSIS
    #
    print("\n===== FUNDAMENTAL ANALYSIS =====")

    for stock in watchlist:

        try:
            symbol = stock["ticker"]

            result = calculate_fundamental_score(symbol)

            if result is None:
                continue

            #
            # ETF
            #
            if result["type"] == "ETF":

                print(
                    f"{symbol} | ETF | "
                    f"Fundamental Score N/A"
                )

                continue

            #
            # STOCK
            #
            print(
                f"{symbol} | "
                f"Revenue:{result['revenue']}/25 | "
                f"Margins:{result['margins']}/15 | "
                f"Debt:{result['debt']}/15 | "
                f"Score:{result['total']}/55 | "
                f"{result['rating']}"
            )

            #
            # Fundamental Alert
            #
            if result["total"] >= 45:

                if not already_sent(symbol):

                    message = build_fundamental_message(
                        result
                    )

                    send_telegram(message)

                    mark_sent(symbol)

        except Exception as e:
            print(
                f"Fundamental Error "
                f"{stock.get('ticker', 'UNKNOWN')} "
                f": {e}"
            )

    print("\nScan Completed")
    print("===================================\n")


if __name__ == "__main__":
    main()
