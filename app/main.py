from datetime import datetime

from app.config.config_loader import (
    load_portfolio,
    load_watchlist
)

from app.portfolio.portfolio_monitor import monitor_position

from app.scanners.watchlist_scanner import (
    scan_buy_opportunity
)

from app.fundamentals.score_engine import (
    calculate_fundamental_score,
    get_rating
)


def main():

    print("\n")
    print("===================================")
    print(f"Market Scan: {datetime.now()}")
    print("===================================")

    #
    # PORTFOLIO MONITOR
    #
    print("\n===== PORTFOLIO =====")

    portfolio = load_portfolio()

    for stock in portfolio:

        try:

            monitor_position(stock)

        except Exception as e:

            print(
                f"Portfolio error "
                f"{stock['symbol']}: {e}"
            )

    #
    # WATCHLIST MONITOR
    #
    print("\n===== WATCHLIST =====")

    watchlist = load_watchlist()

    for stock in watchlist:

        try:

            scan_buy_opportunity(stock)

        except Exception as e:

            print(
                f"Watchlist error "
                f"{stock['symbol']}: {e}"
            )

    #
    # FUNDAMENTAL ANALYSIS
    #
    print("\n===== FUNDAMENTAL ANALYSIS =====")

    for stock in watchlist:

        try:

            symbol = stock["symbol"]

            result = calculate_fundamental_score(
                symbol
            )

            if result is None:
                continue

            rating = get_rating(
                result["total"]
            )

            print(
                f"{symbol} | "
                f"Revenue:{result['revenue']} "
                f"Margins:{result['margins']} "
                f"Debt:{result['debt']} "
                f"Score:{result['total']}/55 "
                f"{rating}"
            )

        except Exception as e:

            print(
                f"Fundamental error "
                f"{stock['symbol']}: {e}"
            )

    print("\nScan completed.\n")


if __name__ == "__main__":
    main()