"""
AI Trader - Integrated Main V2.2 TEST

Changes versus main_integrated_v2_1.py:
1. Forces Python stdout/stderr to UTF-8 when possible.
   This fixes Windows "charmap codec can't encode character" errors
   when Fundamental Analysis prints Unicode emojis.
2. Keeps Watchlist Scanner V2.1 unchanged.
3. Adds optional environment override:
      AI_TRADER_FORCE_SCAN=1
   to execute the scan outside market hours for testing.

Production app/main.py is NOT modified.
"""

from datetime import datetime
import os
import sys

# Windows / redirected-output Unicode protection.
# reconfigure() is supported by Python 3.7+ text streams.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from app.config.config_loader import (
    load_portfolio,
    load_watchlist,
    save_portfolio,
)

from app.portfolio.portfolio_monitor import monitor_position

from watchlist_scanner_v2_1 import (
    scan_buy_opportunity,
)

from app.fundamentals.score_engine import (
    calculate_fundamental_score,
    build_fundamental_message,
)

from app.cache.fundamental_alert_cache import (
    already_sent,
    mark_sent,
)

from app.alerts.telegram_alert import (
    send_telegram,
)

from app.utils.market_hours import (
    is_market_open,
)


def _scan_allowed() -> bool:
    """
    Normal production behavior: scan only when market is open.

    Test override:
        AI_TRADER_FORCE_SCAN=1
    """
    force_scan = os.getenv("AI_TRADER_FORCE_SCAN", "").strip().lower()
    if force_scan in {"1", "true", "yes", "on"}:
        print("TEST MODE: AI_TRADER_FORCE_SCAN enabled.")
        return True

    return is_market_open()


def main() -> None:
    if not _scan_allowed():
        print("Market closed. Skipping scan.")
        return

    print("\n")
    print("===================================")
    print(f"Market Scan: {datetime.now()}")
    print("===================================")

    # PORTFOLIO
    print("\n===== PORTFOLIO MONITOR =====")
    portfolio = load_portfolio()

    for stock in portfolio:
        try:
            monitor_position(stock)
        except Exception as e:
            print(
                f"Portfolio Error "
                f"{stock.get('symbol', 'UNKNOWN')} : {e}"
            )

    save_portfolio(portfolio)

    # WATCHLIST V2.1
    print("\n===== WATCHLIST MONITOR V2.1 =====")
    watchlist = load_watchlist()

    for stock in watchlist:
        try:
            result = scan_buy_opportunity(stock)

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
                f"{stock.get('ticker', 'UNKNOWN')} : {e}"
            )

    # FUNDAMENTAL ANALYSIS
    print("\n===== FUNDAMENTAL ANALYSIS =====")

    for stock in watchlist:
        try:
            symbol = stock["ticker"]
            result = calculate_fundamental_score(symbol)

            if result is None:
                continue

            if result["type"] == "ETF":
                print(
                    f"{symbol} | ETF | "
                    f"Fundamental Score N/A"
                )
                continue

            print(
                f"{symbol} | "
                f"Revenue:{result['revenue']}/25 | "
                f"Margins:{result['margins']}/15 | "
                f"Debt:{result['debt']}/15 | "
                f"Score:{result['total']}/55 | "
                f"{result['rating']}"
            )

            if result["total"] >= 45:
                if not already_sent(symbol):
                    message = build_fundamental_message(result)
                    send_telegram(message)
                    mark_sent(symbol)

        except Exception as e:
            print(
                f"Fundamental Error "
                f"{stock.get('ticker', 'UNKNOWN')} : {e}"
            )

    print("\nScan Completed")
    print("===================================\n")


if __name__ == "__main__":
    main()
