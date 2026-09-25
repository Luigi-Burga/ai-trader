"""
AI Trader - Integrated Main V2.7
Prediction Logger V1.1 + Prediction Tracker V1.1 integration.

Changes versus main_integrated_v2_4.py:
1. Keeps the corrected market-hours condition.
2. Keeps Watchlist Scanner V2.1.1.
3. Keeps Telegram/cache success handling unchanged.
4. Runs Prediction Tracker V1.1 at the start of the scan for prior-day prediction files.
5. Logs EVERY watchlist prediction snapshot after scan_buy_opportunity()
   returns, including non-bullish and REVIEW_DATA results.
6. Prediction logging failures never change the trading signal or stop
   the watchlist scan.
7. Prediction log directory can be overridden with:
       AI_TRADER_PREDICTION_DIR
8. Production app/main.py is NOT modified.
"""

from datetime import datetime
import os
import sys

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

from app.scanners.watchlist_scanner_v2_1_1 import (
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

from prediction_logger_v1_1 import log_prediction
from prediction_tracker import update_all_predictions


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


def _run_prediction_tracker() -> None:
    """
    Evaluate eligible historical prediction snapshots before creating the
    current T0 snapshots. Today's file is excluded by Prediction Tracker
    V1.1, so current predictions remain PENDING.

    Tracker failures are fail-safe and never stop the trading scan.
    """
    prediction_dir = os.getenv("AI_TRADER_PREDICTION_DIR", "").strip()
    base_dir = prediction_dir or None

    try:
        summary = update_all_predictions(
            base_dir=base_dir or "data/predictions",
            include_today=False,
        )
        print(
            "Prediction Tracker | "
            f"files={summary.get('files', 0)} | "
            f"updated={summary.get('updated', 0)} | "
            f"unchanged={summary.get('unchanged', 0)}"
        )
    except Exception as e:
        print(
            f"Prediction Tracker Error: {e} | "
            "scan continues"
        )


def _log_prediction_snapshot(
    symbol: str,
    result: dict,
    *,
    run_id: str,
) -> None:
    """
    Persist one immutable T0 prediction snapshot.

    Logging is deliberately fail-safe: a logger/storage error must not
    change the trading decision or interrupt the watchlist loop.
    """
    prediction_dir = os.getenv("AI_TRADER_PREDICTION_DIR", "").strip()
    base_dir = prediction_dir or None

    try:
        snapshot, path = log_prediction(
            symbol,
            result,
            source="main_integrated_v2_7",
            run_id=run_id,
            base_dir=base_dir,
        )

        print(
            f"{symbol} | Prediction logged | "
            f"raw={snapshot.get('raw_signal')} | "
            f"gate={snapshot.get('gated_signal')} | "
            f"FINAL={snapshot.get('final_signal')} | "
            f"price={snapshot.get('price_at_prediction')} | "
            f"file={path}"
        )
    except Exception as e:
        print(
            f"{symbol} | Prediction Logger Error: {e} | "
            f"scan result preserved"
        )


def main() -> None:
    if not _scan_allowed():
        print("Market closed. Skipping scan.")
        return

    run_id = (
        "MAIN_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print("\n")
    print("===================================")
    print(f"Market Scan: {datetime.now()}")
    print(f"Run ID: {run_id}")
    print("===================================")

    # PREDICTION TRACKER
    # Evaluate prior prediction files before writing the current T0 file.
    _run_prediction_tracker()

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

    # WATCHLIST V2.1.1 + PREDICTION LOGGER
    print("\n===== WATCHLIST MONITOR V2.1.1 =====")
    watchlist = load_watchlist()

    for stock in watchlist:
        symbol = stock.get("ticker", "UNKNOWN")

        try:
            result = scan_buy_opportunity(stock)

            # Log the complete returned prediction state BEFORE moving
            # to the next ticker. This captures T0 for every asset.
            _log_prediction_snapshot(
                symbol,
                result,
                run_id=run_id,
            )

            final_signal = str(
                result.get("final_signal")
                or result.get("signal")
                or "ERROR"
            ).upper()

            print(
                f"{symbol} | "
                f"FINAL={final_signal}"
            )

        except Exception as e:
            print(
                f"Watchlist Error "
                f"{symbol} : {e}"
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
                    telegram_ok = send_telegram(message)

                    if telegram_ok:
                        mark_sent(symbol)
                        print(
                            f"{symbol} | Telegram alert sent successfully "
                            f"and marked as sent."
                        )
                    else:
                        print(
                            f"{symbol} | Telegram alert FAILED; "
                            f"cache unchanged."
                        )

        except Exception as e:
            print(
                f"Fundamental Error "
                f"{stock.get('ticker', 'UNKNOWN')} : {e}"
            )

    print("\nScan Completed")
    print("===================================\n")


if __name__ == "__main__":
    main()
