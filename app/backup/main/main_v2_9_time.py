"""
AI Trader - Integrated Main V2.9-PERF
--------------------------------
Structural migration of V2.7:

1. All application Python code is under app/.
2. Prediction Logger is imported as app.prediction_logger.
3. Prediction Tracker is imported as app.prediction_tracker.
4. Persistent prediction data remains OUTSIDE app/:
       data/predictions/
5. Default prediction storage is resolved from the project root by the
   logger/tracker, so execution is not dependent on the current directory.
6. Keeps the validated V2.7 trading flow unchanged:
   - corrected market-hours condition
   - Watchlist Scanner V2.1.1
   - Telegram/cache handling
   - Prediction Tracker before current T0 logging
   - prediction snapshot for every watchlist result
   - fail-safe prediction logging/tracking
"""

from datetime import datetime
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="backslashreplace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="backslashreplace",
    )

from app.config.config_loader import (
    load_portfolio,
    load_watchlist,
    save_portfolio,
)

from app.portfolio.portfolio_monitor import monitor_position

from app.scanners.watchlist_scanner_v2_1_2 import (
    VERSION as WATCHLIST_SCANNER_VERSION,
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

# V2.8: ALL application modules are under app/
from app.prediction_logger import log_prediction
from app.prediction_tracker import update_all_predictions


def _perf_seconds(start: float) -> float:
    """Return elapsed wall-clock seconds for a performance measurement."""
    return time.perf_counter() - start


def _perf_log(label: str, start: float) -> float:
    """Print one performance measurement and return elapsed seconds."""
    elapsed = _perf_seconds(start)
    print(f"[PERF] {label}: {elapsed:.2f} sec", flush=True)
    return elapsed


def _prediction_directory() -> str:
    """
    Return the optional prediction directory override.

    If not configured, logger/tracker use their project-root default:
        <project_root>/data/predictions/
    """
    return os.getenv(
        "AI_TRADER_PREDICTION_DIR",
        "",
    ).strip()


def _scan_allowed() -> bool:
    """
    Normal production behavior: scan only when market is open.

    Test override:
        AI_TRADER_FORCE_SCAN=1
    """
    force_scan = os.getenv(
        "AI_TRADER_FORCE_SCAN",
        "",
    ).strip().lower()

    if force_scan in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            "TEST MODE: AI_TRADER_FORCE_SCAN enabled."
        )
        return True

    return is_market_open()


def _run_prediction_tracker() -> None:
    """
    Evaluate eligible historical prediction snapshots before creating the
    current T0 snapshots.

    Today's file is excluded so current predictions remain PENDING.

    Tracker failures are fail-safe and never stop the trading scan.
    """
    perf_start = time.perf_counter()
    prediction_dir = _prediction_directory()

    try:
        if prediction_dir:
            summary = update_all_predictions(
                base_dir=prediction_dir,
                include_today=False,
            )
        else:
            # Use prediction_tracker's project-root default.
            summary = update_all_predictions(
                include_today=False,
            )

        print(
            "Prediction Tracker | "
            f"files={summary.get('files', 0)} | "
            f"updated={summary.get('updated', 0)} | "
            f"unchanged={summary.get('unchanged', 0)}"
        )
        _perf_log("Prediction Tracker", perf_start)

    except Exception as e:
        _perf_log("Prediction Tracker (failed)", perf_start)
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

    Logging is fail-safe: storage/logger errors never change the trading
    decision or interrupt the watchlist loop.
    """
    prediction_dir = _prediction_directory()

    try:
        if prediction_dir:
            snapshot, path = log_prediction(
                symbol,
                result,
                source="main_integrated_v2_8",
                run_id=run_id,
                base_dir=prediction_dir,
            )
        else:
            snapshot, path = log_prediction(
                symbol,
                result,
                source="main_integrated_v2_8",
                run_id=run_id,
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
            "scan result preserved"
        )

"""
if not _scan_allowed():
"""
def main() -> None:
    total_perf_start = time.perf_counter()

    if _scan_allowed():
        print("Market closed. Skipping scan.")
        return

    run_id = (
        "MAIN_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    print("\n")
    print("===================================")
    print(f"AI Trader Integrated V2.9-PERF")
    print(f"Market Scan: {datetime.now()}")
    print(f"Run ID: {run_id}")
    print("===================================")

    # PREDICTION TRACKER
    _run_prediction_tracker()

    # PORTFOLIO
    portfolio_perf_start = time.perf_counter()
    print("\n===== PORTFOLIO MONITOR =====")
    portfolio = load_portfolio()

    for stock in portfolio:
        portfolio_symbol = stock.get("symbol", "UNKNOWN")
        ticker_perf_start = time.perf_counter()
        try:
            monitor_position(stock)
            _perf_log(f"Portfolio {portfolio_symbol}", ticker_perf_start)
        except Exception as e:
            _perf_log(f"Portfolio {portfolio_symbol} (failed)", ticker_perf_start)
            print(
                f"Portfolio Error "
                f"{stock.get('symbol', 'UNKNOWN')} : {e}"
            )

    save_portfolio(portfolio)
    _perf_log("Portfolio Monitor TOTAL", portfolio_perf_start)

    # WATCHLIST V2.1.2 + PREDICTION LOGGER
    watchlist_perf_start = time.perf_counter()
    print(
        f"\n===== WATCHLIST MONITOR V{WATCHLIST_SCANNER_VERSION} ====="
    )
    watchlist = load_watchlist()

    for stock in watchlist:
        symbol = stock.get(
            "ticker",
            "UNKNOWN",
        )
        ticker_perf_start = time.perf_counter()

        try:
            result = scan_buy_opportunity(stock)

            # Capture T0 before moving to the next ticker.
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
            _perf_log(f"Watchlist {symbol}", ticker_perf_start)

        except Exception as e:
            _perf_log(f"Watchlist {symbol} (failed)", ticker_perf_start)
            print(
                f"Watchlist Error "
                f"{symbol} : {e}"
            )

    _perf_log("Watchlist Scanner TOTAL", watchlist_perf_start)

    # FUNDAMENTAL ANALYSIS
    fundamental_perf_start = time.perf_counter()
    print("\n===== FUNDAMENTAL ANALYSIS =====")

    for stock in watchlist:
        fundamental_ticker_start = time.perf_counter()
        try:
            symbol = stock["ticker"]
            result = calculate_fundamental_score(
                symbol
            )

            if result is None:
                _perf_log(f"Fundamental {symbol}", fundamental_ticker_start)
                continue

            if result["type"] == "ETF":
                print(
                    f"{symbol} | ETF | "
                    "Fundamental Score N/A"
                )
                _perf_log(f"Fundamental {symbol}", fundamental_ticker_start)
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
                    message = build_fundamental_message(
                        result
                    )
                    telegram_ok = send_telegram(
                        message
                    )

                    if telegram_ok:
                        mark_sent(symbol)
                        print(
                            f"{symbol} | Telegram alert "
                            "sent successfully and marked "
                            "as sent."
                        )
                    else:
                        print(
                            f"{symbol} | Telegram alert FAILED; "
                            "cache unchanged."
                        )

            _perf_log(f"Fundamental {symbol}", fundamental_ticker_start)

        except Exception as e:
            _perf_log(
                f"Fundamental {stock.get('ticker', 'UNKNOWN')} (failed)",
                fundamental_ticker_start,
            )
            print(
                f"Fundamental Error "
                f"{stock.get('ticker', 'UNKNOWN')} : {e}"
            )

    _perf_log("Fundamental Analysis TOTAL", fundamental_perf_start)

    print("\nScan Completed")
    print("===================================")
    _perf_log("AI TRADER TOTAL", total_perf_start)
    print("===================================\n")


if __name__ == "__main__":
    main()
