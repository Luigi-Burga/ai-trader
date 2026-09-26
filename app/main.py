"""
AI Trader - Integrated Main V2.13
--------------------------------
Integration of autonomous Alpaca Paper execution.

Based on Main V2.11:
- corrected market-hours condition
- Watchlist Scanner V2.2.2
- Prediction Tracker / Logger
- Fundamental Analysis + Alert Engine DRY-RUN
- Portfolio Monitor

V2.13 adds:
- TradeIntent creation from validated watchlist BUY signals
- Alpaca Risk Gate
- Autonomous Alpaca Paper Executor V2.2
- Post-execution reconciliation
- Telegram notification only

IMPORTANT
---------
This version is PAPER ONLY through the Alpaca Executor.

Execution safety:
- Only BUY / STRONG_BUY signals are eligible.
- BUY_ON_CONFIRMATION is NOT executed automatically.
- Order type is MARKET.
- BUY quantity is determined by AI Trader confidence: 10/13/16/18/20 shares.
- Risk Gate estimates market-order exposure from current price plus a configurable slippage buffer.
- Autonomous execution is explicitly enabled with:
      AI_TRADER_AUTONOMOUS_EXECUTION=1

Default:
    AI_TRADER_AUTONOMOUS_EXECUTION=0

This default prevents an accidental order when testing main.py V2.12.
The validated Executor remains PAPER ONLY.

The scanner decision logic is not duplicated here.
main.py only translates an eligible scanner result into TradeIntent
and passes it through Risk Gate -> Executor.
"""

from __future__ import annotations

from datetime import datetime
import os
import sys

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

from app.scanners.watchlist_scanner_v2_2_2 import (
    VERSION as WATCHLIST_SCANNER_VERSION,
    scan_buy_opportunity,
)

from app.fundamentals.score_engine import (
    calculate_fundamental_score,
)

from app.alerts.fundamental_alert_engine_v1 import (
    process_result as process_fundamental_alert,
)

from app.utils.market_hours import (
    is_market_open,
)

from app.prediction_logger import log_prediction
from app.prediction_tracker import update_all_predictions

# V2.12 - Autonomous Paper execution
from app.broker.alpaca_trade_intent import TradeIntent
from app.broker.alpaca_position_sizer import calculate_buy_quantity
from app.broker.alpaca_risk_gate import create_risk_gate
from app.broker.alpaca_executor import create_executor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXECUTABLE_SIGNALS = {"BUY", "STRONG_BUY"}
NON_EXECUTABLE_CONFIRMATION_SIGNAL = "BUY_ON_CONFIRMATION"


def _prediction_directory() -> str:
    """Return the optional prediction directory override."""
    return os.getenv(
        "AI_TRADER_PREDICTION_DIR",
        "",
    ).strip()


def _autonomous_execution_enabled() -> bool:
    """
    Explicit opt-in for autonomous Paper execution.

    Default is disabled so running main.py cannot accidentally submit
    an Alpaca Paper order before the operator intentionally enables V2.12.
    """
    value = os.getenv(
        "AI_TRADER_AUTONOMOUS_EXECUTION",
        "",
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _execution_quantity(confidence: float | None = None) -> float:
    """Return confidence-driven BUY quantity; no fixed quantity override."""
    if confidence is None:
        raise ValueError("confidence is required for autonomous BUY sizing.")
    quantity = calculate_buy_quantity(float(confidence))
    if quantity <= 0:
        raise ValueError(
            f"Confidence {float(confidence):.2f}% is below the BUY threshold."
        )
    return float(quantity)


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
    Evaluate eligible historical prediction snapshots before creating
    the current T0 snapshots.

    Today's file is excluded so current predictions remain PENDING.

    Tracker failures are fail-safe and never stop the trading scan.
    """
    prediction_dir = _prediction_directory()

    try:
        if prediction_dir:
            summary = update_all_predictions(
                base_dir=prediction_dir,
                include_today=False,
            )
        else:
            summary = update_all_predictions(
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

    Logging is fail-safe: storage/logger errors never change the trading
    decision or interrupt the watchlist loop.
    """
    prediction_dir = _prediction_directory()

    try:
        if prediction_dir:
            snapshot, path = log_prediction(
                symbol,
                result,
                source="main_integrated_v2_12",
                run_id=run_id,
                base_dir=prediction_dir,
            )
        else:
            snapshot, path = log_prediction(
                symbol,
                result,
                source="main_integrated_v2_12",
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


# ---------------------------------------------------------------------------
# V2.12 Trade Intent / Risk Gate / Executor integration
# ---------------------------------------------------------------------------

def _build_trade_intent(
    symbol: str,
    result: dict,
) -> TradeIntent | None:
    """Convert an eligible scanner BUY result into a MARKET TradeIntent."""
    final_signal = str(
        result.get("final_signal")
        or result.get("signal")
        or "ERROR"
    ).upper()

    if final_signal not in EXECUTABLE_SIGNALS:
        return None

    confidence = result.get("confidence")
    if confidence is None:
        raise ValueError(f"{symbol}: executable signal has no confidence.")
    confidence = float(confidence)

    quantity = calculate_buy_quantity(confidence)
    if quantity <= 0:
        print(f"{symbol} | Confidence={confidence:.2f}% | NO BUY: below minimum confidence.")
        return None

    price = result.get("current_price")
    if price is None or float(price) <= 0:
        raise ValueError(f"{symbol}: current_price is required for MARKET risk estimation.")

    strategy = (
        result.get("engine")
        or result.get("strategy")
        or result.get("scanner")
        or "Watchlist Scanner"
    )

    reason = (
        f"Watchlist Scanner {result.get('version', '')} "
        f"signal={final_signal}; score={result.get('score', '-')}; "
        f"benchmark={result.get('benchmark', '-')}; regime={result.get('regime', '-')}; "
        f"confidence={confidence:.2f}%"
    )

    return TradeIntent(
        symbol=symbol,
        side="buy",
        quantity=quantity,
        order_type="market",
        time_in_force="day",
        reason=reason,
        confidence=confidence,
        strategy=str(strategy),
        signal=final_signal,
    )


def _execute_watchlist_signal(
    symbol: str,
    result: dict,
    *,
    risk_gate,
    executor,
) -> None:
    """
    Execute one eligible watchlist signal through Risk Gate -> Executor.

    No execution is attempted unless V2.12 autonomous execution is
    explicitly enabled.
    """
    final_signal = str(
        result.get("final_signal")
        or result.get("signal")
        or "ERROR"
    ).upper()

    if final_signal == NON_EXECUTABLE_CONFIRMATION_SIGNAL:
        print(
            f"{symbol} | FINAL={final_signal} | "
            "NO EXECUTION: confirmation required."
        )
        return

    if final_signal not in EXECUTABLE_SIGNALS:
        print(
            f"{symbol} | FINAL={final_signal} | "
            "NO EXECUTION."
        )
        return

    if not _autonomous_execution_enabled():
        print(
            f"{symbol} | FINAL={final_signal} | "
            "AUTONOMOUS EXECUTION DISABLED."
        )
        return

    try:
        intent = _build_trade_intent(
            symbol,
            result,
        )

        if intent is None:
            return

        print()
        print(f"{symbol} | AUTONOMOUS PAPER EXECUTION")
        print("-" * 60)
        print(f"Trade Intent: {intent.summary()}")
        print(
            f"Market Price (risk reference): "
            f"${float(result.get('current_price')):,.2f}"
        )
        print(
            f"Confidence: "
            f"{intent.confidence:.2f}%"
        )
        print(
            f"Strategy: "
            f"{intent.strategy}"
        )
        print()

        risk_decision = risk_gate.evaluate(intent, market_price=float(result.get("current_price")))

        print("RISK GATE")
        print("-" * 60)
        print(risk_decision.summary())
        print()

        if not risk_decision.approved:
            print(
                f"{symbol} | ORDER NOT SUBMITTED | "
                "Risk Gate rejected."
            )
            return

        execution_result = executor.execute(
            intent=intent,
            risk_decision=risk_decision,
        )

        print()
        print(
            f"{symbol} | EXECUTION RESULT"
        )
        print("-" * 60)
        print(execution_result.summary())

    except Exception as exc:
        # Fail-safe per ticker: one execution error must not stop the
        # remaining watchlist analysis.
        print(
            f"{symbol} | Execution Error: {exc}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _scan_allowed():
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
    print("AI Trader Integrated V2.12")
    print(f"Market Scan: {datetime.now()}")
    print(f"Run ID: {run_id}")
    print("===================================")

    print()
    print("===== EXECUTION MODE =====")
    if _autonomous_execution_enabled():
        print("AUTONOMOUS PAPER EXECUTION: ENABLED")
        print("Order Type: MARKET")
        print(
            print("Quantity: CONFIDENCE-DRIVEN (10/13/16/18/20)")
        )
        print("Telegram: NOTIFICATION ONLY")
        print("Live Trading: DISABLED")
    else:
        print("AUTONOMOUS PAPER EXECUTION: DISABLED")
        print(
            "Set AI_TRADER_AUTONOMOUS_EXECUTION=1 "
            "to enable Paper execution."
        )

    # Create once per main execution.
    risk_gate = create_risk_gate()
    executor = create_executor()

    # PREDICTION TRACKER
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

    # WATCHLIST V2.2.2 + PREDICTION LOGGER + EXECUTION
    print(
        f"\n===== WATCHLIST MONITOR "
        f"V{WATCHLIST_SCANNER_VERSION} ====="
    )
    watchlist = load_watchlist()

    for stock in watchlist:
        symbol = stock.get(
            "ticker",
            "UNKNOWN",
        )

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

            # V2.12 autonomous Paper execution.
            _execute_watchlist_signal(
                symbol,
                result,
                risk_gate=risk_gate,
                executor=executor,
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
            result = calculate_fundamental_score(
                symbol
            )

            if result is None:
                continue

            if result["type"] == "ETF":
                print(
                    f"{symbol} | ETF | "
                    "Fundamental Score N/A"
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

            # FUNDAMENTAL ALERT ENGINE V1
            # DRY-RUN: evaluates and persists state,
            # but does NOT send Telegram.
            alert_decision = process_fundamental_alert(
                result,
                send=False,
            )

            print(
                f"{symbol} | "
                f"Fundamental Alert="
                f"{alert_decision.alert_class} | "
                f"reason={alert_decision.reason}"
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
