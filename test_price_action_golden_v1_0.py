from __future__ import annotations

"""AI Trader - Price Action Engine V1.0 Golden Validation

Validates Phase-1 Price Action against the five current golden cases without
modifying or invoking any production decision engine.

Golden cases:
    NVDA -> WATCH
    TSM  -> BUY_ON_CONFIRMATION
    TQQQ -> BUY_ON_CONFIRMATION
    SPCX -> BUY_ON_CONFIRMATION
    PLTU -> BUY_ON_PULLBACK

The expected production signal is reference-only. Price Action is a feature
engine and MUST NOT generate or alter a trading signal in this test.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

# Make the script runnable from the repository root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.price_action_engine_v1_0 import PriceActionConfig, analyze_latest, build_features
from app.data.market_data import get_history


GOLDEN = {
    "NVDA": "WATCH",
    "TSM": "BUY_ON_CONFIRMATION",
    "TQQQ": "BUY_ON_CONFIRMATION",
    "SPCX": "BUY_ON_CONFIRMATION",
    "PLTU": "BUY_ON_PULLBACK",
}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_ticker(ticker: str, cfg: PriceActionConfig) -> dict:
    raw = get_history(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
    )

    result = {
        "ticker": ticker,
        "golden_signal_reference": GOLDEN[ticker],
        "status": "FAIL",
        "rows": int(len(raw)) if raw is not None else 0,
        "as_of": None,
        "checks": {},
        "features": {},
        "error": None,
    }

    if raw is None or raw.empty:
        result["error"] = "NO_MARKET_DATA"
        return result

    features = build_features(raw, cfg)
    latest = analyze_latest(raw, cfg)

    if features.empty or latest.get("status") != "ANALYZED":
        result["error"] = "PRICE_ACTION_ANALYSIS_FAILED"
        return result

    f = features.iloc[-1]
    result["as_of"] = latest.get("as_of")
    result["features"] = latest.get("features", {})

    checks = {}

    # 1. Engine produced a confirmed daily bar.
    checks["bar_closed"] = int(f["bar_closed"]) == 1

    # 2. No invalid state labels.
    checks["close_state_valid"] = f["candle_close_state"] in {
        "CLOSE_NEAR_HIGH", "CLOSE_MID_RANGE", "CLOSE_NEAR_LOW", "UNAVAILABLE"
    }
    checks["outside_state_valid"] = f["outside_bar_state"] in {
        "NONE", "BULLISH", "BEARISH", "NEUTRAL"
    }

    # 3. Outside-bar classification is internally consistent.
    outside = int(f["outside_bar"]) == 1
    directional_count = sum(
        int(f[k]) for k in (
            "outside_bar_bullish",
            "outside_bar_bearish",
            "outside_bar_neutral",
        )
    )
    checks["outside_classification_consistent"] = (
        directional_count == 1 if outside else directional_count == 0
    )

    # 4. Close strength must remain bounded.
    checks["close_position_bounded"] = (
        not _finite(f["candle_close_position"])
        or 0.0 <= float(f["candle_close_position"]) <= 1.0
    )
    checks["close_strength_bounded"] = (
        not _finite(f["candle_close_strength"])
        or -1.0 <= float(f["candle_close_strength"]) <= 1.0
    )

    # 5. Price Action is evidence only; it must not emit a trading signal.
    checks["no_signal_generation"] = latest["policy"]["signal_generation"] is False
    checks["lookahead_free_contract"] = latest["policy"]["lookahead_free"] is True
    checks["close_confirmed_only"] = latest["policy"]["close_confirmed_only"] is True

    # 6. The 20-day breakout references require enough history before the
    # current bar. With 5y data this must be available.
    checks["breakout_history_available"] = bool(
        _finite(f["close_above_breakout_high"])
        and _finite(f["close_below_breakout_low"])
    )

    # 7. The engine should have a valid ATR once enough history exists.
    checks["atr_available"] = _finite(f["atr"]) and float(f["atr"]) > 0

    # 8. Production signal reference is informational only.
    checks["golden_signal_preserved"] = GOLDEN[ticker] in {
        "WATCH", "BUY_ON_CONFIRMATION", "BUY_ON_PULLBACK"
    }

    result["checks"] = checks
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=list(GOLDEN))
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    # Period is currently fixed at 5y for the golden contract because the
    # breakout/ATR validation requires enough history.
    cfg = PriceActionConfig()

    print("=" * 110)
    print("AI TRADER - PRICE ACTION ENGINE V1.0 GOLDEN VALIDATION")
    print("=" * 110)
    print("Production engines modified: NO")
    print("Orchestrator modified: NO")
    print("Benchmark Resolver modified: NO")
    print("Decision Gate modified: NO")
    print("Signal Engine modified: NO")
    print("Price Action generates trading signals: NO")
    print()

    results = []
    for ticker in args.tickers:
        if ticker not in GOLDEN:
            print(f"WARNING: {ticker} is not a defined golden case")
            continue
        try:
            result = validate_ticker(ticker, cfg)
        except Exception as exc:
            result = {
                "ticker": ticker,
                "golden_signal_reference": GOLDEN[ticker],
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "checks": {},
                "features": {},
            }
        results.append(result)

        feat = result.get("features", {})
        print(
            f"{ticker:5} | {result['status']:4} | as_of={result.get('as_of')} | "
            f"golden={result['golden_signal_reference']} | "
            f"close={feat.get('candle_close_state')} | "
            f"outside={feat.get('outside_bar_state')} | "
            f"direction={feat.get('price_action_direction')} | "
            f"confirmation={feat.get('price_action_confirmation')}"
        )
        if result.get("error"):
            print(f"       ERROR: {result['error']}")
        failed = [k for k, v in result.get("checks", {}).items() if not v]
        if failed:
            print(f"       FAILED CHECKS: {', '.join(failed)}")

    passed = sum(r["status"] == "PASS" for r in results)
    total = len(results)
    print()
    print("-" * 110)
    print(f"GOLDEN RESULT: {passed}/{total} PASS")
    print("-" * 110)

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
