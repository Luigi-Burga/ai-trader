"""
Watchlist scanner with Characteristic Curve Engine integration.

Compatibility:
- main.py still calls scan_buy_opportunity(stock).
- main.py does not need to be changed.
- buy_target remains informational/legacy and DOES NOT determine BUY.
- SOXL retains its specialized cycle analyzer; the characteristic curve
  is added as a common behavioral layer.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd

from app.ai.characteristic_curve import CurveConfig, analyze
from app.strategies.signal_engine import generate_signal


BENCHMARKS = {
    "NVDA": "QQQ",
    "PLTR": "QQQ",
    "TQQQ": "QQQ",
    "UPRO": "SPY",
    "SMH": "QQQ",
    "SOXL": "SOXX",
    "CCJ": "SPY",
    "VOO": "SPY",
    "CIBR": "QQQ",
    "GDXU": "GDX",
    "AGQ": "SLV",
    "DFEN": "ITA",
    "AMZN": "QQQ",
    "CRWD": "QQQ",
}


def _print_curve(result: dict, buy_target=None):
    ticker = result.get("ticker", "UNKNOWN")
    print("\n" + "=" * 72)
    print(f"CHARACTERISTIC CURVE | {ticker}")
    print("=" * 72)
    print(f"Signal       : {result.get('signal')}")
    print(f"Price        : ${result.get('price', 0):.2f}")
    print(f"Score        : {result.get('score', 0):.1f}/100")
    print(f"Confidence   : {result.get('confidence', 0):.1f}%")
    print(f"Cycle        : {result.get('cycle', '-')}")
    print(f"Matches      : {result.get('matches', 0)}")
    print(f"Similarity   : {result.get('similarity', 0):.1f}%")

    h = result.get("horizons", {})
    print("\nHISTORICAL FORWARD BEHAVIOR")
    for days in (1, 5, 10, 20, 60):
        x = h.get(f"{days}d", {})
        if x:
            print(
                f"{days:>2}D | "
                f"P(up)={x.get('prob_positive', 0)*100:5.1f}% | "
                f"Median={x.get('median_return', 0)*100:+6.2f}% | "
                f"MFE={x.get('mfe_p50', 0)*100:+6.2f}% | "
                f"MAE={x.get('mae_p50', 0)*100:+6.2f}%"
            )

    levels = result.get("levels", {})
    if levels:
        print("\nDYNAMIC TRADE LEVELS")
        print(f"Entry Zone   : ${levels.get('entry_zone_low', 0):.2f} - ${levels.get('entry_zone_high', 0):.2f}")
        print(f"Stop         : ${levels.get('dynamic_stop', 0):.2f}")
        print(f"TP1          : ${levels.get('take_profit_1', 0):.2f}")
        print(f"TP2          : ${levels.get('take_profit_2', 0):.2f}")
        print(f"TP3          : ${levels.get('take_profit_3', 0):.2f}")

    print(f"20D R/R      : {result.get('reward_risk_20d', 0):.2f}")
    if buy_target:
        print(f"Legacy target: ${buy_target:.2f} (informational only)")
    print("=" * 72)


def scan_buy_opportunity(stock):
    ticker = str(stock["ticker"]).upper()
    buy_target = float(stock.get("buy_target", 0) or 0)
    benchmark = stock.get("benchmark") or BENCHMARKS.get(ticker)

    try:
        cfg = CurveConfig(
            period=str(stock.get("curve_period", "5y")),
            interval="1d",
            benchmark=benchmark,
            min_matches=int(stock.get("curve_min_matches", 20)),
            max_matches=int(stock.get("curve_max_matches", 50)),
            min_gap_days=int(stock.get("curve_min_gap_days", 20)),
        )
        result = analyze(ticker, cfg, benchmark=benchmark)
    except Exception as exc:
        print(f"{ticker} => Characteristic Curve error: {exc}")
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "score": 0,
            "confidence": 0,
            "error": str(exc),
        }

    result["watchlist_buy_target"] = buy_target
    result["benchmark"] = benchmark

    _print_curve(result, buy_target=buy_target)

    # IMPORTANT: target price is no longer a BUY decision.
    # It can only generate a legacy informational alert.
    if buy_target and result.get("price", float("inf")) <= buy_target:
        print(f"INFO: {ticker} is below legacy watchlist target ${buy_target:.2f}")

    # Preserve a simple signal field for main.py compatibility.
    final = generate_signal(characteristic_result=result)
    result["final_signal"] = final["signal"]

    return result
