"""
AI Trader - Asset Analysis Orchestrator V1.0
============================================

Orchestrates the historical-analysis engines without modifying main.py,
signal_engine.py or watchlist_scanner.py.

Pipeline:

    ticker
      |
      v
    Asset Age Resolver V1.2
      |
      +-- NEW_LISTING / NLE_EXTENDED_HISTORY --> New Listing Engine V1.3
      |
      +-- MATURE --> Characteristic Curve Engine V1.6.1
      |
      +-- REVIEW_DATA / NO_DATA --> no analysis engine

The orchestrator is intentionally a thin routing layer. It does not invent
signals and does not override the decision produced by NLE or CCE.

For NLE routes, Benchmark Resolver V1.3 is used to provide a validated
benchmark when possible. CCE V1.6.1 already performs its own Benchmark
Resolver V1.3 integration, so the orchestrator does not resolve it twice.

CLI:
    python -m app.ai.asset_analysis_orchestrator --ticker SPCX
    python -m app.ai.asset_analysis_orchestrator --ticker PLTU
    python -m app.ai.asset_analysis_orchestrator --ticker PLTR
    python -m app.ai.asset_analysis_orchestrator --ticker SOXL
    python -m app.ai.asset_analysis_orchestrator --batch SPCX PLTU MSTU ASMG PLTR TSM SOXL TQQQ
    python -m app.ai.asset_analysis_orchestrator --ticker PLTR --pretty

Dependencies:
    pandas, numpy, yfinance
    app.ai.asset_age_resolver_v1_2
    app.ai.new_listing_engine
    app.ai.characteristic_curve_v1_6_1
    app.ai.benchmark_resolver_v1_3
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from app.ai.asset_age_resolver_v1_2 import ResolverConfig, resolve_asset_age
from app.ai.characteristic_curve_v1_6_1 import CurveConfig, analyze as cce_analyze
from app.ai.new_listing_engine import NewListingConfig, analyze as nle_analyze


VERSION = "1.0"
ORCHESTRATOR = "Asset Analysis Orchestrator"

ROUTE_TO_NLE = {"NEW_LISTING", "NLE_EXTENDED_HISTORY"}
ROUTE_TO_CCE = {"MATURE"}
ROUTE_TO_REVIEW = {"REVIEW_DATA", "NO_DATA", "DATA_ERROR"}


def _ticker(value: str) -> str:
    return str(value or "").strip().upper()


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _engine_signal(result: Dict[str, Any]) -> Optional[str]:
    return result.get("operational_signal") or result.get("signal") or result.get("decision")


def _engine_action(result: Dict[str, Any]) -> Optional[str]:
    operational = result.get("operational_decision")
    if isinstance(operational, dict):
        return operational.get("action")
    return result.get("decision") or result.get("signal")


def _engine_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a stable summary while preserving the complete engine result."""
    return {
        "signal": _engine_signal(result),
        "action": _engine_action(result),
        "score": _safe_float(result.get("score")),
        "confidence": _safe_float(result.get("confidence")),
        "current_price": _safe_float(
            result.get("current_price")
            if result.get("current_price") is not None
            else result.get("price")
        ),
        "regime": result.get("regime") or result.get("cycle"),
        "data_quality": result.get("data_quality"),
    }


def _resolve_benchmark_for_nle(
    ticker: str,
    period: str,
    validate: bool,
) -> Dict[str, Any]:
    """Resolve a validated benchmark for NLE without making it mandatory."""
    try:
        from app.ai.benchmark_resolver_v1_3 import resolve_benchmarks

        resolution = resolve_benchmarks(
            ticker,
            manual_benchmark=None,
            allow_market_fallback=True,
            validate=validate,
            period=period,
        )
        return asdict(resolution)
    except Exception as exc:
        return {
            "ticker": ticker,
            "selected_benchmark": None,
            "source": "none",
            "role": "unknown",
            "selection_reason": "benchmark_resolver_error",
            "selection_confidence": 0.0,
            "fallback_reason": str(exc),
            "is_manual": False,
        }


def _run_nle(
    ticker: str,
    period: str,
    interval: str,
    benchmark: Optional[str],
    listing_price: Optional[float],
    validate_benchmark: bool,
) -> Dict[str, Any]:
    cfg = NewListingConfig(
        period=period,
        interval=interval,
        benchmark=benchmark,
    )

    selected_benchmark = benchmark
    benchmark_resolution: Optional[Dict[str, Any]] = None

    if not selected_benchmark:
        benchmark_resolution = _resolve_benchmark_for_nle(
            ticker=ticker,
            period=period,
            validate=validate_benchmark,
        )
        selected_benchmark = benchmark_resolution.get("selected_benchmark")

    result = nle_analyze(
        ticker,
        cfg=cfg,
        benchmark=selected_benchmark,
        listing_price=listing_price,
    )

    result = dict(result)
    result["orchestrator_benchmark"] = selected_benchmark
    result["orchestrator_benchmark_resolution"] = benchmark_resolution
    return result


def _run_cce(
    ticker: str,
    period: str,
    interval: str,
    benchmark: Optional[str],
    validate_benchmark: bool,
) -> Dict[str, Any]:
    cfg = CurveConfig(
        period=period,
        interval=interval,
        benchmark=benchmark,
        auto_resolve_benchmark=True,
        validate_benchmark=validate_benchmark,
    )
    return dict(cce_analyze(ticker, cfg=cfg, benchmark=benchmark))


def analyze_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
    listing_price: Optional[float] = None,
    validate_benchmark: bool = True,
    resolver_config: Optional[ResolverConfig] = None,
) -> Dict[str, Any]:
    """Resolve asset age and run exactly one appropriate analysis engine."""
    symbol = _ticker(ticker)
    if not symbol:
        return {
            "orchestrator": ORCHESTRATOR,
            "version": VERSION,
            "ticker": symbol,
            "status": "ERROR",
            "route": "INVALID_TICKER",
            "engine": "NONE",
            "signal": "NO_DATA",
            "reason": "Ticker is empty.",
        }

    try:
        resolver_cfg = resolver_config or ResolverConfig(
            period=period,
            interval=interval,
        )
        age = resolve_asset_age(symbol, config=resolver_cfg)
    except Exception as exc:
        return {
            "orchestrator": ORCHESTRATOR,
            "version": VERSION,
            "ticker": symbol,
            "status": "ERROR",
            "route": "RESOLVER_ERROR",
            "engine": "NONE",
            "signal": "NO_DATA",
            "reason": f"Asset Age Resolver failed: {exc}",
            "error_type": type(exc).__name__,
        }

    route = str(age.get("route") or "").upper()
    result: Dict[str, Any]
    engine = "NONE"
    engine_error: Optional[Dict[str, str]] = None

    try:
        if route in ROUTE_TO_NLE:
            engine = "NEW_LISTING_ENGINE"
            result = _run_nle(
                symbol,
                period=period if period != "5y" else "1y",
                interval=interval,
                benchmark=benchmark,
                listing_price=listing_price,
                validate_benchmark=validate_benchmark,
            )
        elif route in ROUTE_TO_CCE:
            engine = "CHARACTERISTIC_CURVE_ENGINE"
            result = _run_cce(
                symbol,
                period=period,
                interval=interval,
                benchmark=benchmark,
                validate_benchmark=validate_benchmark,
            )
        elif route in ROUTE_TO_REVIEW:
            result = {
                "signal": "REVIEW_DATA",
                "decision": "REVIEW_DATA",
                "score": 0.0,
                "confidence": 0.0,
            }
        else:
            result = {
                "signal": "REVIEW_DATA",
                "decision": "REVIEW_DATA",
                "score": 0.0,
                "confidence": 0.0,
            }
    except Exception as exc:
        result = {
            "signal": "ENGINE_ERROR",
            "decision": "REVIEW_DATA",
            "score": 0.0,
            "confidence": 0.0,
        }
        engine_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    summary = _engine_summary(result)

    final_status = "ANALYZED" if engine != "NONE" and engine_error is None else "REVIEW"
    if engine_error is not None:
        final_status = "ENGINE_ERROR"

    return {
        "orchestrator": ORCHESTRATOR,
        "version": VERSION,
        "ticker": symbol,
        "as_of": age.get("as_of"),
        "status": final_status,
        "route": route,
        "engine": engine,
        "engine_version": result.get("version"),
        "signal": summary["signal"],
        "action": summary["action"],
        "score": summary["score"],
        "confidence": summary["confidence"],
        "current_price": summary["current_price"],
        "regime": summary["regime"],
        "data_quality": summary["data_quality"] or age.get("data_quality"),
        "asset_age": age,
        "engine_summary": summary,
        "engine_result": result,
        "engine_error": engine_error,
    }


def analyze_many(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
    validate_benchmark: bool = True,
) -> List[Dict[str, Any]]:
    """Analyze multiple tickers independently; one failure does not stop the batch."""
    results: List[Dict[str, Any]] = []
    for ticker in tickers:
        results.append(
            analyze_ticker(
                ticker,
                period=period,
                interval=interval,
                benchmark=benchmark,
                validate_benchmark=validate_benchmark,
            )
        )
    return results


def _print_compact(result: Dict[str, Any]) -> None:
    age = result.get("asset_age") or {}
    print(
        f"{result.get('ticker','?'):>6} | "
        f"route={result.get('route','?'):<22} | "
        f"engine={result.get('engine','?'):<28} | "
        f"signal={str(result.get('signal','?')):<22} | "
        f"score={result.get('score')} | "
        f"conf={result.get('confidence')} | "
        f"days={age.get('trading_days')} | "
        f"features={age.get('valid_feature_rows')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Trader Asset Analysis Orchestrator V1.0"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Single ticker, e.g. SPCX or PLTR")
    group.add_argument("--batch", nargs="+", help="Multiple tickers")

    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--benchmark", default=None, help="Manual benchmark override")
    parser.add_argument("--listing-price", type=float, default=None)
    parser.add_argument(
        "--no-benchmark-validation",
        action="store_true",
        help="Disable Benchmark Resolver V1.3 validation",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print full JSON output",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Reserved for diagnostics; engine errors are still isolated",
    )

    args = parser.parse_args()
    validate = not args.no_benchmark_validation

    if args.ticker:
        result = analyze_ticker(
            args.ticker,
            period=args.period,
            interval=args.interval,
            benchmark=args.benchmark,
            listing_price=args.listing_price,
            validate_benchmark=validate,
        )
        if args.pretty:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            _print_compact(result)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    results = analyze_many(
        args.batch,
        period=args.period,
        interval=args.interval,
        benchmark=args.benchmark,
        validate_benchmark=validate,
    )

    if args.pretty:
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        print("\n===== ASSET ANALYSIS ORCHESTRATOR =====")
        for item in results:
            _print_compact(item)
        print("\n===== ROUTING =====")
        print(
            json.dumps(
                [
                    {
                        "ticker": r["ticker"],
                        "route": r["route"],
                        "engine": r["engine"],
                        "signal": r["signal"],
                        "action": r["action"],
                        "score": r["score"],
                        "confidence": r["confidence"],
                    }
                    for r in results
                ],
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
