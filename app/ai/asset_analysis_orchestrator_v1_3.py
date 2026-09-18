"""
AI Trader - Asset Analysis Orchestrator V1.3
============================================

V1.2 routing corrections:
- Asset Age Resolver V1.2 is the single authority for engine routing.
- NLE receives the full requested analysis period; 5y is no longer truncated to 1y.
- Resolver trading_days and first_valid_date are passed into NLE.
- NLE_EXTENDED_HISTORY is kept on NLE even when true age exceeds NLE's normal
  320-day CCE hand-off threshold, because the resolver selected NLE due to
  insufficient CCE feature depth.
- Benchmark validation uses the same full analysis period.
- Cross-engine summary/current-price normalization from V1.1 is preserved.
- main.py, signal_engine.py and watchlist_scanner.py are not modified.

Pipeline:
    ticker -> Asset Age Resolver V1.2 -> NLE V1.3 / CCE V1.6.1

The orchestrator separates ROUTING from SIGNAL: the resolver decides which
engine runs; the selected engine decides the trading signal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from app.ai.asset_age_resolver_v1_4 import ResolverConfig, resolve_asset_age
from app.ai.characteristic_curve_v1_6_1 import CurveConfig, analyze as cce_analyze
from app.ai.new_listing_engine import NewListingConfig


VERSION = "1.3"
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


def _nested_current_price(result: Dict[str, Any]) -> Optional[float]:
    """Resolve current price across the NLE and CCE output contracts.

    NLE V1.3 exposes the live price under ``metrics.current_price`` and also
    under ``entry.current_price``. CCE V1.6.1 exposes it at the top level.
    V1.2 normalizes these locations without changing either engine.
    """
    candidates = [
        result.get("current_price"),
        result.get("price"),
    ]
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics.get("current_price"))
    entry = result.get("entry")
    if isinstance(entry, dict):
        candidates.append(entry.get("current_price"))
    levels = result.get("levels")
    if isinstance(levels, dict):
        candidates.append(levels.get("current_price"))
    for value in candidates:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _engine_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a stable cross-engine summary while preserving full output."""
    operational = result.get("operational_decision")
    operational_signal = result.get("operational_signal")
    if isinstance(operational, dict):
        operational_action = operational.get("action")
        operational_status = operational.get("status")
    else:
        operational_action = None
        operational_status = None

    return {
        "signal": _engine_signal(result),
        "action": _engine_action(result),
        "score": _safe_float(result.get("score")),
        "confidence": _safe_float(result.get("confidence")),
        "current_price": _nested_current_price(result),
        "regime": result.get("regime") or result.get("cycle"),
        "data_quality": result.get("data_quality"),
        "benchmark": result.get("orchestrator_benchmark"),
        "benchmark_trust": (
            (result.get("orchestrator_benchmark_resolution") or {}).get(
                "benchmark_trust"
            )
        ),
        "benchmark_trust_decision_grade": (
            (result.get("orchestrator_benchmark_resolution") or {}).get(
                "benchmark_trust_decision_grade"
            )
        ),
        "operational_signal": operational_signal,
        "operational_status": operational_status,
        "operational_action": operational_action,
    }


def _resolve_benchmark(
    ticker: str,
    period: str,
    validate: bool,
    listing_days: Optional[int] = None,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve benchmark through Benchmark Resolver V1.6.1.

    V1.6.1 is authoritative for benchmark selection, validation, relevance,
    and trust metadata. The orchestrator only passes the result downstream.
    """
    try:
        from app.ai.benchmark_resolver_v1_6_1 import resolve_benchmarks

        resolution = resolve_benchmarks(
            ticker,
            benchmark=benchmark,
            period=period,
            listing_days=listing_days,
        )
        return dict(resolution)
    except Exception as exc:
        return {
            "ticker": ticker,
            "selected_benchmark": None,
            "source": "none",
            "role": "unknown",
            "selection_reason": "benchmark_resolver_error",
            "selection_confidence": 0.0,
            "fallback_reason": str(exc),
            "is_manual": benchmark is not None,
            "benchmark_status": "UNAVAILABLE",
            "relevance_class": "UNAVAILABLE",
            "benchmark_trust": "NONE",
            "benchmark_trust_decision_grade": False,
            "resolver_version": "1.6.1",
        }


def _run_nle(
    ticker: str,
    period: str,
    interval: str,
    benchmark: Optional[str],
    listing_price: Optional[float],
    validate_benchmark: bool,
    route: str,
    listing_days: Optional[int],
    listing_date: Optional[str],
) -> Dict[str, Any]:
    # The Asset Age Resolver is authoritative for routing. For
    # NLE_EXTENDED_HISTORY, the resolver has explicitly decided that NLE
    # must remain in charge even though the asset may be older than NLE's
    # normal 320-day CCE hand-off threshold. Raise that threshold locally
    # so NLE does not re-route the asset back to CCE.
    mature_days_for_cce = 320
    if route == "NLE_EXTENDED_HISTORY" and listing_days is not None:
        mature_days_for_cce = max(320, int(listing_days) + 1)

    cfg = NewListingConfig(
        period=period,
        interval=interval,
        benchmark=benchmark,
        mature_days_for_cce=mature_days_for_cce,
    )

    selected_benchmark = benchmark
    benchmark_resolution: Optional[Dict[str, Any]] = None

    benchmark_resolution = _resolve_benchmark(
        ticker=ticker,
        period=period,
        validate=validate_benchmark,
        listing_days=listing_days,
        benchmark=benchmark,
    )
    if not selected_benchmark:
        selected_benchmark = benchmark_resolution.get("selected_benchmark")

    # Pass the resolver's actual history age/date into NLE. This prevents
    # the analysis window from becoming the apparent listing age.
    # NLE itself will use the configured period for technical analysis.
    # Use NLE's dataframe-level API so the resolver's authoritative listing
    # age/date can be passed explicitly. Calling NLE's public analyze() first
    # would download the same history twice and would not expose these fields.
    from app.ai.new_listing_engine import download_history, analyze_dataframe

    data = download_history(ticker, cfg, cfg.period)
    bdf = None
    if selected_benchmark:
        bdf = download_history(selected_benchmark, cfg, cfg.period)

    result = analyze_dataframe(
        ticker,
        data,
        bdf if bdf is not None and not bdf.empty else None,
        cfg,
        listing_price,
        listing_days=listing_days,
        listing_date=listing_date,
    )

    result = dict(result)
    result["orchestrator_benchmark"] = selected_benchmark
    result["orchestrator_benchmark_resolution"] = benchmark_resolution
    result["orchestrator_route"] = route
    result["orchestrator_listing_days"] = listing_days
    result["orchestrator_listing_date"] = listing_date
    return result


def _run_cce(
    ticker: str,
    period: str,
    interval: str,
    benchmark: Optional[str],
    validate_benchmark: bool,
    listing_days: Optional[int],
) -> Dict[str, Any]:
    benchmark_resolution = _resolve_benchmark(
        ticker=ticker,
        period=period,
        validate=validate_benchmark,
        listing_days=listing_days,
        benchmark=benchmark,
    )
    selected_benchmark = benchmark_resolution.get("selected_benchmark") or benchmark

    cfg = CurveConfig(
        period=period,
        interval=interval,
        benchmark=selected_benchmark,
        auto_resolve_benchmark=False,
        validate_benchmark=False,
    )
    result = dict(cce_analyze(ticker, cfg=cfg, benchmark=selected_benchmark))
    result["orchestrator_benchmark"] = selected_benchmark
    result["orchestrator_benchmark_resolution"] = benchmark_resolution
    result["orchestrator_listing_days"] = listing_days
    return result


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
                # Do NOT truncate 5y to 1y. The resolver's route and actual
                # listing age must be preserved, while NLE can use the full
                # requested history for an extended-history asset.
                period=period,
                interval=interval,
                benchmark=benchmark,
                listing_price=listing_price,
                validate_benchmark=validate_benchmark,
                route=route,
                listing_days=age.get("trading_days"),
                listing_date=age.get("first_valid_date"),
            )
        elif route in ROUTE_TO_CCE:
            engine = "CHARACTERISTIC_CURVE_ENGINE"
            result = _run_cce(
                symbol,
                period=period,
                interval=interval,
                benchmark=benchmark,
                validate_benchmark=validate_benchmark,
                listing_days=age.get("trading_days"),
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
        "benchmark": summary["benchmark"],
        "benchmark_trust": summary["benchmark_trust"],
        "benchmark_trust_decision_grade": summary[
            "benchmark_trust_decision_grade"
        ],
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
        f"features={age.get('valid_feature_rows')} | "
        f"benchmark={result.get('benchmark')} | "
        f"trust={result.get('benchmark_trust')} | "
        f"grade={result.get('benchmark_trust_decision_grade')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Trader Asset Analysis Orchestrator V1.3"
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
