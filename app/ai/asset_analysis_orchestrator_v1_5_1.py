"""
AI Trader - Asset Analysis Orchestrator V1.5.0
============================================

V1.4 data-flow architecture (preserving V1.3 routing):
- Asset Age Resolver V1.4 is the single authority for engine routing.
- NLE receives the full requested analysis period; 5y is no longer truncated to 1y.
- Resolver trading_days and first_valid_date are passed into NLE.
- NLE_EXTENDED_HISTORY is kept on NLE even when true age exceeds NLE's normal
  320-day CCE hand-off threshold, because the resolver selected NLE due to
  insufficient CCE feature depth.
- Benchmark validation uses the same full analysis period.
- Cross-engine summary/current-price normalization from V1.1 is preserved.
- Market Data V2 is the orchestrator data boundary.
- Asset Age receives the already-loaded raw asset dataframe.
- Benchmark Resolver receives the already-loaded adjusted asset dataframe.
- CCE consumes the resolver-selected benchmark dataframe in memory.
- NLE receives the already-loaded raw asset dataframe.
- auto_adjust=False/True semantics are preserved per existing engine contract.
- main.py, signal_engine.py and watchlist_scanner.py are not modified.

Pipeline:
    ticker
      -> Market Data V2
         -> raw asset dataframe
         -> adjusted asset dataframe
      -> Asset Age Resolver V1.4 (routing)
      -> Benchmark Resolver V1.6.1
      -> NLE V1.4.1 / CCE V1.7

The orchestrator separates ROUTING from SIGNAL: the resolver decides which
engine runs; the selected engine decides the trading signal.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from app.data.market_data import get_history

from app.ai.asset_age_resolver_v1_5 import (
    ResolverConfig,
    resolve_asset_age,
    resolve_from_dataframe,
)
from app.ai.characteristic_curve_v1_7 import CurveConfig, analyze as cce_analyze
from app.ai.new_listing_engine_v1_4_1 import NewListingConfig


VERSION = "1.5.1"
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

    NLE V1.4.1 exposes the live price under ``metrics.current_price`` and also
    under ``entry.current_price``. CCE V1.7 exposes it at the top level.
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
        "appt_x_20d": _engine_appt_x_20d(result),
        "expected_return_20d": _engine_expected_return_20d(result),
        "levels": _engine_levels(result),
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


def _engine_levels(result: Dict[str, Any]) -> Dict[str, Any]:
    """Expose a stable entry/stop/target contract across analysis engines.

    CCE V1.7 keeps its legacy ``levels`` payload, so the orchestrator adds
    normalized aliases without removing or modifying the original payload.
    NLE outputs are normalized at the orchestrator boundary, including
    ascending pullback-zone bounds, while preserving the raw engine result.
    """
    # CCE exposes its operational levels under ``levels``.
    # NLE V1.4.1 exposes the equivalent payload under ``entry``.
    # Normalize both contracts without changing either engine.
    raw = result.get("levels")
    if not isinstance(raw, dict):
        raw = result.get("entry")
    if not isinstance(raw, dict):
        raw = result.get("legacy_levels")
    if not isinstance(raw, dict):
        raw = {}

    def first(*keys: str) -> Optional[float]:
        for key in keys:
            value = _safe_float(raw.get(key))
            if value is not None:
                return value
        return None

    current_price = first("current_price", "price", "entry_price")
    entry_price = first("entry_price", "preferred_entry", "current_price")
    zone_low = first(
        "entry_zone_low", "zone_low", "historical_pullback_low"
    )
    zone_high = first(
        "entry_zone_high", "zone_high", "historical_pullback_high"
    )
    stop = first(
        "stop", "stop_loss", "dynamic_stop", "historical_stop_reference"
    )
    tp1 = first("take_profit_1", "tp_1", "historical_tp_50", "tp1")
    tp2 = first("take_profit_2", "tp_2", "historical_tp_75", "tp2")
    tp3 = first("take_profit_3", "tp_3", "historical_tp_90", "tp3")
    risk_pct = first("risk_pct")

    # Boundary normalization: NLE V1.4.1 can emit pullback bounds in
    # descending order (e.g. zone_low=55.66, zone_high=52.43).  The
    # orchestrator owns the cross-engine output contract, so normalize the
    # two bounds here without modifying the source engine result.
    #
    # Important: entry_price/preferred_entry is deliberately NOT clamped or
    # moved into the zone.  It is an independent engine level and must retain
    # its original meaning.  Decision Gate remains responsible for validating
    # the semantics of the resulting levels.
    if zone_low is not None and zone_high is not None and zone_low > zone_high:
        zone_low, zone_high = zone_high, zone_low

    return {
        "current_price": current_price,
        "entry_price": entry_price,
        "entry_zone_low": zone_low,
        "entry_zone_high": zone_high,
        "stop": stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "take_profit_3": tp3,
        "risk_pct": risk_pct,
    }


def _engine_appt_x_20d(result: Dict[str, Any]) -> Optional[float]:
    """Read the additive V1.7 APPT field from either supported location."""
    historical = result.get("historical_pattern")
    if not isinstance(historical, dict):
        historical = {}
    return _safe_float(
        result.get("appt_x_20d", historical.get("appt_x_20d"))
    )


def _engine_expected_return_20d(result: Dict[str, Any]) -> Optional[float]:
    historical = result.get("historical_pattern")
    if not isinstance(historical, dict):
        historical = {}
    return _safe_float(
        result.get("expected_return_20d", historical.get("expected_return_20d"))
    )


def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize the shared asset DataFrame to canonical OHLCV columns.

    Market Data V2 may preserve yfinance MultiIndex columns. The shared
    asset context must expose a single-level dataframe before it is passed
    to Asset Age, Benchmark Resolver, or CCE. No investment logic is changed.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy(deep=True)

    if isinstance(x.columns, pd.MultiIndex):
        selected: Dict[str, Any] = {}
        for col in x.columns:
            parts = [str(part).strip() for part in col]
            for part in parts:
                key = part.lower()
                if key in {"open", "high", "low", "close", "volume"}:
                    canonical = key.capitalize()
                    if canonical not in selected:
                        selected[canonical] = col
                    break

        if selected:
            x = x[[selected[name] for name in selected]]
            x.columns = list(selected.keys())

    rename: Dict[Any, str] = {}
    for col in x.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.capitalize()
    x = x.rename(columns=rename)
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    required = ("Open", "High", "Low", "Close", "Volume")
    if any(col not in x.columns for col in required):
        return pd.DataFrame()

    x = x[list(required)].copy()
    for col in required:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.replace([float("inf"), float("-inf")], pd.NA)
    x = x.dropna(subset=["High", "Low", "Close"])
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _market_history(
    ticker: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Read market history exclusively through Market Data V2.

    The auto_adjust flag is intentionally explicit because the existing engines
    use different yfinance semantics:
      - Asset Age / NLE: auto_adjust=False
      - Benchmark Resolver / CCE: auto_adjust=True

    Market Data V2 keeps those variants in separate cache keys, so this preserves
    engine semantics while eliminating direct yfinance access from the
    orchestrator.
    """
    if not ticker:
        return pd.DataFrame()

    raw = get_history(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
        group_by="column",
        threads=False,
    )
    return _normalize_ohlcv(raw)


def _resolve_benchmark(
    ticker: str,
    period: str,
    interval: str,
    validate: bool,
    listing_days: Optional[int] = None,
    benchmark: Optional[str] = None,
    asset_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Resolve benchmark through Benchmark Resolver V1.6.1.

    The orchestrator supplies the already-loaded adjusted asset dataframe.
    Benchmark Resolver remains authoritative for selection/validation.

    ``_selected_benchmark_data`` is an internal transport field returned only
    when requested. It is consumed by the selected engine and removed from the
    public result contract.
    """
    try:
        from app.ai.benchmark_resolver_v1_6_1 import resolve_benchmarks

        resolution = resolve_benchmarks(
            ticker,
            benchmark=benchmark,
            period=period,
            listing_days=listing_days,
            asset_df=asset_df,
            return_selected_data=True,
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
            "_selected_benchmark_data": None,
        }


def _pop_selected_benchmark_data(
    resolution: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Extract resolver-selected benchmark data without exposing it publicly."""
    value = resolution.pop("_selected_benchmark_data", None)
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value
    return None


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
    asset_df: pd.DataFrame,
    adjusted_asset_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    # The Asset Age Resolver is authoritative for routing. For
    # NLE_EXTENDED_HISTORY, the resolver has explicitly decided that NLE
    # must remain in charge even though the asset may be older than NLE's
    # normal 320-day CCE hand-off threshold.
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
        interval=interval,
        validate=validate_benchmark,
        listing_days=listing_days,
        benchmark=benchmark,
        asset_df=(
            adjusted_asset_df
            if adjusted_asset_df is not None and not adjusted_asset_df.empty
            else None
        ),
    )

    if not selected_benchmark:
        selected_benchmark = benchmark_resolution.get("selected_benchmark")

    # Benchmark Resolver already downloaded the selected benchmark in the
    # adjusted form for validation. NLE historically uses auto_adjust=False,
    # so preserve that semantic with a single Market Data V2 cache-backed read.
    bdf = None
    if selected_benchmark:
        bdf = _market_history(
            selected_benchmark,
            period=period,
            interval=interval,
            auto_adjust=False,
        )

    from app.ai.new_listing_engine_v1_4_1 import analyze_dataframe

    result = analyze_dataframe(
        ticker,
        asset_df,
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
    result["orchestrator_nle_version"] = result.get("version")
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
    adjusted_asset_df: pd.DataFrame,
) -> Dict[str, Any]:
    benchmark_resolution = _resolve_benchmark(
        ticker=ticker,
        period=period,
        interval=interval,
        validate=validate_benchmark,
        listing_days=listing_days,
        benchmark=benchmark,
        asset_df=adjusted_asset_df,
    )

    selected_benchmark = benchmark_resolution.get("selected_benchmark") or benchmark
    selected_benchmark_df = _pop_selected_benchmark_data(benchmark_resolution)

    cfg = CurveConfig(
        period=period,
        interval=interval,
        benchmark=selected_benchmark,
        auto_resolve_benchmark=False,
        validate_benchmark=False,
    )

    from app.ai.characteristic_curve_v1_7 import analyze_dataframe

    # CCE consumes the same adjusted asset dataframe that Benchmark Resolver
    # validated. The selected benchmark dataframe comes directly from the
    # resolver's in-memory result, avoiding a second Market Data request.
    result = dict(
        analyze_dataframe(
            ticker,
            adjusted_asset_df,
            benchmark_df=selected_benchmark_df,
            cfg=cfg,
            benchmark_resolution=benchmark_resolution,
        )
    )
    result["orchestrator_benchmark"] = selected_benchmark
    result["orchestrator_benchmark_resolution"] = benchmark_resolution
    result["orchestrator_listing_days"] = listing_days
    result["orchestrator_cce_version"] = result.get("version")
    result["orchestrator_appt_x_20d"] = _engine_appt_x_20d(result)
    result["orchestrator_levels"] = _engine_levels(result)
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

        # DATA CONTEXT
        # ------------
        # Fetch the asset once in each semantic price representation required
        # by the existing engines. Both reads are cache-backed by Market Data V2.
        #
        # raw_asset_df:
        #   Asset Age + NLE, preserving their historical auto_adjust=False
        #   behavior.
        #
        # adjusted_asset_df:
        #   Benchmark Resolver + CCE, preserving their historical
        #   auto_adjust=True behavior.
        #
        # This is deliberate: changing adjustment semantics here would alter
        # technical features and could change historical signals.
        raw_asset_df = _market_history(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
        )

        # IMPORTANT:
        # The orchestrator already owns the first raw Market Data request.
        # Never call resolve_asset_age() again when that request returns empty.
        # resolve_asset_age() would issue the same Market Data request a second
        # time, creating the duplicate-download path observed with BRK.B.
        #
        # Always resolve routing from the already-loaded dataframe. An empty
        # dataframe is a valid input to resolve_from_dataframe(); the resolver
        # returns route=NO_DATA without performing another Yahoo request.
        age = resolve_from_dataframe(
            symbol,
            raw_asset_df,
            config=resolver_cfg,
        )

        if raw_asset_df.empty:
            adjusted_asset_df = pd.DataFrame()
        else:
            adjusted_asset_df = _market_history(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
            )
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
                asset_df=raw_asset_df,
                adjusted_asset_df=(
                    adjusted_asset_df
                    if not adjusted_asset_df.empty
                    else None
                ),
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
                adjusted_asset_df=adjusted_asset_df,
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
        "appt_x_20d": summary["appt_x_20d"],
        "expected_return_20d": summary["expected_return_20d"],
        "levels": summary["levels"],
        "regime": summary["regime"],
        "data_quality": summary["data_quality"] or age.get("data_quality"),
        "benchmark": summary["benchmark"],
        "benchmark_trust": summary["benchmark_trust"],
        "benchmark_trust_decision_grade": summary[
            "benchmark_trust_decision_grade"
        ],
        "asset_age": age,
        "data_context": {
            "market_data_version": "2.0",
            "asset_raw_loaded": not raw_asset_df.empty,
            "asset_adjusted_loaded": not adjusted_asset_df.empty,
            "asset_rows_raw": int(len(raw_asset_df)),
            "asset_rows_adjusted": int(len(adjusted_asset_df)),
            "shared_asset_context": True,
            "adjustment_semantics_preserved": True,
        },
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


def _fmt(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for human-readable validation output."""
    if value is None:
        return "N/A"
    parsed = _safe_float(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    """Format a decimal expected return as a percentage."""
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed * 100:.{digits}f}%"


def _print_compact(result: Dict[str, Any]) -> None:
    age = result.get("asset_age") or {}
    levels = result.get("levels") or {}
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
    print(
        f"       | APPT_X_20D={_fmt(result.get('appt_x_20d'))} | "
        f"expected_return_20d={_fmt_pct(result.get('expected_return_20d'))} | "
        f"entry={_fmt(levels.get('entry_price'), 2)} | "
        f"zone={_fmt(levels.get('entry_zone_low'), 2)}-"
        f"{_fmt(levels.get('entry_zone_high'), 2)} | "
        f"stop={_fmt(levels.get('stop'), 2)} | "
        f"TP1={_fmt(levels.get('take_profit_1'), 2)} | "
        f"TP2={_fmt(levels.get('take_profit_2'), 2)} | "
        f"TP3={_fmt(levels.get('take_profit_3'), 2)}"
    )


def _print_v17_validation(results: List[Dict[str, Any]]) -> None:
    """Validate presentation of additive CCE V1.7 fields without changing analysis."""
    cce_results = [
        r for r in results
        if r.get("engine") == "CHARACTERISTIC_CURVE_ENGINE"
    ]
    nle_results = [
        r for r in results
        if r.get("engine") == "NEW_LISTING_ENGINE"
    ]

    fields = {
        "APPT_X_20D": lambda r: r.get("appt_x_20d"),
        "EXPECTED_RETURN_20D": lambda r: r.get("expected_return_20d"),
        "CURRENT_PRICE": lambda r: r.get("current_price"),
        "ENTRY_PRICE": lambda r: (r.get("levels") or {}).get("entry_price"),
        "ENTRY_ZONE_LOW": lambda r: (r.get("levels") or {}).get("entry_zone_low"),
        "ENTRY_ZONE_HIGH": lambda r: (r.get("levels") or {}).get("entry_zone_high"),
        "STOP": lambda r: (r.get("levels") or {}).get("stop"),
        "TP1": lambda r: (r.get("levels") or {}).get("take_profit_1"),
        "TP2": lambda r: (r.get("levels") or {}).get("take_profit_2"),
        "TP3": lambda r: (r.get("levels") or {}).get("take_profit_3"),
    }

    print("\n===== CCE V1.7 PRESENTATION / VALIDATION =====")
    print(f"CCE assets tested : {len(cce_results)}")
    print(f"NLE assets tested : {len(nle_results)}")

    for name, getter in fields.items():
        present = sum(getter(r) is not None for r in cce_results)
        total = len(cce_results)
        status = "PASS" if present == total else "PARTIAL"
        print(f"{name:<20}: {present:>2}/{total:<2} | {status}")

    transport_ok = all(
        r.get("engine") != "CHARACTERISTIC_CURVE_ENGINE"
        or (
            "appt_x_20d" in r
            and "expected_return_20d" in r
            and isinstance(r.get("levels"), dict)
            and "engine_result" in r
        )
        for r in results
    )

    print(f"V1.7 transport      : {'PASS' if transport_ok else 'FAIL'}")
    print("\n----- CCE V1.7 DATA -----")
    header = (
        "Ticker | APPT_X_20D | ExpRet | Entry | ZoneLow | ZoneHigh | "
        "Stop | TP1 | TP2 | TP3"
    )
    print(header)
    print("-" * len(header))

    for result in cce_results:
        levels = result.get("levels") or {}
        print(
            f"{result.get('ticker','?'):>6} | "
            f"{_fmt(result.get('appt_x_20d'), 4):>10} | "
            f"{_fmt_pct(result.get('expected_return_20d')):>6} | "
            f"{_fmt(levels.get('entry_price'), 2):>8} | "
            f"{_fmt(levels.get('entry_zone_low'), 2):>8} | "
            f"{_fmt(levels.get('entry_zone_high'), 2):>9} | "
            f"{_fmt(levels.get('stop'), 2):>8} | "
            f"{_fmt(levels.get('take_profit_1'), 2):>8} | "
            f"{_fmt(levels.get('take_profit_2'), 2):>8} | "
            f"{_fmt(levels.get('take_profit_3'), 2):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Trader Asset Analysis Orchestrator V1.5.1"
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
        help="Disable Benchmark Resolver validation",
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
        _print_v17_validation(results)
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
