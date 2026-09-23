"""
AI Trader - Production Validation Suite V1.2

Purpose
-------
Production pre-deployment regression gate comparing:

    V1.3
        app.ai.asset_analysis_orchestrator_v1_3

vs

    V1.4.2
        app.ai.asset_analysis_orchestrator_v1_4_2
        + Market Data V2
        + Benchmark Resolver V1.6.1
        + Asset Age Resolver V1.4

V1.2 changes from V1.1:
- Treats expected NLE version changes (1.3 -> 1.4.1) as ARCHITECTURE_DIFF.
- Treats the known SPCX NLE Close/Adj-Close correction as EXPECTED_DATA_PATH.
- Adds a narrowly scoped SPCX NLE market-data drift rule for return_5d and RSI.
- Preserves strict regression checking for all decision/routing/signal fields.
- Never treats an arbitrary numeric difference as acceptable merely because it is small.
- Handles pandas DataFrame / Series / numpy values safely.
- Adds repeatability/stability validation across two consecutive runs.
- Separates functional regressions from market-data drift between runs.
- Requires stable routing, engine, signal/action and benchmark/trust decisions.
- Produces a production-readiness gate for both functional and repeatability checks.
- Ignores internal benchmark DataFrame payloads rather than comparing them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# Production matrix
# ---------------------------------------------------------------------------

TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AVGO", "TSM",
    "META", "AAPL", "BRK.B", "JPM", "V", "MA",
    "TQQQ", "SOXL", "UPRO", "DFEN", "GDXU", "AGQ", "SPCX",
]

V1_3_MODULE = "app.ai.asset_analysis_orchestrator_v1_3"
V1_4_2_MODULE = "app.ai.asset_analysis_orchestrator_v1_4_2"

EXPECTED_MARKET_DATA_VERSION = "2.0"
EXPECTED_BENCHMARK_RESOLVER_VERSION = "1.6.1"
EXPECTED_ASSET_AGE_RESOLVER_VERSION = "1.4"
EXPECTED_NLE_VERSION = "1.4.1"

DEFAULT_STABILITY_RUNS = 2

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

REGRESSION = "REGRESSION"
ARCHITECTURE_DIFF = "ARCHITECTURE_DIFF"
EXPECTED_DATA_PATH = "EXPECTED_DATA_PATH"
MARKET_DATA_DRIFT = "MARKET_DATA_DRIFT"
UNEXPECTED_DIFF = "UNEXPECTED_DIFF"

# ---------------------------------------------------------------------------
# Known architecture-only fields
# ---------------------------------------------------------------------------

ARCHITECTURE_PREFIXES = (
    "data_context.",
    "engine_result.benchmark_resolution.",
    "engine_result.benchmark_",
    "engine_result.orchestrator_benchmark_resolution.",
    "engine_result.orchestrator_nle_version",
    "engine_result.orchestrator_asset_age_",
    "engine_result.orchestrator_benchmark_",
)

ARCHITECTURE_EXACT = {
    "version",
    "engine_version",
    "engine_result.version",
    "engine_result.engine_version",
    "engine_result.orchestrator_version",
}

# Expected NLE implementation-version migration.
NLE_VERSION_PATHS = {
    "engine_result.version",
    "engine_version",
    "engine_result.engine_version",
}

INTERNAL_OBJECT_SUFFIXES = (
    "._selected_benchmark_data",
    ".selected_benchmark_data",
    ".benchmark_data",
    ".asset_df",
    ".benchmark_df",
    "._asset_df",
    "._benchmark_df",
)

# ---------------------------------------------------------------------------
# Strict observable fields
# ---------------------------------------------------------------------------

CRITICAL_EXACT = {
    "status",
    "route",
    "engine",
    "engine_version",
    "signal",
    "action",
    "score",
    "confidence",
    "current_price",
    "regime",
    "data_quality",
    "benchmark",
    "benchmark_trust",
    "benchmark_trust_decision_grade",
    "asset_age.route",
    "asset_age.recommendation",
    "asset_age.engine",
    "asset_age.trading_days",
    "asset_age.valid_feature_rows",
    "asset_age.forward_complete_rows",
    "asset_age.max_independent_matches",
    "asset_age.cce_eligible",
    "asset_age.nle_eligible",
    "engine_result.signal",
    "engine_result.action",
    "engine_result.decision",
    "engine_result.score",
    "engine_result.confidence",
    "engine_result.regime",
    "engine_result.cycle",
    "engine_result.matches",
    "engine_result.primary_matches",
    "engine_result.fallback_matches",
    "engine_result.fallback_ratio",
    "engine_result.confirmation_required",
    "engine_result.confirmed",
    "engine_result.confirmation_score",
    "engine_result.classification",
    "engine_result.operational_signal",
    "engine_result.operational_status",
    "engine_result.operational_action",
    "engine_result.historical_probability_20d",
    "engine_result.historical_return_20d",
    "engine_result.historical_mfe_20d",
    "engine_result.historical_mae_20d",
    "engine_result.reward_risk_20d",
    "engine_result.rs_momentum",
    "engine_result.metrics.atr_pct",
    "engine_result.metrics.ema20",
    "engine_result.metrics.ema50",
    "engine_result.metrics.rsi",
    "engine_result.metrics.return_5d",
    "engine_result.metrics.return_20d",
    "engine_result.metrics.return_since_listing",
    "engine_result.metrics.volume_ratio",
    "engine_result.metrics.relative_strength_20d",
    "engine_result.metrics.drawdown_from_high",
    "engine_result.metrics.recovery_from_low",
    "engine_result.entry.current_price",
    "engine_result.entry.preferred_entry",
    "engine_result.entry.stop_loss",
    "engine_result.entry.tp_1",
    "engine_result.entry.tp_2",
    "engine_result.entry.tp_3",
    "engine_result.entry.zone_low",
    "engine_result.entry.zone_high",
    "engine_summary.signal",
    "engine_summary.action",
    "engine_summary.score",
    "engine_summary.confidence",
    "engine_summary.current_price",
    "engine_summary.regime",
    "engine_summary.benchmark",
    "engine_summary.benchmark_trust",
    "engine_summary.benchmark_trust_decision_grade",
}

# Small, field-specific tolerances. These are only for genuine sequential
# market-data drift, not for changed decisions.
MARKET_DATA_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "current_price": (0.20, 0.002),
    "engine_result.entry.current_price": (0.20, 0.002),
    "engine_result.entry.preferred_entry": (0.25, 0.002),
    "engine_result.entry.stop_loss": (0.30, 0.003),
    "engine_result.entry.tp_1": (0.35, 0.003),
    "engine_result.entry.tp_2": (0.40, 0.003),
    "engine_result.entry.tp_3": (0.50, 0.003),
    "engine_result.entry.zone_low": (0.30, 0.003),
    "engine_result.entry.zone_high": (0.30, 0.003),
    "engine_result.metrics.current_price": (0.20, 0.002),
    "engine_result.metrics.atr_pct": (0.05, 0.01),
    "engine_result.metrics.drawdown_from_high": (0.15, 0.01),
    "engine_result.metrics.ema20": (0.25, 0.002),
    "engine_result.metrics.ema50": (0.25, 0.002),
    "engine_result.metrics.recovery_from_low": (0.15, 0.01),
    "engine_result.metrics.return_5d": (0.15, 0.015),
    "engine_result.metrics.return_20d": (0.20, 0.015),
    "engine_result.metrics.return_since_listing": (0.20, 0.015),
    "engine_result.metrics.rsi": (0.15, 0.003),
    "engine_result.metrics.volume_ratio": (0.05, 0.05),
    "engine_summary.current_price": (0.20, 0.002),
}

# SPCX is the only ticker where the NLE Close/Adj-Close correction is known
# to change downstream metrics. Keep this exception deliberately narrow.
SPCX_NLE_DRIFT_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "engine_result.metrics.return_5d": (0.20, 0.02),
    "engine_result.metrics.rsi": (0.25, 0.005),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_import(name: str):
    return importlib.import_module(name)


def _version_of(module: Any) -> Optional[str]:
    for key in ("VERSION", "__version__", "VERSION_STRING"):
        value = getattr(module, key, None)
        if value is not None:
            return str(value)
    return None


def _is_nan(value: Any) -> bool:
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except Exception:
        return False


def _is_tabular(value: Any) -> bool:
    try:
        import pandas as pd
        import numpy as np
        return isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    except Exception:
        return False


def _values_equal(old: Any, new: Any) -> bool:
    if _is_tabular(old) or _is_tabular(new):
        if old is None or new is None:
            return old is None and new is None
        try:
            import pandas as pd
            if isinstance(old, pd.DataFrame) and isinstance(new, pd.DataFrame):
                return old.equals(new)
            if isinstance(old, pd.Series) and isinstance(new, pd.Series):
                return old.equals(new)
            import numpy as np
            if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
                return np.array_equal(old, new, equal_nan=True)
        except Exception:
            return False

    if _is_nan(old) and _is_nan(new):
        return True

    try:
        result = old == new
        if isinstance(result, bool):
            return result
        return bool(result)
    except Exception:
        return False


def _numeric_pair(old: Any, new: Any) -> Optional[Tuple[float, float]]:
    if isinstance(old, bool) or isinstance(new, bool):
        return None
    try:
        a = float(old)
        b = float(new)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def _within_tolerance(old: Any, new: Any, absolute: float, relative: float) -> bool:
    pair = _numeric_pair(old, new)
    if pair is None:
        return False
    a, b = pair
    delta = abs(a - b)
    scale = max(abs(a), abs(b), 1e-12)
    return delta <= absolute or (delta / scale) <= relative


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if _is_tabular(value):
        out[prefix.rstrip(".")] = value
        return out

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, path))
        return out

    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            path = f"{prefix}[{idx}]"
            out.update(_flatten(child, path))
        return out

    out[prefix.rstrip(".")] = value
    return out


def _is_internal_object_path(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in INTERNAL_OBJECT_SUFFIXES)


def _is_architecture_path(path: str) -> bool:
    return path in ARCHITECTURE_EXACT or any(
        path.startswith(prefix) for prefix in ARCHITECTURE_PREFIXES
    )


def _is_nle_version_path(path: str) -> bool:
    return path in NLE_VERSION_PATHS


def _is_spcx_nle_path(ticker: str, old_result: Any, new_result: Any, path: str) -> bool:
    if ticker.upper() != "SPCX":
        return False
    route = str(
        (new_result or {}).get("route", (old_result or {}).get("route", ""))
    ).upper()
    engine = str(
        (new_result or {}).get("engine", (old_result or {}).get("engine", ""))
    ).upper()
    return route == "NEW_LISTING" or "NEW_LISTING" in engine or path.startswith(
        "engine_result.metrics."
    )


def _classify(
    ticker: str,
    path: str,
    old: Any,
    new: Any,
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if _values_equal(old, new):
        return None

    if _is_internal_object_path(path):
        return {
            "classification": "IGNORED_INTERNAL_OBJECT",
            "reason": "Internal tabular payload is not an observable production output.",
        }

    if _is_nle_version_path(path):
        return {
            "classification": ARCHITECTURE_DIFF,
            "reason": "Expected NLE V1.3 -> V1.4.1 implementation-version migration.",
        }

    if _is_architecture_path(path):
        return {
            "classification": ARCHITECTURE_DIFF,
            "reason": "Expected V1.4.2 architecture/metadata difference.",
        }

    # Known SPCX Close/Adj-Close correction.
    if (
        ticker.upper() == "SPCX"
        and path == "engine_result.metrics.relative_strength_20d"
    ):
        return {
            "classification": EXPECTED_DATA_PATH,
            "reason": "Known SPCX NLE Close-vs-Adj-Close data-path correction.",
        }

    # Critical observable decision fields always remain regressions unless
    # the exact field is covered by a deliberately narrow drift rule.
    if ticker.upper() == "SPCX" and path in SPCX_NLE_DRIFT_TOLERANCES:
        absolute, relative = SPCX_NLE_DRIFT_TOLERANCES[path]
        if _within_tolerance(old, new, absolute, relative):
            return {
                "classification": MARKET_DATA_DRIFT,
                "reason": (
                    "Known SPCX NLE data-path drift within the V1.1 "
                    "field-specific tolerance."
                ),
            }

    if path in CRITICAL_EXACT:
        tolerance = MARKET_DATA_TOLERANCES.get(path)
        if tolerance and _within_tolerance(old, new, *tolerance):
            return {
                "classification": MARKET_DATA_DRIFT,
                "reason": "Small sequential market-data drift within field-specific tolerance.",
            }
        return {
            "classification": REGRESSION,
            "reason": "Critical observable behavior changed unexpectedly.",
        }

    tolerance = MARKET_DATA_TOLERANCES.get(path)
    if tolerance and _within_tolerance(old, new, *tolerance):
        return {
            "classification": MARKET_DATA_DRIFT,
            "reason": "Small sequential market-data drift within field-specific tolerance.",
        }

    return {
        "classification": UNEXPECTED_DIFF,
        "reason": "Non-critical difference not covered by an explicit rule.",
    }


def _extract_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    engine_summary = result.get("engine_summary") or {}
    engine_result = result.get("engine_result") or {}
    asset_age = result.get("asset_age") or {}

    metrics = engine_result.get("metrics") or {}
    entry = engine_result.get("entry") or {}

    return {
        "status": result.get("status"),
        "route": result.get("route"),
        "engine": result.get("engine"),
        "engine_version": result.get("engine_version"),
        "signal": result.get("signal"),
        "action": result.get("action"),
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "current_price": result.get("current_price"),
        "regime": result.get("regime"),
        "data_quality": result.get("data_quality"),
        "benchmark": result.get("benchmark"),
        "benchmark_trust": result.get("benchmark_trust"),
        "benchmark_trust_decision_grade": result.get("benchmark_trust_decision_grade"),
        "asset_age.route": asset_age.get("route"),
        "asset_age.recommendation": asset_age.get("recommendation"),
        "asset_age.engine": asset_age.get("engine"),
        "asset_age.trading_days": asset_age.get("trading_days"),
        "asset_age.valid_feature_rows": asset_age.get("valid_feature_rows"),
        "asset_age.forward_complete_rows": asset_age.get("forward_complete_rows"),
        "asset_age.max_independent_matches": asset_age.get("max_independent_matches"),
        "asset_age.cce_eligible": asset_age.get("cce_eligible"),
        "asset_age.nle_eligible": asset_age.get("nle_eligible"),
        "engine_result.signal": engine_result.get("signal"),
        "engine_result.action": engine_result.get("action"),
        "engine_result.decision": engine_result.get("decision"),
        "engine_result.score": engine_result.get("score"),
        "engine_result.confidence": engine_result.get("confidence"),
        "engine_result.regime": engine_result.get("regime"),
        "engine_result.cycle": engine_result.get("cycle"),
        "engine_result.matches": engine_result.get("matches"),
        "engine_result.primary_matches": engine_result.get("primary_matches"),
        "engine_result.fallback_matches": engine_result.get("fallback_matches"),
        "engine_result.fallback_ratio": engine_result.get("fallback_ratio"),
        "engine_result.confirmation_required": engine_result.get("confirmation_required"),
        "engine_result.confirmed": engine_result.get("confirmed"),
        "engine_result.confirmation_score": engine_result.get("confirmation_score"),
        "engine_result.classification": engine_result.get("classification"),
        "engine_result.operational_signal": engine_result.get("operational_signal"),
        "engine_result.operational_status": engine_result.get("operational_status"),
        "engine_result.operational_action": engine_result.get("operational_action"),
        "engine_result.historical_probability_20d": engine_result.get("historical_probability_20d"),
        "engine_result.historical_return_20d": engine_result.get("historical_return_20d"),
        "engine_result.historical_mfe_20d": engine_result.get("historical_mfe_20d"),
        "engine_result.historical_mae_20d": engine_result.get("historical_mae_20d"),
        "engine_result.reward_risk_20d": engine_result.get("reward_risk_20d"),
        "engine_result.rs_momentum": engine_result.get("rs_momentum"),
        "engine_result.metrics.atr_pct": metrics.get("atr_pct"),
        "engine_result.metrics.ema20": metrics.get("ema20"),
        "engine_result.metrics.ema50": metrics.get("ema50"),
        "engine_result.metrics.rsi": metrics.get("rsi"),
        "engine_result.metrics.return_5d": metrics.get("return_5d"),
        "engine_result.metrics.return_20d": metrics.get("return_20d"),
        "engine_result.metrics.return_since_listing": metrics.get("return_since_listing"),
        "engine_result.metrics.volume_ratio": metrics.get("volume_ratio"),
        "engine_result.metrics.relative_strength_20d": metrics.get("relative_strength_20d"),
        "engine_result.metrics.drawdown_from_high": metrics.get("drawdown_from_high"),
        "engine_result.metrics.recovery_from_low": metrics.get("recovery_from_low"),
        "engine_result.entry.current_price": entry.get("current_price"),
        "engine_result.entry.preferred_entry": entry.get("preferred_entry"),
        "engine_result.entry.stop_loss": entry.get("stop_loss"),
        "engine_result.entry.tp_1": entry.get("tp_1"),
        "engine_result.entry.tp_2": entry.get("tp_2"),
        "engine_result.entry.tp_3": entry.get("tp_3"),
        "engine_result.entry.zone_low": entry.get("zone_low"),
        "engine_result.entry.zone_high": entry.get("zone_high"),
        "engine_summary.signal": engine_summary.get("signal"),
        "engine_summary.action": engine_summary.get("action"),
        "engine_summary.score": engine_summary.get("score"),
        "engine_summary.confidence": engine_summary.get("confidence"),
        "engine_summary.current_price": engine_summary.get("current_price"),
        "engine_summary.regime": engine_summary.get("regime"),
        "engine_summary.benchmark": engine_summary.get("benchmark"),
        "engine_summary.benchmark_trust": engine_summary.get("benchmark_trust"),
        "engine_summary.benchmark_trust_decision_grade": engine_summary.get(
            "benchmark_trust_decision_grade"
        ),
    }


def compare_results(
    ticker: str,
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
) -> Dict[str, Any]:
    old_flat = _flatten(old_result)
    new_flat = _flatten(new_result)

    paths = sorted(set(old_flat) | set(new_flat))
    differences = []
    ignored_object_paths = []

    for path in paths:
        old = old_flat.get(path)
        new = new_flat.get(path)

        classification = _classify(
            ticker=ticker,
            path=path,
            old=old,
            new=new,
            old_result=old_result,
            new_result=new_result,
        )

        if classification is None:
            continue

        if classification["classification"] == "IGNORED_INTERNAL_OBJECT":
            ignored_object_paths.append(path)
            continue

        pair = _numeric_pair(old, new)
        abs_delta = None
        rel_delta = None
        if pair is not None:
            a, b = pair
            abs_delta = abs(a - b)
            rel_delta = abs_delta / max(abs(a), abs(b), 1e-12)

        differences.append({
            "path": path,
            "old": old,
            "new": new,
            "classification": classification["classification"],
            "reason": classification["reason"],
            "abs_delta": abs_delta,
            "rel_delta": rel_delta,
        })

    counts = {
        "regressions": sum(d["classification"] == REGRESSION for d in differences),
        "architecture_differences": sum(
            d["classification"] == ARCHITECTURE_DIFF for d in differences
        ),
        "expected_data_path_differences": sum(
            d["classification"] == EXPECTED_DATA_PATH for d in differences
        ),
        "market_data_drift": sum(
            d["classification"] == MARKET_DATA_DRIFT for d in differences
        ),
        "unexpected_differences": sum(
            d["classification"] == UNEXPECTED_DIFF for d in differences
        ),
        "ignored_object_paths": len(ignored_object_paths),
        "errors": 0,
    }

    passed = (
        counts["regressions"] == 0
        and counts["unexpected_differences"] == 0
        and counts["errors"] == 0
    )

    if not passed:
        classification = "REGRESSION" if counts["regressions"] else "UNEXPECTED_DIFF"
    elif counts["architecture_differences"] or counts["expected_data_path_differences"] or counts["market_data_drift"]:
        classification = "NON_CRITICAL_DIFF"
    else:
        classification = "MATCH"

    return {
        "status": classification,
        "pass": passed,
        "fields_compared": len(paths),
        "differences": len(differences),
        "regressions": counts["regressions"],
        "architecture_differences": counts["architecture_differences"],
        "expected_data_path_differences": counts["expected_data_path_differences"],
        "market_data_drift": counts["market_data_drift"],
        "unexpected_differences": counts["unexpected_differences"],
        "errors": counts["errors"],
        "ignored_object_paths": ignored_object_paths,
        "diffs": differences,
        "v1_3_summary": _extract_summary(old_result),
        "v1_4_2_summary": _extract_summary(new_result),
    }


def _call_orchestrator(module: Any, ticker: str, period: str, benchmark: Optional[str]):
    # Support the common entry points used by the project versions.
    candidates = ("analyze_ticker", "analyze_asset", "run_analysis", "analyze")
    for name in candidates:
        fn = getattr(module, name, None)
        if fn is None:
            continue

        attempts = [
            {"ticker": ticker, "period": period, "benchmark": benchmark},
            {"ticker": ticker, "period": period},
            {"ticker": ticker},
        ]

        for kwargs in attempts:
            try:
                return fn(**kwargs)
            except TypeError:
                continue

    raise AttributeError(
        f"No supported orchestrator entry point found in {module.__name__}: "
        + ", ".join(candidates)
    )


def _normalize_result(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        converted = result.to_dict()
        if isinstance(converted, dict):
            return converted
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError(f"Unsupported orchestrator result type: {type(result)!r}")


def preflight() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    for module_name, expected in (
        (V1_3_MODULE, "1.3"),
        (V1_4_2_MODULE, "1.4.2"),
    ):
        try:
            module = _safe_import(module_name)
            checks[module_name] = {
                "ok": True,
                "module_version": _version_of(module),
                "expected_version": expected,
            }
        except Exception as exc:
            checks[module_name] = {
                "ok": False,
                "error": str(exc),
            }

    try:
        module = _safe_import("app.data.market_data")
        checks["app.data.market_data"] = {
            "ok": True,
            "version": _version_of(module),
            "expected_version": EXPECTED_MARKET_DATA_VERSION,
        }
    except Exception as exc:
        checks["app.data.market_data"] = {"ok": False, "error": str(exc)}

    try:
        module = _safe_import("app.ai.benchmark_resolver_v1_6_1")
        checks["benchmark_resolver"] = {
            "ok": True,
            "version": _version_of(module),
            "expected_version": EXPECTED_BENCHMARK_RESOLVER_VERSION,
        }
    except Exception as exc:
        checks["benchmark_resolver"] = {"ok": False, "error": str(exc)}

    try:
        module = _safe_import("app.ai.asset_age_resolver_v1_4")
        checks["asset_age_resolver"] = {
            "ok": True,
            "version": _version_of(module),
            "expected_version": EXPECTED_ASSET_AGE_RESOLVER_VERSION,
        }
    except Exception as exc:
        checks["asset_age_resolver"] = {"ok": False, "error": str(exc)}

    return checks


def _preflight_ok(checks: Dict[str, Any]) -> bool:
    for item in checks.values():
        if not item.get("ok"):
            return False
        expected = item.get("expected_version")
        actual = item.get("module_version", item.get("version"))
        if expected is not None and actual is not None and str(actual) != str(expected):
            return False
    return True


def run_suite(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)

    preflight_result = preflight()

    report: Dict[str, Any] = {
        "suite": "Production Validation Suite",
        "version": "1.2",
        "generated_at": started.isoformat(),
        "v1_3_module": V1_3_MODULE,
        "v1_4_2_module": V1_4_2_MODULE,
        "market_data_layer": {
            "ok": preflight_result.get("app.data.market_data", {}).get("ok", False),
            "version": preflight_result.get("app.data.market_data", {}).get("version"),
        },
        "benchmark_resolver": {
            "ok": preflight_result.get("benchmark_resolver", {}).get("ok", False),
            "version": preflight_result.get("benchmark_resolver", {}).get("version"),
        },
        "asset_age_resolver": {
            "ok": preflight_result.get("asset_age_resolver", {}).get("ok", False),
            "version": preflight_result.get("asset_age_resolver", {}).get("version"),
        },
        "tickers": list(tickers),
        "benchmark": benchmark,
        "period": period,
        "interval": interval,
        "preflight": preflight_result,
        "results": [],
    }

    try:
        old_module = _safe_import(V1_3_MODULE)
        new_module = _safe_import(V1_4_2_MODULE)
    except Exception:
        report["overall"] = "FAIL"
        report["pass"] = False
        report["production_gate"] = {
            "REGRESSION": 0,
            "UNEXPECTED_DIFF": 0,
            "ERROR": 1,
            "PASS_CONDITION": "REGRESSION=0 AND UNEXPECTED_DIFF=0 AND ERROR=0",
        }
        report["totals"] = {
            "regressions": 0,
            "architecture_differences": 0,
            "expected_data_path_differences": 0,
            "market_data_drift": 0,
            "unexpected_differences": 0,
            "errors": 1,
        }
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        return report

    for ticker in tickers:
        ticker_started = datetime.now(timezone.utc)
        item: Dict[str, Any] = {"ticker": ticker}

        try:
            old_raw = _call_orchestrator(old_module, ticker, period, benchmark)
            old_result = _normalize_result(old_raw)

            new_raw = _call_orchestrator(new_module, ticker, period, benchmark)
            new_result = _normalize_result(new_raw)

            comparison = compare_results(ticker, old_result, new_result)

            item.update({
                "classification": comparison["status"],
                "pass": comparison["pass"],
                "counts": {
                    "regressions": comparison["regressions"],
                    "architecture_differences": comparison["architecture_differences"],
                    "expected_data_path_differences": comparison[
                        "expected_data_path_differences"
                    ],
                    "market_data_drift": comparison["market_data_drift"],
                    "unexpected_differences": comparison["unexpected_differences"],
                    "ignored_object_paths": comparison["ignored_object_paths"],
                    "errors": comparison["errors"],
                },
                "differences": comparison["diffs"],
                "ignored_object_paths": comparison["ignored_object_paths"],
                "v1_3_summary": comparison["v1_3_summary"],
                "v1_4_2_summary": comparison["v1_4_2_summary"],
            })
        except Exception as exc:
            item.update({
                "classification": "ERROR",
                "pass": False,
                "counts": {
                    "regressions": 0,
                    "architecture_differences": 0,
                    "expected_data_path_differences": 0,
                    "market_data_drift": 0,
                    "unexpected_differences": 0,
                    "ignored_object_paths": 0,
                    "errors": 1,
                },
                "differences": [],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

        item["started_at"] = ticker_started.isoformat()
        item["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["results"].append(item)

    totals = {
        "regressions": sum(x["counts"]["regressions"] for x in report["results"]),
        "architecture_differences": sum(
            x["counts"]["architecture_differences"] for x in report["results"]
        ),
        "expected_data_path_differences": sum(
            x["counts"]["expected_data_path_differences"] for x in report["results"]
        ),
        "market_data_drift": sum(
            x["counts"]["market_data_drift"] for x in report["results"]
        ),
        "unexpected_differences": sum(
            x["counts"]["unexpected_differences"] for x in report["results"]
        ),
        "errors": sum(x["counts"]["errors"] for x in report["results"]),
    }

    report["totals"] = totals
    report["production_gate"] = {
        "REGRESSION": totals["regressions"],
        "UNEXPECTED_DIFF": totals["unexpected_differences"],
        "ERROR": totals["errors"],
        "PASS_CONDITION": "REGRESSION=0 AND UNEXPECTED_DIFF=0 AND ERROR=0",
    }

    report["overall"] = (
        "PASS"
        if totals["regressions"] == 0
        and totals["unexpected_differences"] == 0
        and totals["errors"] == 0
        and _preflight_ok(preflight_result)
        else "FAIL"
    )
    report["pass"] = report["overall"] == "PASS"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

    return report


def self_test() -> bool:
    base = {
        "route": "MATURE",
        "engine": "CHARACTERISTIC_CURVE_ENGINE",
        "signal": "WATCH",
        "action": "WATCH",
        "score": 60.0,
        "confidence": 55.0,
        "engine_result": {
            "metrics": {
                "rsi": 59.33,
                "return_5d": 5.93,
                "relative_strength_20d": 10.0,
            }
        },
    }

    new = json.loads(json.dumps(base))
    new["engine_result"]["metrics"]["rsi"] = 59.52
    new["engine_result"]["metrics"]["return_5d"] = 6.10
    new["engine_result"]["metrics"]["relative_strength_20d"] = 10.27
    new["engine_result"]["version"] = "1.4.1"
    base["engine_result"]["version"] = "1.3"

    result = compare_results("SPCX", base, new)
    classes = {d["path"]: d["classification"] for d in result["diffs"]}

    assert classes["engine_result.version"] == ARCHITECTURE_DIFF
    assert classes["engine_result.metrics.relative_strength_20d"] == EXPECTED_DATA_PATH
    assert classes["engine_result.metrics.rsi"] == MARKET_DATA_DRIFT
    assert classes["engine_result.metrics.return_5d"] == MARKET_DATA_DRIFT
    assert result["pass"] is True

    # True decision regression must still fail.
    reg_new = json.loads(json.dumps(base))
    reg_new["signal"] = "BUY"
    reg = compare_results("SPCX", base, reg_new)
    assert reg["regressions"] == 1
    assert reg["pass"] is False

    # Unknown difference must fail.
    unk_new = json.loads(json.dumps(base))
    unk_new["some_new_field"] = "unexpected"
    unk = compare_results("SPCX", base, unk_new)
    assert unk["unexpected_differences"] == 1
    assert unk["pass"] is False

    # DataFrame-safe equality.
    try:
        import pandas as pd
        df1 = pd.DataFrame({"Close": [1.0, 2.0]})
        df2 = pd.DataFrame({"Close": [1.0, 2.0]})
        assert _values_equal(df1, df2) is True
    except ImportError:
        pass

    # Stability: market price changes are intentionally ignored.
    stable_a = json.loads(json.dumps(base))
    stable_b = json.loads(json.dumps(base))
    stable_a["current_price"] = 100.0
    stable_b["current_price"] = 101.0
    stable = compare_stability("TEST", stable_a, stable_b)
    assert stable["pass"] is True

    # Stability: a decision/routing change must fail.
    unstable_b = json.loads(json.dumps(base))
    unstable_b["signal"] = "BUY"
    unstable = compare_stability("TEST", base, unstable_b)
    assert unstable["pass"] is False
    assert unstable["decision_regressions"] >= 1

    return True


def _stability_key(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fields that must remain stable between consecutive production runs.

    Numeric market values are intentionally excluded here because Yahoo data
    can legitimately move between executions. Decision/routing fields are
    strict.
    """
    keys = (
        "status",
        "route",
        "engine",
        "engine_version",
        "signal",
        "action",
        "regime",
        "data_quality",
        "benchmark",
        "benchmark_trust",
        "benchmark_trust_decision_grade",
        "asset_age.route",
        "asset_age.recommendation",
        "asset_age.engine",
        "asset_age.cce_eligible",
        "asset_age.nle_eligible",
        "engine_result.signal",
        "engine_result.action",
        "engine_result.decision",
        "engine_result.cycle",
        "engine_result.confirmation_required",
        "engine_result.confirmed",
        "engine_result.classification",
        "engine_result.operational_signal",
        "engine_result.operational_status",
        "engine_result.operational_action",
    )
    return {k: summary.get(k) for k in keys}


def compare_stability(
    ticker: str,
    first_result: Dict[str, Any],
    second_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two consecutive executions of the same orchestrator version.

    Goal: detect instability in production decisions while allowing normal
    market-data movement between executions.
    """
    first = _extract_summary(first_result)
    second = _extract_summary(second_result)

    first_key = _stability_key(first)
    second_key = _stability_key(second)

    diffs = []
    for path in sorted(set(first_key) | set(second_key)):
        old = first_key.get(path)
        new = second_key.get(path)
        if not _values_equal(old, new):
            diffs.append({
                "path": path,
                "old": old,
                "new": new,
                "classification": REGRESSION,
                "reason": "Production decision/routing instability between consecutive runs.",
            })

    passed = len(diffs) == 0

    return {
        "ticker": ticker,
        "pass": passed,
        "classification": "STABLE" if passed else "UNSTABLE",
        "decision_regressions": len(diffs),
        "differences": diffs,
        "first_summary": first,
        "second_summary": second,
    }


def run_stability_suite(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute V1.4.2 twice consecutively and compare decision/routing outputs.

    This deliberately does not require current_price/RSI/EMA/etc. to be
    identical because market data may change between executions.
    """
    started = datetime.now(timezone.utc)
    module = _safe_import(V1_4_2_MODULE)

    results = []

    for ticker in tickers:
        item = {"ticker": ticker}
        try:
            first_raw = _call_orchestrator(module, ticker, period, benchmark)
            first_result = _normalize_result(first_raw)

            second_raw = _call_orchestrator(module, ticker, period, benchmark)
            second_result = _normalize_result(second_raw)

            comparison = compare_stability(
                ticker, first_result, second_result
            )
            item.update(comparison)
        except Exception as exc:
            item.update({
                "pass": False,
                "classification": "ERROR",
                "decision_regressions": 0,
                "differences": [],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

        results.append(item)

    unstable = sum(not r["pass"] for r in results)
    errors = sum(r["classification"] == "ERROR" for r in results)

    return {
        "suite": "Production Validation Suite - Stability",
        "version": "1.2",
        "generated_at": started.isoformat(),
        "module": V1_4_2_MODULE,
        "period": period,
        "interval": interval,
        "benchmark": benchmark,
        "runs_per_ticker": 2,
        "tickers": list(tickers),
        "results": results,
        "totals": {
            "stable": sum(r["pass"] for r in results),
            "unstable": unstable,
            "errors": errors,
        },
        "production_gate": {
            "DECISION_INSTABILITY": unstable,
            "ERROR": errors,
            "PASS_CONDITION": "DECISION_INSTABILITY=0 AND ERROR=0",
        },
        "overall": "PASS" if unstable == 0 and errors == 0 else "FAIL",
        "pass": unstable == 0 and errors == 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Trader Production Validation Suite V1.2")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--json", default="production_validation_v1_1.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stability", action="store_true",
                        help="Run V1.4.2 twice consecutively and validate decision stability.")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("Production Validation Suite V1.2 self-test: PASS")
        return 0

    if args.preflight:
        result = preflight()
        print(json.dumps(result, indent=2, default=str))
        return 0 if _preflight_ok(result) else 1

    if args.stability:
        report = run_stability_suite(
            tickers=args.tickers,
            period=args.period,
            interval=args.interval,
            benchmark=args.benchmark,
        )

        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

        print("=" * 78)
        print("AI TRADER - PRODUCTION VALIDATION SUITE V1.2 STABILITY")
        print("=" * 78)
        print(f"Overall : {report['overall']}")
        print(f"Pass    : {report['pass']}")
        print()
        print("Production Stability Gate")
        print(f"  DECISION_INSTABILITY : {report['production_gate']['DECISION_INSTABILITY']}")
        print(f"  ERROR                : {report['production_gate']['ERROR']}")
        print()
        print("Per ticker")
        print("-" * 78)
        for item in report["results"]:
            print(
                f"{item['ticker']:7s} "
                f"{item['classification']:10s} "
                f"PASS={str(item['pass']):5s} "
                f"DECISION_DIFF={item.get('decision_regressions', 0):2d}"
            )
        print("-" * 78)
        print(f"JSON: {args.json}")

        return 0 if report["pass"] else 1

    report = run_suite(
        tickers=args.tickers,
        period=args.period,
        interval=args.interval,
        benchmark=args.benchmark,
    )

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    print("=" * 78)
    print("AI TRADER - PRODUCTION VALIDATION SUITE V1.1")
    print("=" * 78)
    print(f"Overall : {report['overall']}")
    print(f"Pass    : {report['pass']}")
    print()
    print("Production Gate")
    print(f"  REGRESSION      : {report['production_gate']['REGRESSION']}")
    print(f"  UNEXPECTED_DIFF : {report['production_gate']['UNEXPECTED_DIFF']}")
    print(f"  ERROR           : {report['production_gate']['ERROR']}")
    print()
    print("Per ticker")
    print("-" * 78)
    for item in report["results"]:
        c = item["counts"]
        print(
            f"{item['ticker']:7s} "
            f"{item['classification']:18s} "
            f"PASS={str(item['pass']):5s} "
            f"REG={c['regressions']:2d} "
            f"ARCH={c['architecture_differences']:3d} "
            f"PATH={c['expected_data_path_differences']:2d} "
            f"DRIFT={c['market_data_drift']:2d} "
            f"UNEXP={c['unexpected_differences']:2d} "
            f"ERR={c['errors']:1d}"
        )
    print("-" * 78)
    print(f"JSON: {args.json}")

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
