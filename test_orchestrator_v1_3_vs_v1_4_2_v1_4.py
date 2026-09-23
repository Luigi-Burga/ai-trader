#!/usr/bin/env python3
"""
test_orchestrator_v1_3_vs_v1_4_2_v1_4.py

Regression comparator for Asset Analysis Orchestrator V1.3 vs V1.4.2.

V1.4 improvements:
- Safely compares scalar values, dicts, lists, pandas Series/DataFrames,
  numpy arrays and other complex objects.
- Prevents "truth value of a DataFrame is ambiguous".
- Ignores internal non-decision DataFrame/object payloads from the flattened
  comparison when they are not part of the observable contract.
- Preserves explicit classifications:
    MATCH
    REGRESSION
    ARCHITECTURE_DIFF
    EXPECTED_DATA_PATH
    MARKET_DATA_DRIFT
    UNEXPECTED_DIFF
    ERROR
- PASS requires zero regressions, zero unexpected differences and zero errors.

Usage:
    python -m test_orchestrator_v1_3_vs_v1_4_2_v1_4 --self-test

    python -m test_orchestrator_v1_3_vs_v1_4_2_v1_4 \
        --tickers NVDA MSFT GOOGL SPCX

    python -m test_orchestrator_v1_3_vs_v1_4_2_v1_4 \
        --tickers NVDA MSFT GOOGL SPCX \
        --json comparison_orchestrator_v1_3_vs_v1_4_2_v1_4.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VERSION = "1.4"
V1_3_MODULE = "app.ai.asset_analysis_orchestrator_v1_3"
V1_4_2_MODULE = "app.ai.asset_analysis_orchestrator_v1_4_2"


# ---------------------------------------------------------------------------
# Optional pandas / numpy support
# ---------------------------------------------------------------------------

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import numpy as np
except Exception:
    np = None


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------

ARCHITECTURE_PREFIXES = (
    "data_context.",
    "engine_result.benchmark_resolution.",
    "engine_result.benchmark_",
    "engine_result.orchestrator_",
)

ARCHITECTURE_EXACT = {
    "version",
    "engine_version",
    "engine_result.version",
}

EXPECTED_DATA_PATH_FIELDS = {
    "engine_result.metrics.relative_strength_20d",
}

EXPECTED_DATA_PATH_TICKERS = {"SPCX"}
EXPECTED_DATA_PATH_ROUTES = {"NEW_LISTING"}
EXPECTED_DATA_PATH_ENGINES = {
    "NEW_LISTING_ENGINE",
    "NEW_LISTING_ENGINE_V1_3",
    "NEW_LISTING_ENGINE_V1_4_1",
}

CRITICAL_EXACT_PATHS = {
    "status",
    "route",
    "engine",
    "benchmark",
    "benchmark_trust",
    "benchmark_trust_decision_grade",
    "signal",
    "action",
    "score",
    "confidence",
    "regime",
    "cycle",
    "engine_result.signal",
    "engine_result.decision",
    "engine_result.score",
    "engine_result.confidence",
    "engine_result.regime",
    "engine_result.cycle",
    "engine_result.benchmark",
    "engine_result.benchmark_trust",
    "engine_result.benchmark_trust_decision_grade",
    "engine_result.confirmation_required",
    "engine_result.confirmed",
    "engine_result.confirmation_score",
    "engine_result.confirmation_decision",
    "engine_result.classification",
    "engine_result.operational_signal",
    "engine_result.operational_decision",
    "asset_age.route",
    "asset_age.recommendation",
    "asset_age.eligibility",
    "asset_age.eligible",
}

MARKET_DATA_DRIFT_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "current_price": (0.20, 0.0020),
    "engine_result.entry.current_price": (0.20, 0.0020),
    "engine_result.entry.preferred_entry": (0.25, 0.0020),
    "engine_result.entry.stop_loss": (0.30, 0.0030),
    "engine_result.entry.tp_1": (0.35, 0.0030),
    "engine_result.entry.tp_2": (0.40, 0.0030),
    "engine_result.entry.tp_3": (0.50, 0.0030),
    "engine_result.entry.zone_low": (0.30, 0.0030),
    "engine_result.entry.zone_high": (0.30, 0.0030),
    "engine_result.metrics.current_price": (0.20, 0.0020),
    "engine_result.metrics.atr_pct": (0.05, 0.0100),
    "engine_result.metrics.drawdown_from_high": (0.15, 0.0100),
    "engine_result.metrics.ema20": (0.25, 0.0020),
    "engine_result.metrics.ema50": (0.25, 0.0020),
    "engine_result.metrics.recovery_from_low": (0.15, 0.0100),
    "engine_result.metrics.return_5d": (0.15, 0.0150),
    "engine_result.metrics.return_20d": (0.20, 0.0150),
    "engine_result.metrics.return_since_listing": (0.20, 0.0150),
    "engine_result.metrics.rsi": (0.15, 0.0030),
    "engine_result.metrics.volume_ratio": (0.05, 0.05),
    "engine_summary.current_price": (0.20, 0.0020),
}

MARKET_DATA_DRIFT_SUFFIXES = (
    ".metrics.current_price",
    ".metrics.atr_pct",
    ".metrics.drawdown_from_high",
    ".metrics.ema20",
    ".metrics.ema50",
    ".metrics.recovery_from_low",
    ".metrics.return_5d",
    ".metrics.return_20d",
    ".metrics.return_since_listing",
    ".metrics.rsi",
    ".metrics.volume_ratio",
    ".entry.current_price",
    ".entry.preferred_entry",
    ".entry.stop_loss",
    ".entry.tp_1",
    ".entry.tp_2",
    ".entry.tp_3",
    ".entry.zone_low",
    ".entry.zone_high",
)

# These are implementation payloads rather than observable outputs.
# If a future orchestrator exposes a DataFrame here, comparing the object
# itself is not useful for regression testing.
IGNORED_OBJECT_PATH_SUFFIXES = (
    "._selected_benchmark_data",
    ".selected_benchmark_data",
    ".benchmark_data",
    ".asset_df",
    ".benchmark_df",
    "._asset_df",
    "._benchmark_df",
)


@dataclass
class Diff:
    path: str
    old: Any
    new: Any
    classification: str
    reason: str
    abs_delta: Optional[float] = None
    rel_delta: Optional[float] = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_complex_tabular(value: Any) -> bool:
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        return True
    if np is not None and isinstance(value, np.ndarray):
        return True
    return False


def _safe_scalar(value: Any) -> Any:
    """
    Convert common numpy scalar values into native Python values so that
    JSON serialization and equality checks remain deterministic.
    """
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _values_equal(old: Any, new: Any) -> bool:
    """
    Safe equality for scalar and complex objects.

    Crucially, this never evaluates a pandas DataFrame/Series directly in
    boolean context, avoiding:
        ValueError: The truth value of a DataFrame is ambiguous
    """
    if old is new:
        return True

    old = _safe_scalar(old)
    new = _safe_scalar(new)

    if old is None or new is None:
        return old is None and new is None

    if pd is not None:
        if isinstance(old, pd.DataFrame) and isinstance(new, pd.DataFrame):
            try:
                return bool(old.equals(new))
            except Exception:
                return False

        if isinstance(old, pd.Series) and isinstance(new, pd.Series):
            try:
                return bool(old.equals(new))
            except Exception:
                return False

        # Timestamp / Timedelta equality is scalar.
        if isinstance(old, (pd.Timestamp, pd.Timedelta)) or isinstance(
            new, (pd.Timestamp, pd.Timedelta)
        ):
            try:
                result = old == new
                return isinstance(result, bool) and result
            except Exception:
                return False

    if np is not None:
        if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
            try:
                return bool(np.array_equal(old, new, equal_nan=True))
            except Exception:
                return False

    if isinstance(old, Mapping) and isinstance(new, Mapping):
        if set(old.keys()) != set(new.keys()):
            return False
        return all(_values_equal(old[k], new[k]) for k in old.keys())

    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(old) != len(new):
            return False
        return all(_values_equal(a, b) for a, b in zip(old, new))

    try:
        result = old == new
    except Exception:
        return False

    if isinstance(result, bool):
        return result

    # numpy bool_ or scalar-like boolean
    if np is not None and isinstance(result, np.bool_):
        return bool(result)

    # Never call bool() on DataFrame/Series/ndarray.
    if _is_complex_tabular(result):
        return False

    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _numeric_delta(a: Any, b: Any) -> Tuple[Optional[float], Optional[float]]:
    a = _safe_scalar(a)
    b = _safe_scalar(b)

    if not (_is_number(a) and _is_number(b)):
        return None, None

    a = float(a)
    b = float(b)

    if not (math.isfinite(a) and math.isfinite(b)):
        return None, None

    delta = abs(b - a)
    scale = max(abs(a), abs(b), 1e-12)
    return delta, delta / scale


def _within_tolerance(path: str, old: Any, new: Any) -> bool:
    if not (_is_number(old) and _is_number(new)):
        return False

    tol = MARKET_DATA_DRIFT_TOLERANCES.get(path)
    if tol is None:
        return False

    abs_tol, rel_tol = tol
    abs_delta, rel_delta = _numeric_delta(old, new)

    if abs_delta is None or rel_delta is None:
        return False

    return abs_delta <= abs_tol or rel_delta <= rel_tol


def _architecture_match(path: str) -> bool:
    if path in ARCHITECTURE_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in ARCHITECTURE_PREFIXES)


def _expected_data_path_match(
    path: str,
    ticker: str,
    old_result: Mapping[str, Any],
    new_result: Mapping[str, Any],
) -> bool:
    if path not in EXPECTED_DATA_PATH_FIELDS:
        return False

    if ticker.upper() not in EXPECTED_DATA_PATH_TICKERS:
        return False

    old_route = str(
        _get_path(old_result, "route")
        or _get_path(old_result, "asset_age.route")
        or ""
    ).upper()

    new_route = str(
        _get_path(new_result, "route")
        or _get_path(new_result, "asset_age.route")
        or ""
    ).upper()

    old_engine = str(
        _get_path(old_result, "engine")
        or _get_path(old_result, "engine_result.engine")
        or ""
    ).upper()

    new_engine = str(
        _get_path(new_result, "engine")
        or _get_path(new_result, "engine_result.engine")
        or ""
    ).upper()

    route_ok = (
        not old_route
        or not new_route
        or (
            old_route in EXPECTED_DATA_PATH_ROUTES
            and new_route in EXPECTED_DATA_PATH_ROUTES
        )
    )

    engine_ok = (
        not old_engine
        or not new_engine
        or (
            old_engine in EXPECTED_DATA_PATH_ENGINES
            and new_engine in EXPECTED_DATA_PATH_ENGINES
        )
    )

    return route_ok and engine_ok


def _market_data_path_candidate(path: str) -> bool:
    if path in MARKET_DATA_DRIFT_TOLERANCES:
        return True
    return any(path.endswith(suffix) for suffix in MARKET_DATA_DRIFT_SUFFIXES)


def _should_ignore_object_path(path: str, old: Any, new: Any) -> bool:
    if not (_is_complex_tabular(old) or _is_complex_tabular(new)):
        return False

    return any(path.endswith(suffix) for suffix in IGNORED_OBJECT_PATH_SUFFIXES)


def classify_diff(
    path: str,
    old: Any,
    new: Any,
    ticker: str,
    old_result: Mapping[str, Any],
    new_result: Mapping[str, Any],
) -> Diff:
    abs_delta, rel_delta = _numeric_delta(old, new)

    if _architecture_match(path):
        return Diff(
            path,
            old,
            new,
            "ARCHITECTURE_DIFF",
            "Known V1.4.2 architecture/metadata change.",
            abs_delta,
            rel_delta,
        )

    if _expected_data_path_match(
        path, ticker, old_result, new_result
    ):
        return Diff(
            path,
            old,
            new,
            "EXPECTED_DATA_PATH",
            "Known SPCX NLE Close-vs-Adj-Close data-path correction.",
            abs_delta,
            rel_delta,
        )

    if path in CRITICAL_EXACT_PATHS:
        return Diff(
            path,
            old,
            new,
            "REGRESSION",
            "Observable decision/routing field changed.",
            abs_delta,
            rel_delta,
        )

    if _market_data_path_candidate(path) and _within_tolerance(
        path, old, new
    ):
        return Diff(
            path,
            old,
            new,
            "MARKET_DATA_DRIFT",
            "Small field-specific numerical difference consistent with "
            "sequential market-data snapshots.",
            abs_delta,
            rel_delta,
        )

    if path.startswith("asset_age.") and path.endswith(
        (
            ".trading_days",
            ".valid_feature_rows",
            ".forward_complete_rows",
            ".max_independent_matches",
        )
    ):
        return Diff(
            path,
            old,
            new,
            "REGRESSION",
            "Asset-age eligibility/history changed.",
            abs_delta,
            rel_delta,
        )

    return Diff(
        path,
        old,
        new,
        "UNEXPECTED_DIFF",
        "Difference is not covered by an explicit comparator rule.",
        abs_delta,
        rel_delta,
    )


def _get_path(obj: Mapping[str, Any], path: str) -> Any:
    cur: Any = obj

    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]

    return cur


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten mappings/lists without recursively expanding DataFrames, Series
    or numpy arrays.

    Tabular objects remain leaf values so that _values_equal() can compare
    them safely. Internal tabular payloads can subsequently be ignored by
    _should_ignore_object_path().
    """
    out: Dict[str, Any] = {}

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key = str(key)
            path = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(value, path))

    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            out.update(_flatten(value, path))

    elif _is_complex_tabular(obj):
        out[prefix] = obj

    else:
        out[prefix] = _safe_scalar(obj)

    return out


def compare_results(
    ticker: str,
    old_result: Mapping[str, Any],
    new_result: Mapping[str, Any],
) -> Dict[str, Any]:
    old_flat = _flatten(old_result)
    new_flat = _flatten(new_result)

    all_paths = sorted(set(old_flat) | set(new_flat))
    diffs: List[Diff] = []
    ignored_object_paths: List[str] = []

    for path in all_paths:
        old = old_flat.get(path)
        new = new_flat.get(path)

        if _should_ignore_object_path(path, old, new):
            ignored_object_paths.append(path)
            continue

        if _values_equal(old, new):
            continue

        diffs.append(
            classify_diff(
                path,
                old,
                new,
                ticker,
                old_result,
                new_result,
            )
        )

    counts = {
        "regressions": sum(
            d.classification == "REGRESSION" for d in diffs
        ),
        "architecture_differences": sum(
            d.classification == "ARCHITECTURE_DIFF" for d in diffs
        ),
        "expected_data_path_differences": sum(
            d.classification == "EXPECTED_DATA_PATH" for d in diffs
        ),
        "market_data_drift": sum(
            d.classification == "MARKET_DATA_DRIFT" for d in diffs
        ),
        "unexpected_differences": sum(
            d.classification == "UNEXPECTED_DIFF" for d in diffs
        ),
        "ignored_object_paths": len(ignored_object_paths),
        "errors": 0,
    }

    if counts["regressions"] > 0:
        classification = "REGRESSION"
    elif counts["unexpected_differences"] > 0:
        classification = "UNEXPECTED_DIFF"
    elif diffs:
        classification = "NON_CRITICAL_DIFF"
    else:
        classification = "MATCH"

    passed = (
        counts["regressions"] == 0
        and counts["unexpected_differences"] == 0
        and counts["errors"] == 0
    )

    return {
        "ticker": ticker,
        "classification": classification,
        "pass": passed,
        "counts": counts,
        "ignored_object_paths": ignored_object_paths,
        "differences": [
            {
                "path": d.path,
                "old": _json_safe(d.old),
                "new": _json_safe(d.new),
                "classification": d.classification,
                "reason": d.reason,
                "abs_delta": d.abs_delta,
                "rel_delta": d.rel_delta,
            }
            for d in diffs
        ],
    }


def _json_safe(value: Any) -> Any:
    """
    Make report values JSON-safe without trying to serialize large internal
    tabular payloads.
    """
    value = _safe_scalar(value)

    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd is not None and isinstance(value, pd.Timedelta):
        return str(value)

    if pd is not None and isinstance(value, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "shape": list(value.shape),
            "columns": [str(c) for c in value.columns],
        }

    if pd is not None and isinstance(value, pd.Series):
        return {
            "__type__": "Series",
            "shape": list(value.shape),
            "name": str(value.name),
        }

    if np is not None and isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return repr(value)


def _extract_runner(module: Any):
    candidates = (
        "analyze_ticker",
        "analyze_asset",
        "analyze",
        "run_analysis",
        "run",
        "orchestrate",
    )

    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, name

    raise AttributeError(
        f"No supported orchestrator entry point found in "
        f"{module.__name__}; tried {', '.join(candidates)}"
    )


def _call_runner(
    fn,
    ticker: str,
    benchmark: Optional[str],
    period: str,
):
    attempts = [
        {
            "ticker": ticker,
            "benchmark": benchmark,
            "period": period,
        },
        {
            "ticker": ticker,
            "benchmark": benchmark,
        },
        {
            "ticker": ticker,
            "period": period,
        },
        {
            "ticker": ticker,
        },
    ]

    last_error = None

    for kwargs in attempts:
        if benchmark is None:
            kwargs.pop("benchmark", None)

        try:
            return fn(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    try:
        return fn(ticker, benchmark, period)
    except TypeError:
        if last_error is not None:
            raise last_error
        raise


def run_one(
    module_name: str,
    ticker: str,
    benchmark: Optional[str],
    period: str,
) -> Tuple[Dict[str, Any], str]:
    module = importlib.import_module(module_name)
    fn, fn_name = _extract_runner(module)

    result = _call_runner(
        fn,
        ticker,
        benchmark,
        period,
    )

    if hasattr(result, "to_dict") and callable(result.to_dict):
        result = result.to_dict()

    if not isinstance(result, Mapping):
        raise TypeError(
            f"{module_name}.{fn_name} returned "
            f"{type(result).__name__}, expected a mapping/dict-like result."
        )

    return dict(result), fn_name


def summarize_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    paths = (
        "status",
        "route",
        "engine",
        "benchmark",
        "benchmark_trust",
        "benchmark_trust_decision_grade",
        "signal",
        "action",
        "score",
        "confidence",
        "current_price",
        "regime",
        "cycle",
        "data_quality",
        "asset_age.route",
        "asset_age.trading_days",
        "asset_age.valid_feature_rows",
        "asset_age.forward_complete_rows",
        "asset_age.max_independent_matches",
        "engine_result.signal",
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
        "engine_result.metrics.relative_strength_20d",
        "engine_result.metrics.atr_pct",
        "engine_result.metrics.ema20",
        "engine_result.metrics.ema50",
        "engine_result.metrics.rsi",
        "engine_result.metrics.return_5d",
        "engine_result.metrics.return_20d",
        "engine_result.metrics.return_since_listing",
        "engine_result.metrics.volume_ratio",
        "engine_result.entry.preferred_entry",
        "engine_result.entry.stop_loss",
        "engine_result.entry.tp_1",
        "engine_result.entry.tp_2",
        "engine_result.entry.tp_3",
    )

    return {
        path: _json_safe(_get_path(result, path))
        for path in paths
    }


def _run_self_test() -> int:
    base = {
        "status": "ANALYZED",
        "route": "MATURE",
        "engine": "CCE",
        "benchmark": "SPY",
        "benchmark_trust": "CONTEXT_ONLY",
        "signal": "WATCH",
        "score": 55.0,
        "confidence": 47.0,
        "engine_result": {
            "score": 55.0,
            "confidence": 47.0,
            "signal": "WATCH",
            "metrics": {
                "ema20": 100.0,
                "rsi": 50.0,
            },
        },
    }

    # 1. Architecture fields are allowed.
    new = json.loads(json.dumps(base))
    new["version"] = "1.4.2"
    new["engine_result"]["benchmark_source"] = "BENCHMARK_RESOLVER"

    result = compare_results("NVDA", base, new)
    assert result["pass"]
    assert result["counts"]["architecture_differences"] == 2

    # 2. SPCX relative-strength data-path correction is expected.
    old = {
        "route": "NEW_LISTING",
        "engine": "NEW_LISTING_ENGINE",
        "engine_result": {
            "metrics": {"relative_strength_20d": 12.86}
        },
    }
    new = {
        "route": "NEW_LISTING",
        "engine": "NEW_LISTING_ENGINE",
        "engine_result": {
            "metrics": {"relative_strength_20d": 13.14}
        },
    }

    result = compare_results("SPCX", old, new)
    assert result["pass"]
    assert result["counts"]["expected_data_path_differences"] == 1

    # 3. Small numerical movement is market-data drift.
    old = json.loads(json.dumps(base))
    old["current_price"] = 152.345
    old["engine_result"]["metrics"]["ema20"] = 147.0889

    new = json.loads(json.dumps(old))
    new["current_price"] = 152.245
    new["engine_result"]["metrics"]["ema20"] = 147.0794

    result = compare_results("SPCX", old, new)
    assert result["pass"]
    assert result["counts"]["market_data_drift"] == 2

    # 4. Observable decision change is still a regression.
    new = json.loads(json.dumps(old))
    new["current_price"] = 152.245
    new["signal"] = "BUY"

    result = compare_results("SPCX", old, new)
    assert not result["pass"]
    assert result["counts"]["regressions"] == 1

    # 5. Unknown difference is not silently accepted.
    old = json.loads(json.dumps(base))
    new = json.loads(json.dumps(base))
    new["unknown_field"] = 123

    result = compare_results("NVDA", old, new)
    assert not result["pass"]
    assert result["counts"]["unexpected_differences"] == 1

    # 6. DataFrame equality must not raise.
    if pd is not None:
        old = {
            "route": "MATURE",
            "engine_result": {
                "asset_df": pd.DataFrame(
                    {"Close": [100.0, 101.0]}
                )
            },
        }
        new = {
            "route": "MATURE",
            "engine_result": {
                "asset_df": pd.DataFrame(
                    {"Close": [100.0, 101.0]}
                )
            },
        }

        result = compare_results("NVDA", old, new)
        assert result["pass"]

        # Different internal DataFrames are ignored as non-observable payloads.
        new["engine_result"]["asset_df"] = pd.DataFrame(
            {"Close": [100.0, 102.0]}
        )
        result = compare_results("NVDA", old, new)
        assert result["pass"]
        assert result["counts"]["ignored_object_paths"] == 1

    print("SELF-TEST: PASS")
    return 0


def _print_report(report: Mapping[str, Any]) -> None:
    print("=" * 82)
    print("ORCHESTRATOR V1.3 vs V1.4.2 — COMPARATOR V1.4")
    print("=" * 82)
    print(f"Overall: {report['overall']}")
    print(f"Tickers: {', '.join(report['tickers'])}")
    print()

    print("Totals:")
    for key, value in report["totals"].items():
        print(f"  {key}: {value}")

    print()

    for item in report["results"]:
        counts = item["counts"]

        print(
            f"{item['ticker']:8s} | "
            f"{item['classification']:18s} | "
            f"PASS={str(item['pass']):5s} | "
            f"REG={counts['regressions']:2d} | "
            f"ARCH={counts['architecture_differences']:3d} | "
            f"PATH={counts['expected_data_path_differences']:2d} | "
            f"DRIFT={counts['market_data_drift']:2d} | "
            f"UNEXP={counts['unexpected_differences']:2d} | "
            f"IGN={counts['ignored_object_paths']:2d}"
        )

        if item.get("error"):
            print(f"    ERROR: {item['error']}")

        for diff in item.get("differences", []):
            if diff["classification"] in {
                "REGRESSION",
                "UNEXPECTED_DIFF",
            }:
                print(
                    f"    {diff['classification']:16s} "
                    f"{diff['path']}: "
                    f"{diff['old']} -> {diff['new']}"
                )

    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Asset Analysis Orchestrator V1.3 "
            "vs V1.4.2."
        )
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["NVDA", "MSFT", "GOOGL", "SPCX"],
    )
    parser.add_argument(
        "--benchmark",
        default=None,
    )
    parser.add_argument(
        "--period",
        default="5y",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    results: List[Dict[str, Any]] = []

    for ticker in args.tickers:
        started = datetime.now(timezone.utc).isoformat()

        try:
            old_result, old_runner = run_one(
                V1_3_MODULE,
                ticker,
                args.benchmark,
                args.period,
            )

            new_result, new_runner = run_one(
                V1_4_2_MODULE,
                ticker,
                args.benchmark,
                args.period,
            )

            comparison = compare_results(
                ticker=ticker,
                old_result=old_result,
                new_result=new_result,
            )

            comparison.update(
                {
                    "runner": {
                        "v1_3": old_runner,
                        "v1_4_2": new_runner,
                    },
                    "v1_3_summary": summarize_result(
                        old_result
                    ),
                    "v1_4_2_summary": summarize_result(
                        new_result
                    ),
                    "started_at": started,
                    "completed_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )

        except Exception as exc:
            comparison = {
                "ticker": ticker,
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
                "started_at": started,
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        results.append(comparison)

    totals = {
        "regressions": sum(
            r["counts"]["regressions"] for r in results
        ),
        "architecture_differences": sum(
            r["counts"]["architecture_differences"]
            for r in results
        ),
        "expected_data_path_differences": sum(
            r["counts"]["expected_data_path_differences"]
            for r in results
        ),
        "market_data_drift": sum(
            r["counts"]["market_data_drift"]
            for r in results
        ),
        "unexpected_differences": sum(
            r["counts"]["unexpected_differences"]
            for r in results
        ),
        "ignored_object_paths": sum(
            r["counts"].get("ignored_object_paths", 0)
            for r in results
        ),
        "errors": sum(
            r["counts"]["errors"] for r in results
        ),
    }

    if totals["errors"]:
        overall = "FAIL"
    elif (
        totals["regressions"]
        or totals["unexpected_differences"]
    ):
        overall = "FAIL"
    else:
        overall = "PASS"

    report = {
        "comparator": (
            "test_orchestrator_v1_3_vs_v1_4_2_v1_4"
        ),
        "version": VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "v1_3_module": V1_3_MODULE,
        "v1_4_2_module": V1_4_2_MODULE,
        "tickers": list(args.tickers),
        "benchmark": args.benchmark,
        "period": args.period,
        "overall": overall,
        "pass": overall == "PASS",
        "totals": totals,
        "classification_rules": {
            "REGRESSION": (
                "Observable decision/routing change or "
                "asset-age eligibility change."
            ),
            "ARCHITECTURE_DIFF": (
                "Known V1.4.2 architecture/metadata fields."
            ),
            "EXPECTED_DATA_PATH": (
                "Known SPCX NLE relative-strength "
                "Close-vs-Adj-Close correction."
            ),
            "MARKET_DATA_DRIFT": (
                "Small field-specific numerical differences "
                "in known market-data-derived fields caused "
                "by sequential snapshots."
            ),
            "UNEXPECTED_DIFF": (
                "Any difference not covered by explicit rules."
            ),
            "IGNORED_OBJECT": (
                "Internal DataFrame/Series/ndarray payloads "
                "that are not observable contract fields."
            ),
            "PASS_CONDITION": (
                "REGRESSION=0, UNEXPECTED_DIFF=0, ERROR=0."
            ),
        },
        "market_data_drift_tolerances": {
            path: {
                "absolute": tol[0],
                "relative": tol[1],
            }
            for path, tol in (
                MARKET_DATA_DRIFT_TOLERANCES.items()
            )
        },
        "results": results,
    }

    _print_report(report)

    if args.json_path:
        with open(
            args.json_path,
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                report,
                fh,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"JSON report written to: {args.json_path}"
        )

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
