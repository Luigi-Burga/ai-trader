#!/usr/bin/env python3
"""
AI Trader - Production Validation Suite V1.3

Purpose
-------
Current-architecture production gate for AI Trader.

Unlike V1.2, this suite DOES NOT require the retired
app.ai.asset_analysis_orchestrator_v1_3 module.

Validated production architecture:
    Watchlist Scanner V2.1.2
        -> Orchestrator V1.4.2
        -> Asset Age Resolver V1.4
        -> CCE V1.6.1 / NLE V1.4.1
        -> Benchmark Resolver V1.6.1
        -> Market Data V2.0
        -> Decision Gate V1.2
        -> Signal Engine V1.1

V1.3 validates:
- Required production modules and versions.
- All 19 production tickers.
- Route/engine consistency.
- Engine-version consistency.
- Required benchmark/trust metadata.
- Asset Age routing contract.
- CCE mature-history contract.
- NLE new-listing contract.
- Market Data V2 data-context contract.
- Decision safety invariants.
- No unexpected execution errors.
- Two-run decision stability when --stability is requested.

IMPORTANT
---------
This is a CURRENT-ARCHITECTURE validation suite. It intentionally does
not compare against Orchestrator V1.3, because V1.3 is retired.

Usage
-----
    python -m production_validation_suite_v1_3 --self-test

    python -m production_validation_suite_v1_3 --preflight

    python -m production_validation_suite_v1_3 \
        --tickers NVDA MSFT GOOGL AMZN AVGO TSM META AAPL BRK.B JPM V MA \
        TQQQ SOXL UPRO DFEN GDXU AGQ SPCX \
        --json production_validation_v1_3.json

    python -m production_validation_suite_v1_3 \
        --stability \
        --tickers NVDA MSFT GOOGL AMZN AVGO TSM META AAPL BRK.B JPM V MA \
        TQQQ SOXL UPRO DFEN GDXU AGQ SPCX \
        --json production_stability_v1_3.json

Exit codes
----------
0 = production gate PASS
1 = production gate FAIL
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


VERSION = "1.3.1"

TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AVGO", "TSM",
    "META", "AAPL", "BRK.B", "JPM", "V", "MA",
    "TQQQ", "SOXL", "UPRO", "DFEN", "GDXU", "AGQ", "SPCX",
]

# Current production modules.
ORCHESTRATOR_MODULE = "app.ai.asset_analysis_orchestrator_v1_4_2"
SCANNER_MODULE = "app.scanners.watchlist_scanner_v2_1_2"
MARKET_DATA_MODULE = "app.data.market_data"
BENCHMARK_MODULE = "app.ai.benchmark_resolver_v1_6_1"
ASSET_AGE_MODULE = "app.ai.asset_age_resolver_v1_4"
CCE_MODULE = "app.ai.characteristic_curve_v1_6_1"
NLE_MODULE = "app.ai.new_listing_engine_v1_4_1"
DECISION_GATE_MODULE = "app.strategies.decision_gate_v1_2"
SIGNAL_ENGINE_MODULE = "app.strategies.signal_engine_v1_1"

EXPECTED_VERSIONS = {
    ORCHESTRATOR_MODULE: "1.4.2",
    MARKET_DATA_MODULE: "2.0",
    BENCHMARK_MODULE: "1.6.1",
    ASSET_AGE_MODULE: "1.4",
    CCE_MODULE: "1.6.1",
    NLE_MODULE: "1.4.1",
    DECISION_GATE_MODULE: "1.2",
    SIGNAL_ENGINE_MODULE: "1.1",
}

VALID_ROUTES = {"MATURE", "NEW_LISTING", "NLE_EXTENDED_HISTORY", "REVIEW_DATA", "NO_DATA", "DATA_ERROR"}
VALID_ENGINES = {
    "CHARACTERISTIC_CURVE_ENGINE",
    "NEW_LISTING_ENGINE",
    "REVIEW_DATA",
    "NONE",
}
VALID_STATUS = {"ANALYZED", "REVIEW", "REVIEW_DATA", "NO_DATA", "DATA_ERROR", "ENGINE_ERROR"}

# Operational signals that are explicitly non-bullish.
NON_BULLISH = {
    "WATCH",
    "HOLD",
    "REDUCE",
    "SELL",
    "REVIEW_DATA",
    "NO_DATA",
    "DATA_ERROR",
}

# A benchmark with no decision-grade trust must not silently become a
# confirmed BUY/STRONG_BUY operational decision.
BULLISH_SIGNALS = {"BUY", "STRONG_BUY"}

# Asset Age should route mature assets to CCE and early assets to NLE.
ROUTE_ENGINE = {
    "MATURE": "CHARACTERISTIC_CURVE_ENGINE",
    "NEW_LISTING": "NEW_LISTING_ENGINE",
    "NLE_EXTENDED_HISTORY": "NEW_LISTING_ENGINE",
    "REVIEW_DATA": "REVIEW_DATA",
    "NO_DATA": "NONE",
    "DATA_ERROR": "NONE",
}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _import(name: str):
    return importlib.import_module(name)


def _version(module: Any) -> Optional[str]:
    for key in ("VERSION", "__version__", "VERSION_STRING"):
        value = getattr(module, key, None)
        if value is not None:
            return str(value)
    return None


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
    raise TypeError(f"Unsupported result type: {type(result)!r}")


def _call_analyze(module: Any, ticker: str, period: str, interval: str,
                  benchmark: Optional[str]) -> Dict[str, Any]:
    fn = getattr(module, "analyze_ticker", None)
    if fn is None:
        raise AttributeError(
            f"{module.__name__} does not expose analyze_ticker()"
        )

    result = fn(
        ticker,
        period=period,
        interval=interval,
        benchmark=benchmark,
        listing_price=None,
        validate_benchmark=True,
        resolver_config=None,
    )
    return _normalize_result(result)


def _get_nested(result: Dict[str, Any], *keys: str, default=None):
    current: Any = result
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    return float(value)


def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    engine_result = result.get("engine_result") or {}
    asset_age = result.get("asset_age") or {}
    data_context = result.get("data_context") or {}

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
        "benchmark_trust_decision_grade": result.get(
            "benchmark_trust_decision_grade"
        ),
        "asset_age.route": asset_age.get("route"),
        "asset_age.recommendation": asset_age.get("recommendation"),
        "asset_age.engine": asset_age.get("engine"),
        "asset_age.trading_days": asset_age.get("trading_days"),
        "asset_age.valid_feature_rows": asset_age.get("valid_feature_rows"),
        "asset_age.cce_eligible": asset_age.get("cce_eligible"),
        "asset_age.nle_eligible": asset_age.get("nle_eligible"),
        "engine_result.signal": engine_result.get("signal"),
        "engine_result.action": engine_result.get("action"),
        "engine_result.decision": engine_result.get("decision"),
        "engine_result.score": engine_result.get("score"),
        "engine_result.confidence": engine_result.get("confidence"),
        "engine_result.cycle": engine_result.get("cycle"),
        "engine_result.operational_signal": engine_result.get(
            "operational_signal"
        ),
        "data_context.market_data_version": data_context.get(
            "market_data_version"
        ),
        "data_context.asset_raw_loaded": data_context.get(
            "asset_raw_loaded"
        ),
        "data_context.asset_adjusted_loaded": data_context.get(
            "asset_adjusted_loaded"
        ),
        "data_context.shared_asset_context": data_context.get(
            "shared_asset_context"
        ),
        "data_context.adjustment_semantics_preserved": data_context.get(
            "adjustment_semantics_preserved"
        ),
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    for module_name, expected in EXPECTED_VERSIONS.items():
        try:
            module = _import(module_name)
            actual = _version(module)
            checks[module_name] = {
                "ok": actual == expected,
                "version": actual,
                "expected_version": expected,
            }
            if actual != expected:
                checks[module_name]["error"] = (
                    f"Expected {expected}, found {actual}"
                )
        except Exception as exc:
            checks[module_name] = {
                "ok": False,
                "expected_version": expected,
                "error": str(exc),
            }

    # Scanner is validated structurally/version-wise, but it may not expose
    # VERSION through the same pattern in every environment.
    try:
        scanner = _import(SCANNER_MODULE)
        scanner_version = _version(scanner)
        checks[SCANNER_MODULE] = {
            "ok": scanner_version == "2.1.2",
            "version": scanner_version,
            "expected_version": "2.1.2",
        }
    except Exception as exc:
        checks[SCANNER_MODULE] = {
            "ok": False,
            "expected_version": "2.1.2",
            "error": str(exc),
        }

    return checks


def _preflight_ok(checks: Dict[str, Any]) -> bool:
    return all(item.get("ok") for item in checks.values())


# ---------------------------------------------------------------------------
# Functional validation
# ---------------------------------------------------------------------------

def validate_result(ticker: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate one current-production Orchestrator V1.4.2 result.

    The suite checks architecture and safety contracts rather than asserting
    a particular investment signal. Signal values are market-dependent.
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    status = result.get("status")
    route = result.get("route")
    engine = result.get("engine")
    engine_version = result.get("engine_version")
    signal = result.get("signal")
    action = result.get("action")

    asset_age = result.get("asset_age") or {}
    engine_result = result.get("engine_result") or {}
    data_context = result.get("data_context") or {}

    # 1. Basic result contract.
    if status not in VALID_STATUS:
        errors.append({
            "field": "status",
            "reason": f"Invalid status: {status!r}",
        })

    if route not in VALID_ROUTES:
        errors.append({
            "field": "route",
            "reason": f"Invalid route: {route!r}",
        })

    if engine not in VALID_ENGINES:
        errors.append({
            "field": "engine",
            "reason": f"Invalid engine: {engine!r}",
        })

    # 2. Explicit REVIEW_DATA / NO_DATA / NONE contract.
    # This is a valid terminal safety state emitted by Orchestrator V1.4.2
    # when Asset Age cannot obtain usable history for an asset. It must not
    # be treated as a regression and must not enter BUY gates.
    review_no_data = (
        status == "REVIEW"
        and route == "NO_DATA"
        and engine == "NONE"
    )
    if review_no_data:
        if signal != "REVIEW_DATA":
            errors.append({
                "field": "review_data_contract.signal",
                "reason": (
                    "REVIEW/NO_DATA/NONE must emit signal REVIEW_DATA; "
                    f"found {signal!r}"
                ),
            })
        if action != "REVIEW_DATA":
            errors.append({
                "field": "review_data_contract.action",
                "reason": (
                    "REVIEW/NO_DATA/NONE must emit action REVIEW_DATA; "
                    f"found {action!r}"
                ),
            })
        if result.get("data_quality") != "NO_DATA":
            errors.append({
                "field": "review_data_contract.data_quality",
                "reason": (
                    "REVIEW/NO_DATA/NONE must have data_quality NO_DATA; "
                    f"found {result.get('data_quality')!r}"
                ),
            })
        if (result.get("engine_version") is not None):
            errors.append({
                "field": "review_data_contract.engine_version",
                "reason": "NONE engine must not expose an engine version.",
            })
        if asset_age.get("route") != "NO_DATA":
            errors.append({
                "field": "review_data_contract.asset_age.route",
                "reason": "Asset Age route must be NO_DATA.",
            })
        if asset_age.get("recommendation") != "REVIEW_DATA":
            errors.append({
                "field": "review_data_contract.asset_age.recommendation",
                "reason": "Asset Age recommendation must be REVIEW_DATA.",
            })
        if asset_age.get("cce_eligible") is not False:
            errors.append({
                "field": "review_data_contract.asset_age.cce_eligible",
                "reason": "NO_DATA route cannot be CCE eligible.",
            })
        if asset_age.get("nle_eligible") is not False:
            errors.append({
                "field": "review_data_contract.asset_age.nle_eligible",
                "reason": "NO_DATA route cannot be NLE eligible.",
            })
        if engine_result.get("signal") != "REVIEW_DATA":
            errors.append({
                "field": "review_data_contract.engine_result.signal",
                "reason": "Engine result signal must be REVIEW_DATA.",
            })
        if engine_result.get("decision") != "REVIEW_DATA":
            errors.append({
                "field": "review_data_contract.engine_result.decision",
                "reason": "Engine result decision must be REVIEW_DATA.",
            })
        if _as_float(result.get("score")) != 0.0:
            errors.append({
                "field": "review_data_contract.score",
                "reason": "REVIEW/NO_DATA/NONE score must be 0.0.",
            })
        if _as_float(result.get("confidence")) != 0.0:
            errors.append({
                "field": "review_data_contract.confidence",
                "reason": "REVIEW/NO_DATA/NONE confidence must be 0.0.",
            })

    # 3. Route / engine consistency.
    if route in ROUTE_ENGINE and engine != ROUTE_ENGINE[route]:
        errors.append({
            "field": "route_engine_consistency",
            "reason": f"{route} must use {ROUTE_ENGINE[route]}, found {engine}",
        })

    # 3. Engine version contract.
    if engine == "CHARACTERISTIC_CURVE_ENGINE":
        if engine_version != "1.6.1":
            errors.append({
                "field": "engine_version",
                "reason": (
                    "Mature assets must use CCE V1.6.1; "
                    f"found {engine_version!r}"
                ),
            })

    elif engine == "NEW_LISTING_ENGINE":
        if engine_version != "1.4.1":
            errors.append({
                "field": "engine_version",
                "reason": (
                    "New listings must use NLE V1.4.1; "
                    f"found {engine_version!r}"
                ),
            })

    # 4. Asset Age consistency.
    asset_route = asset_age.get("route")
    asset_recommendation = asset_age.get("recommendation")
    asset_engine = asset_age.get("engine")

    if asset_route and asset_route != route:
        errors.append({
            "field": "asset_age.route",
            "reason": f"Asset Age route {asset_route!r} != orchestrator route {route!r}",
        })

    if route == "MATURE":
        if asset_recommendation != "USE_CCE":
            errors.append({
                "field": "asset_age.recommendation",
                "reason": (
                    "MATURE route must recommend USE_CCE; "
                    f"found {asset_recommendation!r}"
                ),
            })
        if asset_engine != "CHARACTERISTIC_CURVE_ENGINE":
            errors.append({
                "field": "asset_age.engine",
                "reason": "MATURE route must use CCE.",
            })

    if route in {"NEW_LISTING", "NLE_EXTENDED_HISTORY"}:
        if asset_recommendation != "USE_NLE":
            errors.append({
                "field": "asset_age.recommendation",
                "reason": (
                    "New-listing route must recommend USE_NLE; "
                    f"found {asset_recommendation!r}"
                ),
            })
        if asset_engine != "NEW_LISTING_ENGINE":
            errors.append({
                "field": "asset_age.engine",
                "reason": "New-listing route must use NLE.",
            })

    # 5. Benchmark contract for analyzed assets.
    if status == "ANALYZED":
        if not result.get("benchmark"):
            errors.append({
                "field": "benchmark",
                "reason": "ANALYZED result has no selected benchmark.",
            })

        if result.get("benchmark_trust") not in {
            "HIGH", "MEDIUM", "CONTEXT_ONLY", "NONE"
        }:
            errors.append({
                "field": "benchmark_trust",
                "reason": (
                    "Invalid benchmark trust: "
                    f"{result.get('benchmark_trust')!r}"
                ),
            })

        if result.get("benchmark_trust") in {"HIGH", "MEDIUM"}:
            if result.get("benchmark_trust_decision_grade") is not True:
                errors.append({
                    "field": "benchmark_trust_decision_grade",
                    "reason": (
                        "HIGH/MEDIUM benchmark trust must be "
                        "decision-grade."
                    ),
                })

    # 6. Market Data V2 context contract.
    if data_context:
        md_version = data_context.get("market_data_version")
        if md_version != "2.0":
            errors.append({
                "field": "data_context.market_data_version",
                "reason": f"Expected Market Data 2.0, found {md_version!r}",
            })

        if status == "ANALYZED":
            for field in (
                "asset_raw_loaded",
                "asset_adjusted_loaded",
                "shared_asset_context",
                "adjustment_semantics_preserved",
            ):
                if data_context.get(field) is not True:
                    errors.append({
                        "field": f"data_context.{field}",
                        "reason": "Production data-context invariant is false.",
                    })

    # 7. CCE contract.
    if engine == "CHARACTERISTIC_CURVE_ENGINE":
        if not _is_number(engine_result.get("matches")):
            errors.append({
                "field": "engine_result.matches",
                "reason": "CCE result must expose numeric historical matches.",
            })

        if _as_float(engine_result.get("matches")) is not None:
            if _as_float(engine_result.get("matches")) < 0:
                errors.append({
                    "field": "engine_result.matches",
                    "reason": "CCE matches cannot be negative.",
                })

    # 8. NLE contract.
    if engine == "NEW_LISTING_ENGINE":
        classification = engine_result.get("classification")
        if not classification:
            errors.append({
                "field": "engine_result.classification",
                "reason": "NLE must expose its listing classification.",
            })

        if ticker.upper() == "SPCX" and route == "NEW_LISTING":
            # SPCX is explicitly expected to have early-history NLE routing.
            if classification != "NEW_LISTING_EARLY":
                errors.append({
                    "field": "engine_result.classification",
                    "reason": (
                        "SPCX production fixture is expected to remain "
                        "NEW_LISTING_EARLY."
                    ),
                })

    # 9. Decision safety invariant.
    trust = result.get("benchmark_trust")
    operational_signal = (
        engine_result.get("operational_signal")
        or result.get("signal")
    )

    if (
        operational_signal in BULLISH_SIGNALS
        and trust not in {"HIGH", "MEDIUM"}
    ):
        errors.append({
            "field": "decision_safety",
            "reason": (
                "Confirmed BUY/STRONG_BUY is not allowed when benchmark "
                f"trust is {trust!r}."
            ),
        })

    # 10. BUY_ON_CONFIRMATION safety.
    if signal == "BUY_ON_CONFIRMATION":
        if action not in {"BUY_ON_CONFIRMATION", "BUY", "STRONG_BUY"}:
            errors.append({
                "field": "signal_action_consistency",
                "reason": (
                    f"BUY_ON_CONFIRMATION signal has incompatible action "
                    f"{action!r}."
                ),
            })

    # 11. Numeric sanity where values are expected.
    if status == "ANALYZED":
        score = result.get("score")
        confidence = result.get("confidence")

        if _is_number(score) and not 0.0 <= float(score) <= 100.0:
            errors.append({
                "field": "score",
                "reason": f"Score outside 0-100: {score!r}",
            })

        if _is_number(confidence) and not 0.0 <= float(confidence) <= 100.0:
            errors.append({
                "field": "confidence",
                "reason": f"Confidence outside 0-100: {confidence!r}",
            })

    passed = len(errors) == 0

    return {
        "ticker": ticker,
        "pass": passed,
        "classification": "PASS" if passed else "REGRESSION",
        "errors": errors,
        "warnings": warnings,
        "summary": _result_summary(result),
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

STABILITY_FIELDS = (
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


def compare_stability(
    ticker: str,
    first: Dict[str, Any],
    second: Dict[str, Any],
) -> Dict[str, Any]:
    a = _result_summary(first)
    b = _result_summary(second)

    differences = []

    for field in STABILITY_FIELDS:
        old = a.get(field)
        new = b.get(field)
        if old != new:
            differences.append({
                "field": field,
                "old": old,
                "new": new,
                "reason": (
                    "Production routing/decision field changed "
                    "between consecutive runs."
                ),
            })

    return {
        "ticker": ticker,
        "pass": len(differences) == 0,
        "classification": (
            "STABLE" if not differences else "UNSTABLE"
        ),
        "decision_regressions": len(differences),
        "differences": differences,
        "first_summary": a,
        "second_summary": b,
    }


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

def run_suite(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    ticker_list = list(tickers)
    checks = preflight()

    report: Dict[str, Any] = {
        "suite": "Production Validation Suite",
        "version": VERSION,
        "generated_at": started.isoformat(),
        "architecture": "CURRENT_V1_4_2",
        "orchestrator": ORCHESTRATOR_MODULE,
        "scanner": SCANNER_MODULE,
        "market_data_layer": "2.0",
        "benchmark_resolver": "1.6.1",
        "asset_age_resolver": "1.4",
        "cce": "1.6.1",
        "nle": "1.4.1",
        "decision_gate": "1.2",
        "signal_engine": "1.1",
        "tickers": ticker_list,
        "benchmark": benchmark,
        "period": period,
        "interval": interval,
        "preflight": checks,
        "results": [],
    }

    # Preflight failure is an execution/configuration error, not an
    # investment-model regression.
    if not _preflight_ok(checks):
        report["overall"] = "FAIL"
        report["pass"] = False
        report["production_gate"] = {
            "REGRESSION": 0,
            "UNEXPECTED_DIFF": 0,
            "ERROR": 1,
            "PASS_CONDITION": (
                "REGRESSION=0 AND UNEXPECTED_DIFF=0 AND ERROR=0"
            ),
        }
        report["totals"] = {
            "passed": 0,
            "failed": 0,
            "regressions": 0,
            "unexpected_differences": 0,
            "errors": 1,
        }
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        return report

    module = _import(ORCHESTRATOR_MODULE)

    for ticker in ticker_list:
        started_ticker = datetime.now(timezone.utc)

        try:
            result = _call_analyze(
                module,
                ticker,
                period,
                interval,
                benchmark,
            )
            validation = validate_result(ticker, result)

            item = {
                "ticker": ticker,
                "classification": validation["classification"],
                "pass": validation["pass"],
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "summary": validation["summary"],
                "started_at": started_ticker.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            item = {
                "ticker": ticker,
                "classification": "ERROR",
                "pass": False,
                "errors": [{
                    "field": "execution",
                    "reason": str(exc),
                }],
                "warnings": [],
                "summary": {},
                "traceback": traceback.format_exc(),
                "started_at": started_ticker.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        report["results"].append(item)

    regressions = sum(
        (not item["pass"]) and item["classification"] == "REGRESSION"
        for item in report["results"]
    )
    errors = sum(
        item["classification"] == "ERROR"
        for item in report["results"]
    )
    passed = sum(bool(item["pass"]) for item in report["results"])

    report["totals"] = {
        "passed": passed,
        "failed": len(report["results"]) - passed,
        "regressions": regressions,
        "unexpected_differences": 0,
        "errors": errors,
    }

    report["production_gate"] = {
        "REGRESSION": regressions,
        "UNEXPECTED_DIFF": 0,
        "ERROR": errors,
        "PASS_CONDITION": (
            "REGRESSION=0 AND UNEXPECTED_DIFF=0 AND ERROR=0"
        ),
    }

    report["overall"] = (
        "PASS"
        if regressions == 0 and errors == 0
        else "FAIL"
    )
    report["pass"] = report["overall"] == "PASS"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

    return report


def run_stability_suite(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    ticker_list = list(tickers)
    checks = preflight()

    report: Dict[str, Any] = {
        "suite": "Production Validation Suite - Stability",
        "version": VERSION,
        "generated_at": started.isoformat(),
        "module": ORCHESTRATOR_MODULE,
        "period": period,
        "interval": interval,
        "benchmark": benchmark,
        "runs_per_ticker": 2,
        "tickers": ticker_list,
        "preflight": checks,
        "results": [],
    }

    if not _preflight_ok(checks):
        report["totals"] = {
            "stable": 0,
            "unstable": 0,
            "errors": 1,
        }
        report["production_gate"] = {
            "DECISION_INSTABILITY": 0,
            "ERROR": 1,
            "PASS_CONDITION": (
                "DECISION_INSTABILITY=0 AND ERROR=0"
            ),
        }
        report["overall"] = "FAIL"
        report["pass"] = False
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        return report

    module = _import(ORCHESTRATOR_MODULE)

    for ticker in ticker_list:
        try:
            first = _call_analyze(
                module, ticker, period, interval, benchmark
            )
            second = _call_analyze(
                module, ticker, period, interval, benchmark
            )

            item = compare_stability(ticker, first, second)

        except Exception as exc:
            item = {
                "ticker": ticker,
                "pass": False,
                "classification": "ERROR",
                "decision_regressions": 0,
                "differences": [],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        report["results"].append(item)

    unstable = sum(
        item["classification"] == "UNSTABLE"
        for item in report["results"]
    )
    errors = sum(
        item["classification"] == "ERROR"
        for item in report["results"]
    )
    stable = sum(bool(item["pass"]) for item in report["results"])

    report["totals"] = {
        "stable": stable,
        "unstable": unstable,
        "errors": errors,
    }
    report["production_gate"] = {
        "DECISION_INSTABILITY": unstable,
        "ERROR": errors,
        "PASS_CONDITION": (
            "DECISION_INSTABILITY=0 AND ERROR=0"
        ),
    }
    report["overall"] = (
        "PASS" if unstable == 0 and errors == 0 else "FAIL"
    )
    report["pass"] = report["overall"] == "PASS"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> None:
    """
    Self-test the validation rules without requiring the AI Trader runtime.
    """
    review_data = {
        "status": "REVIEW",
        "route": "NO_DATA",
        "engine": "NONE",
        "engine_version": None,
        "signal": "REVIEW_DATA",
        "action": "REVIEW_DATA",
        "score": 0.0,
        "confidence": 0.0,
        "current_price": None,
        "regime": None,
        "data_quality": "NO_DATA",
        "benchmark": None,
        "benchmark_trust": None,
        "benchmark_trust_decision_grade": None,
        "asset_age": {
            "route": "NO_DATA",
            "recommendation": "REVIEW_DATA",
            "engine": "UNKNOWN",
            "trading_days": 0,
            "valid_feature_rows": 0,
            "cce_eligible": False,
            "nle_eligible": False,
        },
        "engine_result": {
            "signal": "REVIEW_DATA",
            "decision": "REVIEW_DATA",
            "score": 0.0,
            "confidence": 0.0,
        },
        "data_context": {
            "market_data_version": "2.0",
            "asset_raw_loaded": False,
            "asset_adjusted_loaded": False,
            "shared_asset_context": True,
            "adjustment_semantics_preserved": True,
        },
    }
    review_check = validate_result("BRK.B", review_data)
    assert review_check["pass"], (
        "REVIEW/NO_DATA/NONE contract failed: "
        + json.dumps(review_check, indent=2, default=str)
    )

    mature = {
        "status": "ANALYZED",
        "route": "MATURE",
        "engine": "CHARACTERISTIC_CURVE_ENGINE",
        "engine_version": "1.6.1",
        "signal": "WATCH",
        "action": "WATCH",
        "score": 61.0,
        "confidence": 55.0,
        "current_price": 100.0,
        "regime": "CONFIRMED_BULL",
        "data_quality": "EXCELLENT",
        "benchmark": "SPY",
        "benchmark_trust": "HIGH",
        "benchmark_trust_decision_grade": True,
        "asset_age": {
            "route": "MATURE",
            "recommendation": "USE_CCE",
            "engine": "CHARACTERISTIC_CURVE_ENGINE",
            "trading_days": 1254,
            "valid_feature_rows": 1003,
            "cce_eligible": True,
            "nle_eligible": False,
        },
        "engine_result": {
            "matches": 21,
            "signal": "WATCH",
            "operational_signal": "WATCH",
        },
        "data_context": {
            "market_data_version": "2.0",
            "asset_raw_loaded": True,
            "asset_adjusted_loaded": True,
            "shared_asset_context": True,
            "adjustment_semantics_preserved": True,
        },
    }

    result = validate_result("NVDA", mature)
    assert result["pass"] is True, result

    # Wrong route/engine must fail.
    bad = json.loads(json.dumps(mature))
    bad["route"] = "MATURE"
    bad["engine"] = "NEW_LISTING_ENGINE"
    result = validate_result("NVDA", bad)
    assert result["pass"] is False
    assert any(
        e["field"] == "route_engine_consistency"
        for e in result["errors"]
    )

    # Confirmed bullish decision without decision-grade benchmark must fail.
    unsafe = json.loads(json.dumps(mature))
    unsafe["signal"] = "BUY"
    unsafe["action"] = "BUY"
    unsafe["benchmark_trust"] = "CONTEXT_ONLY"
    unsafe["benchmark_trust_decision_grade"] = False
    unsafe["engine_result"]["operational_signal"] = "BUY"
    result = validate_result("NVDA", unsafe)
    assert result["pass"] is False
    assert any(
        e["field"] == "decision_safety"
        for e in result["errors"]
    )

    # Stability ignores market values but not decisions.
    second = json.loads(json.dumps(mature))
    second["current_price"] = 101.0
    second["score"] = 62.0
    stable = compare_stability("NVDA", mature, second)
    assert stable["pass"] is True

    second["signal"] = "BUY"
    unstable = compare_stability("NVDA", mature, second)
    assert unstable["pass"] is False
    assert unstable["decision_regressions"] >= 1

    print("Production Validation Suite V1.3 self-test: PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 88)
    print("AI TRADER - PRODUCTION VALIDATION SUITE V1.3")
    print("=" * 88)
    print(f"Architecture : {report.get('architecture', 'CURRENT_V1_4_2')}")
    print(f"Overall      : {report['overall']}")
    print(f"Pass         : {report['pass']}")
    print()

    gate = report.get("production_gate", {})
    print("Production Gate")
    for key in ("REGRESSION", "UNEXPECTED_DIFF", "ERROR"):
        if key in gate:
            print(f"  {key:16s}: {gate[key]}")

    print()
    print("Per ticker")
    print("-" * 88)

    for item in report.get("results", []):
        print(
            f"{item['ticker']:7s} "
            f"{item['classification']:12s} "
            f"PASS={str(item['pass']):5s} "
            f"ERRORS={len(item.get('errors', [])):2d}"
        )

        for error in item.get("errors", []):
            print(
                f"    ERROR {error.get('field')}: "
                f"{error.get('reason')}"
            )

    print("-" * 88)

    totals = report.get("totals", {})
    print(f"Totals: {json.dumps(totals, ensure_ascii=False)}")


def _print_stability(report: Dict[str, Any]) -> None:
    print("=" * 88)
    print("AI TRADER - PRODUCTION VALIDATION SUITE V1.3 STABILITY")
    print("=" * 88)
    print(f"Overall : {report['overall']}")
    print(f"Pass    : {report['pass']}")
    print()

    gate = report["production_gate"]
    print(
        f"DECISION_INSTABILITY : "
        f"{gate['DECISION_INSTABILITY']}"
    )
    print(f"ERROR                : {gate['ERROR']}")
    print()

    for item in report["results"]:
        print(
            f"{item['ticker']:7s} "
            f"{item['classification']:10s} "
            f"PASS={str(item['pass']):5s} "
            f"DECISION_DIFF={item.get('decision_regressions', 0):2d}"
        )

    print("-" * 88)
    print(f"Totals: {json.dumps(report['totals'], ensure_ascii=False)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Trader Production Validation Suite V1.3"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
        help="Production ticker matrix.",
    )
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--json",
        default="production_validation_v1_3.json",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Run current V1.4.2 twice per ticker.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate installed production modules only.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local rule self-tests without the production runtime.",
    )

    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    if args.preflight:
        result = preflight()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if _preflight_ok(result) else 1

    if args.stability:
        report = run_stability_suite(
            tickers=args.tickers,
            period=args.period,
            interval=args.interval,
            benchmark=args.benchmark,
        )
        output = args.json
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(
                report,
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        _print_stability(report)
        print(f"JSON: {output}")
        return 0 if report["pass"] else 1

    report = run_suite(
        tickers=args.tickers,
        period=args.period,
        interval=args.interval,
        benchmark=args.benchmark,
    )
    output = args.json

    with open(output, "w", encoding="utf-8") as fh:
        json.dump(
            report,
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    _print_report(report)
    print(f"JSON: {output}")

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
