"""
AI Trader - Decision Gate V1.5
==============================

Multi-engine decision gate for the production pipeline.

Pipeline:
    Asset Age -> Orchestrator -> CCE/NLE -> Decision Gate -> Signal Engine

V1.5 changes from V1.4
----------------------
1. Explicit engine-aware policy:
   - CCE / mature assets use historical evidence gates.
   - NLE / new listings use confirmation + risk + benchmark-context gates.
2. NLE is never forced to provide CCE-only metrics such as:
   historical_probability_20d, reward_risk_20d, fallback_ratio or match_quality.
3. BUY / STRONG_BUY from the engine may become executable BUY only when the
   engine-specific policy is satisfied.
4. BUY_ON_CONFIRMATION, WATCH, SELL, AVOID and REVIEW_DATA are preserved.
5. Backward-compatible public API:
       apply_decision_gate(result, config=...)
6. V1.4 CCE policy is preserved for mature CCE assets.

The gate does not manufacture statistical evidence. Missing metrics are
handled according to the engine contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


VERSION = "1.5"

DECISION_GRADE_TRUST = frozenset({"HIGH", "MEDIUM", "LOW"})
CONTEXT_ONLY_TRUST = frozenset({"CONTEXT_ONLY", "NONE", ""})

BLOCKED_REGIMES = frozenset({
    "CAPITULATION",
    "DOWNTREND",
    "DISTRIBUTION",
    "EXTREME_MOMENTUM",
})

MATCH_QUALITY_RANK = {
    "NONE": 0,
    "WEAK": 1,
    "MODERATE": 2,
    "GOOD": 3,
    "STRONG": 4,
}


@dataclass(frozen=True)
class GateConfig:
    # Engine detection.
    cce_engine_names: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "CHARACTERISTIC_CURVE_ENGINE",
            "CCE",
            "CHARACTERISTIC_CURVE",
        })
    )
    nle_engine_names: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "NEW_LISTING_ENGINE",
            "NLE",
            "NEW_LISTING",
        })
    )

    # Shared quality / benchmark policy.
    min_trading_confidence: float = 60.0
    min_match_quality: str = "GOOD"
    min_nle_confidence: float = 60.0
    decision_grade_trust: frozenset[str] = field(
        default_factory=lambda: DECISION_GRADE_TRUST
    )
    blocked_regimes: frozenset[str] = field(
        default_factory=lambda: BLOCKED_REGIMES
    )

    # CCE hard opportunity/risk gates.
    max_fallback_ratio: float = 0.50
    min_historical_probability: float = 0.60
    min_reward_risk: float = 1.25
    min_appt_x: float = 0.0

    # CCE medium-quality path retained from V1.4.
    min_medium_confidence: float = 50.0
    min_medium_match_quality: str = "MODERATE"
    strong_opportunity_appt_x: float = 0.50
    strong_opportunity_probability: float = 0.65
    strong_opportunity_reward_risk: float = 1.50

    # NLE confirmation policy.
    nle_require_confirmation: bool = True
    nle_min_confirmation_score: float = 4.0
    nle_min_confirmation_ratio: float = 0.6666666667
    nle_allow_context_only_buy: bool = False

    # Shared dynamic risk geometry.
    require_current_inside_entry_zone: bool = True


DEFAULT_CONFIG = GateConfig()


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _get(result: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _detect_engine(result: Mapping[str, Any]) -> str:
    explicit = _norm(_get(result, "engine_type", "decision_engine", default=""))
    if explicit:
        if explicit in {"CCE", "CHARACTERISTIC_CURVE_ENGINE", "CHARACTERISTIC_CURVE"}:
            return "CCE"
        if explicit in {"NLE", "NEW_LISTING_ENGINE", "NEW_LISTING"}:
            return "NLE"

    engine = _norm(result.get("engine"))
    version = _norm(result.get("engine_version"))
    route = _norm(result.get("route"))

    if engine in DEFAULT_CONFIG.cce_engine_names or "CHARACTERISTIC_CURVE" in engine:
        return "CCE"
    if engine in DEFAULT_CONFIG.nle_engine_names or "NEW_LISTING_ENGINE" in engine:
        return "NLE"
    if "NEW_LISTING" in route:
        return "NLE"
    if "MATURE" in route:
        return "CCE"

    # NLE result payloads sometimes identify themselves only inside
    # engine_result / classification.
    nested = _mapping(result.get("engine_result"))
    nested_engine = _norm(nested.get("engine"))
    if "NEW_LISTING" in nested_engine:
        return "NLE"
    if "CHARACTERISTIC_CURVE" in nested_engine:
        return "CCE"

    classification = _norm(
        _get(result, "classification", default=nested.get("classification"))
    )
    if classification.startswith("NEW_LISTING"):
        return "NLE"

    return "UNKNOWN"


def _extract_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    historical = _mapping(result.get("historical_pattern"))
    decision = _mapping(result.get("decision_context"))
    quality = _mapping(result.get("decision_quality"))
    levels = _mapping(result.get("levels"))
    confirmation = _mapping(result.get("confirmation"))

    if not levels:
        levels = _mapping(result.get("dynamic_levels"))
    if not levels:
        levels = _mapping(result.get("entry"))

    metrics: Dict[str, Any] = {
        "trading_confidence": _float(
            _get(
                result,
                "trading_confidence",
                "confidence",
                default=decision.get("confidence", quality.get("confidence")),
            )
        ),
        "match_quality": _norm(
            _get(
                result,
                "match_quality",
                default=decision.get(
                    "match_quality", quality.get("match_quality")
                ),
            )
        ),
        "fallback_ratio": _float(
            _get(
                result,
                "fallback_ratio",
                default=decision.get("fallback_ratio"),
            )
        ),
        "historical_probability": _float(
            _get(
                result,
                "historical_probability_20d",
                default=historical.get("probability_positive_20d"),
            )
        ),
        "reward_risk": _float(
            _get(
                result,
                "reward_risk_20d",
                "historical_excursion_ratio_20d",
                default=historical.get("reward_risk_20d"),
            )
        ),
        "appt_x": _float(
            _get(result, "appt_x_20d", default=historical.get("appt_x_20d"))
        ),
        "regime": _norm(
            _get(
                result,
                "cycle",
                "regime",
                default=decision.get("cycle"),
            )
        ),
        "benchmark_trust": _norm(
            _get(
                result,
                "benchmark_trust",
                default=decision.get("benchmark_trust"),
            )
        ),
        "benchmark_grade": _get(
            result,
            "benchmark_trust_decision_grade",
            default=decision.get("benchmark_trust_decision_grade"),
        ),
        "current_price": _float(
            _get(result, "current_price", default=levels.get("current_price"))
        ),
        "entry_price": _float(
            _get(
                result,
                "entry_price",
                "preferred_entry",
                default=levels.get("entry_price", levels.get("preferred_entry")),
            )
        ),
        "entry_zone_low": _float(
            _get(
                result,
                "entry_zone_low",
                "zone_low",
                default=levels.get("entry_zone_low", levels.get("zone_low")),
            )
        ),
        "entry_zone_high": _float(
            _get(
                result,
                "entry_zone_high",
                "zone_high",
                default=levels.get("entry_zone_high", levels.get("zone_high")),
            )
        ),
        "stop": _float(
            _get(
                result,
                "stop",
                "dynamic_stop",
                "stop_loss",
                default=levels.get(
                    "stop",
                    levels.get("dynamic_stop", levels.get("stop_loss")),
                ),
            )
        ),
        "confirmation_required": _get(
            result,
            "confirmation_required",
            default=confirmation.get(
                "required", decision.get("confirmation_required")
            ),
        ),
        "confirmed": _get(
            result,
            "confirmed",
            default=confirmation.get("confirmed", decision.get("confirmed")),
        ),
        "confirmation_score": _float(
            _get(
                result,
                "confirmation_score",
                default=confirmation.get("score"),
            )
        ),
        "confirmation_max_score": _float(
            _get(
                result,
                "confirmation_max_score",
                default=confirmation.get("max_score"),
            )
        ),
        "listing_days": _float(
            _get(
                result,
                "trading_days",
                "listing_age_days",
                default=decision.get("trading_days"),
            )
        ),
    }

    return metrics


def _benchmark_gate(m: Mapping[str, Any], config: GateConfig) -> bool:
    trust = m["benchmark_trust"]
    grade = m["benchmark_grade"]

    if trust in config.decision_grade_trust:
        if grade is None:
            return True
        return bool(grade)

    # NLE context-only benchmark can be used for context, but not to
    # authorize a standalone BUY unless explicitly configured.
    if trust in CONTEXT_ONLY_TRUST:
        return config.nle_allow_context_only_buy

    return False


def _dynamic_entry_stop_gate(
    m: Mapping[str, Any],
    config: GateConfig,
) -> bool:
    current = m["current_price"]
    entry = m["entry_price"]
    low = m["entry_zone_low"]
    high = m["entry_zone_high"]
    stop = m["stop"]

    values = (current, entry, low, high, stop)
    if any(v is None for v in values):
        return False
    if any(v <= 0 for v in values):
        return False
    if low > high:
        return False
    if not (low <= entry <= high):
        return False
    if stop >= entry:
        return False
    if stop > high:
        return False

    if config.require_current_inside_entry_zone:
        if current < stop or current > high:
            return False
    else:
        if current < stop:
            return False

    return True


def _confirmation_gate(m: Mapping[str, Any], config: GateConfig) -> bool:
    if not config.nle_require_confirmation:
        return True

    confirmed = m["confirmed"]
    required = m["confirmation_required"]

    if confirmed is True:
        return True

    if required is False:
        return True

    score = m["confirmation_score"]
    max_score = m["confirmation_max_score"]
    if score is None:
        return False

    if max_score is not None and max_score > 0:
        ratio = score / max_score
        return (
            score >= config.nle_min_confirmation_score
            and ratio >= config.nle_min_confirmation_ratio
        )

    return score >= config.nle_min_confirmation_score


def _quality_evaluation(
    m: Mapping[str, Any],
    config: GateConfig,
) -> Dict[str, Any]:
    confidence = m["trading_confidence"]
    match_quality = m["match_quality"]

    confidence_full = (
        confidence is not None
        and confidence >= config.min_trading_confidence
    )
    min_full_rank = MATCH_QUALITY_RANK.get(_norm(config.min_match_quality), 3)
    match_full = (
        match_quality in MATCH_QUALITY_RANK
        and MATCH_QUALITY_RANK[match_quality] >= min_full_rank
    )

    confidence_medium = (
        confidence is not None
        and confidence >= config.min_medium_confidence
    )
    min_medium_rank = MATCH_QUALITY_RANK.get(
        _norm(config.min_medium_match_quality), 2
    )
    match_medium = (
        match_quality in MATCH_QUALITY_RANK
        and MATCH_QUALITY_RANK[match_quality] >= min_medium_rank
    )

    full_quality = confidence_full and match_full
    medium_quality = confidence_medium and match_medium

    appt = m["appt_x"]
    probability = m["historical_probability"]
    reward_risk = m["reward_risk"]

    strong_opportunity = (
        appt is not None
        and appt >= config.strong_opportunity_appt_x
        and probability is not None
        and probability >= config.strong_opportunity_probability
        and reward_risk is not None
        and reward_risk >= config.strong_opportunity_reward_risk
    )

    return {
        "confidence_full": confidence_full,
        "match_full": match_full,
        "full_quality": full_quality,
        "confidence_medium": confidence_medium,
        "match_medium": match_medium,
        "medium_quality": medium_quality,
        "strong_opportunity": strong_opportunity,
        "quality_mode": (
            "FULL" if full_quality else "MEDIUM" if medium_quality else "LOW"
        ),
    }


def _evaluate_cce(
    result: Mapping[str, Any],
    m: Mapping[str, Any],
    config: GateConfig,
) -> Dict[str, Any]:
    trust_ok = _benchmark_gate(m, config)
    fallback_ok = (
        m["fallback_ratio"] is not None
        and m["fallback_ratio"] <= config.max_fallback_ratio
    )
    probability_ok = (
        m["historical_probability"] is not None
        and m["historical_probability"] >= config.min_historical_probability
    )
    rr_ok = (
        m["reward_risk"] is not None
        and m["reward_risk"] >= config.min_reward_risk
    )
    appt_ok = (
        m["appt_x"] is not None
        and m["appt_x"] >= config.min_appt_x
    )
    dynamic_ok = _dynamic_entry_stop_gate(m, config)
    regime_ok = m["regime"] not in config.blocked_regimes

    quality = _quality_evaluation(m, config)

    hard_gates = {
        "benchmark_trust": trust_ok,
        "fallback_ratio": fallback_ok,
        "historical_probability": probability_ok,
        "reward_risk": rr_ok,
        "appt_x": appt_ok,
        "dynamic_entry_stop": dynamic_ok,
        "regime": regime_ok,
    }

    quality_gates = {
        "trading_confidence": quality["confidence_full"],
        "match_quality": quality["match_full"],
        "medium_quality": quality["medium_quality"],
        "strong_opportunity": quality["strong_opportunity"],
    }

    hard_failed = [k for k, v in hard_gates.items() if not v]
    quality_failed = [
        k for k, v in {
            "trading_confidence": quality["confidence_full"],
            "match_quality": quality["match_full"],
        }.items()
        if not v
    ]

    hard_passed = not hard_failed

    executable_buy = hard_passed and (
        quality["full_quality"]
        or (quality["medium_quality"] and quality["strong_opportunity"])
    )

    if not hard_passed:
        path = "HARD_GATE_BLOCK"
    elif quality["full_quality"]:
        path = "FULL_QUALITY"
    elif quality["medium_quality"] and quality["strong_opportunity"]:
        path = "STRONG_OPPORTUNITY_MEDIUM_QUALITY"
    else:
        path = "QUALITY_CONFIRMATION"

    return {
        "engine_policy": "CCE",
        "all_passed": executable_buy,
        "hard_gates_passed": hard_passed,
        "hard_gates": hard_gates,
        "quality_gates": quality_gates,
        "gates": {
            **hard_gates,
            "trading_confidence": quality["confidence_full"],
            "match_quality": quality["match_full"],
        },
        "failed_gates": hard_failed + quality_failed,
        "hard_failed_gates": hard_failed,
        "quality_failed_gates": quality_failed,
        "quality": quality,
        "decision_path": path,
        "metrics": dict(m),
    }


def _evaluate_nle(
    result: Mapping[str, Any],
    m: Mapping[str, Any],
    config: GateConfig,
) -> Dict[str, Any]:
    confidence = m["trading_confidence"]
    confidence_ok = (
        confidence is not None and confidence >= config.min_nle_confidence
    )
    benchmark_ok = _benchmark_gate(m, config)
    confirmation_ok = _confirmation_gate(m, config)
    dynamic_ok = _dynamic_entry_stop_gate(m, config)
    regime_ok = m["regime"] not in config.blocked_regimes

    hard_gates = {
        "benchmark_trust": benchmark_ok,
        "nle_confidence": confidence_ok,
        "confirmation": confirmation_ok,
        "dynamic_entry_stop": dynamic_ok,
        "regime": regime_ok,
    }

    hard_failed = [k for k, v in hard_gates.items() if not v]
    all_passed = not hard_failed

    if not all_passed:
        path = "HARD_GATE_BLOCK"
    else:
        path = "NLE_CONFIRMATION_PASS"

    return {
        "engine_policy": "NLE",
        "all_passed": all_passed,
        "hard_gates_passed": all_passed,
        "hard_gates": hard_gates,
        "quality_gates": {
            "nle_confidence": confidence_ok,
        },
        "gates": dict(hard_gates),
        "failed_gates": hard_failed,
        "hard_failed_gates": hard_failed,
        "quality_failed_gates": (
            [] if confidence_ok else ["nle_confidence"]
        ),
        "quality": {
            "confidence_ok": confidence_ok,
            "quality_mode": "NLE",
        },
        "decision_path": path,
        "metrics": dict(m),
    }


def evaluate_buy_gates(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Evaluate the engine-specific BUY gates."""
    engine = _detect_engine(result)
    m = _extract_metrics(result)

    if engine == "CCE":
        return _evaluate_cce(result, m, config)
    if engine == "NLE":
        return _evaluate_nle(result, m, config)

    # Unknown engine: fail closed. Never manufacture an engine-specific BUY.
    return {
        "engine_policy": "UNKNOWN",
        "all_passed": False,
        "hard_gates_passed": False,
        "hard_gates": {"engine_identified": False},
        "quality_gates": {},
        "gates": {"engine_identified": False},
        "failed_gates": ["engine_identified"],
        "hard_failed_gates": ["engine_identified"],
        "quality_failed_gates": [],
        "quality": {"quality_mode": "UNKNOWN"},
        "decision_path": "UNKNOWN_ENGINE_BLOCK",
        "metrics": dict(m),
    }


def build_decision(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Build the operational decision without changing engine evidence."""

    engine_signal = _norm(
        _get(
            result,
            "engine_signal",
            "signal",
            "operational_signal",
            default="WATCH",
        )
    )

    engine_policy = _detect_engine(result)
    gate = evaluate_buy_gates(result, config=config)
    m = gate["metrics"]

    if engine_signal in {"BUY", "STRONG_BUY"}:
        if gate["all_passed"]:
            operational_signal = "BUY"
            gate_status = "PASSED"
            gate_reason = (
                "cce_all_buy_gates_passed"
                if engine_policy == "CCE"
                else "nle_all_buy_gates_passed"
            )
        else:
            operational_signal = "BUY_ON_CONFIRMATION"
            gate_status = "BLOCKED"
            gate_reason = (
                "cce_buy_gate_failed"
                if engine_policy == "CCE"
                else "nle_buy_gate_failed"
            )

    elif engine_signal == "BUY_ON_CONFIRMATION":
        operational_signal = "BUY_ON_CONFIRMATION"
        gate_status = "CONFIRMATION_REQUIRED"
        gate_reason = "engine_requires_confirmation"

    elif engine_signal in {"SELL", "STRONG_SELL"}:
        operational_signal = "SELL"
        gate_status = "NOT_APPLICABLE"
        gate_reason = "non_bullish_signal_preserved"

    elif engine_signal in {"AVOID", "WATCH"}:
        operational_signal = engine_signal
        gate_status = "NOT_APPLICABLE"
        gate_reason = "non_bullish_signal_preserved"

    elif engine_signal in {"REVIEW_DATA", "NO_DATA", "INSUFFICIENT_HISTORY", "ERROR"}:
        operational_signal = engine_signal
        gate_status = "NOT_APPLICABLE"
        gate_reason = "terminal_non_actionable_signal_preserved"

    else:
        operational_signal = engine_signal or "WATCH"
        gate_status = "NOT_APPLICABLE"
        gate_reason = "unknown_signal_preserved"

    return {
        "version": VERSION,
        "engine_policy": engine_policy,
        "engine_signal": engine_signal,
        "operational_signal": operational_signal,
        "signal": operational_signal,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "decision_path": gate["decision_path"],
        "benchmark": result.get("benchmark") or result.get("selected_benchmark"),
        "benchmark_trust": m["benchmark_trust"] or None,
        "benchmark_trust_decision_grade": (
            bool(m["benchmark_grade"])
            if m["benchmark_grade"] is not None
            else m["benchmark_trust"] in config.decision_grade_trust
        ),
        "buy_gates_passed": gate["all_passed"],
        "hard_gates_passed": gate["hard_gates_passed"],
        "failed_gates": gate["failed_gates"],
        "hard_failed_gates": gate["hard_failed_gates"],
        "quality_failed_gates": gate["quality_failed_gates"],
        "gates": gate["gates"],
        "hard_gates": gate["hard_gates"],
        "quality_gates": gate["quality_gates"],
        "quality": gate["quality"],
        "metrics": m,
    }


def apply_decision_gate(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Backward-compatible public API."""
    return build_decision(result, config=config)


def _levels(
    *,
    current: float = 100.0,
    entry: float = 100.0,
    low: float = 95.0,
    high: float = 100.0,
    stop: float = 95.0,
) -> Dict[str, float]:
    return {
        "current_price": current,
        "entry_price": entry,
        "entry_zone_low": low,
        "entry_zone_high": high,
        "stop": stop,
    }


def _base_cce() -> Dict[str, Any]:
    return {
        "ticker": "TEST",
        "engine": "CHARACTERISTIC_CURVE_ENGINE",
        "engine_version": "1.7",
        "route": "MATURE",
        "signal": "BUY",
        "benchmark": "SPY",
        "benchmark_trust": "HIGH",
        "benchmark_trust_decision_grade": True,
        "trading_confidence": 75.0,
        "match_quality": "GOOD",
        "fallback_ratio": 0.10,
        "historical_probability_20d": 0.70,
        "reward_risk_20d": 2.00,
        "appt_x_20d": 0.80,
        "cycle": "CONSOLIDATION",
        "levels": _levels(),
    }


def _base_nle() -> Dict[str, Any]:
    return {
        "ticker": "SPCX",
        "engine": "NEW_LISTING_ENGINE",
        "engine_version": "1.4.1",
        "route": "NEW_LISTING",
        "signal": "BUY",
        "benchmark": "SPY",
        "benchmark_trust": "HIGH",
        "benchmark_trust_decision_grade": True,
        "confidence": 68.05,
        "regime": "BULLISH",
        "levels": {
            "current_price": 151.87,
            "entry_price": 148.31,
            "entry_zone_low": 144.74,
            "entry_zone_high": 153.65,
            "stop": 130.48,
        },
        "confirmation": {
            "required": True,
            "confirmed": True,
            "score": 4,
            "max_score": 6,
        },
        "trading_days": 70,
    }


def _run_case(name: str, result: Dict[str, Any], expected: str) -> Dict[str, Any]:
    output = apply_decision_gate(result)
    actual = output["operational_signal"]
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "decision_path": output["decision_path"],
        "engine_policy": output["engine_policy"],
        "failed_gates": output["failed_gates"],
        "passed": actual == expected,
    }


def run_self_test() -> bool:
    cases = []

    # CCE V1.4-compatible cases.
    cases.append(("CCE_FULL_PASS", _base_cce(), "BUY"))

    case = _base_cce()
    case["trading_confidence"] = 55.64
    case["match_quality"] = "MODERATE"
    case["appt_x_20d"] = 0.921634
    case["historical_probability_20d"] = 0.70
    case["reward_risk_20d"] = 2.828
    case["fallback_ratio"] = 0.05
    case["cycle"] = "CONFIRMED_BULL"
    cases.append(("CCE_TSM_POLICY", case, "BUY"))

    case = _base_cce()
    case["benchmark_trust"] = "CONTEXT_ONLY"
    case["benchmark_trust_decision_grade"] = False
    cases.append(("CCE_CONTEXT_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_cce()
    case["fallback_ratio"] = 0.51
    cases.append(("CCE_FALLBACK_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_cce()
    case["historical_probability_20d"] = 0.59
    cases.append(("CCE_PROBABILITY_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_cce()
    case["reward_risk_20d"] = 1.24
    cases.append(("CCE_RR_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_cce()
    case["appt_x_20d"] = -0.01
    cases.append(("CCE_NEGATIVE_APPT", case, "BUY_ON_CONFIRMATION"))

    case = _base_cce()
    case["cycle"] = "DOWNTREND"
    cases.append(("CCE_BLOCKED_REGIME", case, "BUY_ON_CONFIRMATION"))

    # NLE-specific cases.
    cases.append(("NLE_FULL_PASS", _base_nle(), "BUY"))

    case = _base_nle()
    # Deliberately omit all CCE-only historical metrics.
    cases.append(("NLE_WITHOUT_CCE_METRICS", case, "BUY"))

    case = _base_nle()
    case["benchmark_trust"] = "CONTEXT_ONLY"
    case["benchmark_trust_decision_grade"] = False
    cases.append(("NLE_CONTEXT_ONLY_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_nle()
    case["confirmation"]["confirmed"] = False
    cases.append(("NLE_CONFIRMATION_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_nle()
    case["confidence"] = 59.99
    cases.append(("NLE_LOW_CONFIDENCE", case, "BUY_ON_CONFIRMATION"))

    case = _base_nle()
    case["regime"] = "DOWNTREND"
    cases.append(("NLE_BLOCKED_REGIME", case, "BUY_ON_CONFIRMATION"))

    case = _base_nle()
    case["signal"] = "BUY_ON_CONFIRMATION"
    cases.append(("NLE_ENGINE_CONFIRMATION", case, "BUY_ON_CONFIRMATION"))

    # Signal preservation / terminal routes.
    for signal in ("WATCH", "AVOID", "SELL", "REVIEW_DATA", "NO_DATA"):
        case = _base_cce()
        case["signal"] = signal
        cases.append((f"PRESERVE_{signal}", case, signal))

    passed = 0
    failures = []

    print("=" * 118)
    print("AI TRADER - DECISION GATE V1.5 SELF-TEST")
    print("=" * 118)
    print()
    print(
        f"{'Case':34} | {'Policy':7} | {'Expected':20} | "
        f"{'Actual':20} | {'Path':34} | Result"
    )
    print("-" * 118)

    for name, result, expected in cases:
        item = _run_case(name, result, expected)
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{name:34} | {item['engine_policy']:7} | "
            f"{expected:20} | {item['actual']:20} | "
            f"{item['decision_path']:34} | {status}"
        )
        if item["passed"]:
            passed += 1
        else:
            failures.append(item)

    print()
    print("=" * 118)
    print(f"RESULT: {passed}/{len(cases)} cases passed")
    print("=" * 118)

    if failures:
        print()
        print("FAILED CASES:")
        for failure in failures:
            print(json.dumps(failure, indent=2))
        print()
        print("DECISION GATE V1.5 SELF-TEST: FAIL")
        return False

    print("DECISION GATE V1.5 SELF-TEST: PASS")
    return True


def _demo_tsm() -> None:
    result = _base_cce()
    result.update({
        "ticker": "TSM",
        "trading_confidence": 55.64,
        "match_quality": "MODERATE",
        "fallback_ratio": 0.05,
        "historical_probability_20d": 0.70,
        "reward_risk_20d": 2.828,
        "appt_x_20d": 0.921634,
        "cycle": "CONFIRMED_BULL",
    })
    print(json.dumps(build_decision(result), indent=2))


def _demo_spcx() -> None:
    print(json.dumps(build_decision(_base_nle()), indent=2))


def _run_real_ticker(ticker: str) -> int:
    """
    Execute the real production path:

        Orchestrator V1.5.1 -> engine (CCE/NLE) -> Decision Gate V1.5

    The Gate itself remains independent; the Orchestrator is imported lazily
    so normal self-tests do not require the complete application stack.
    """
    from app.ai.asset_analysis_orchestrator_v1_5_1 import analyze_ticker

    symbol = str(ticker or "").strip().upper()
    if not symbol:
        print("ERROR: ticker cannot be empty.")
        return 2

    try:
        analysis = analyze_ticker(symbol)
    except Exception as exc:
        print(f"{symbol} => Orchestrator error: {exc}")
        return 1

    route = str(analysis.get("route") or "").upper()
    raw_signal = str(analysis.get("signal") or "").upper()

    print("=" * 118)
    print(f"AI TRADER - DECISION GATE V1.5 REAL TEST: {symbol}")
    print("=" * 118)
    print(f"route={route or 'UNKNOWN'}")
    print(f"engine={analysis.get('engine') or 'NONE'}")
    print(f"engine_version={analysis.get('engine_version') or 'UNKNOWN'}")
    print(f"raw_signal={raw_signal or 'UNKNOWN'}")
    print(f"benchmark={analysis.get('benchmark') or 'NONE'}")
    print(f"benchmark_trust={analysis.get('benchmark_trust') or 'NONE'}")
    print()

    # REVIEW_DATA and other terminal routes must not be forced through BUY
    # gates. This mirrors the Watchlist Scanner safety contract.
    if route in {"REVIEW_DATA", "NO_DATA", "DATA_ERROR"} or raw_signal in {
        "REVIEW_DATA",
        "NO_DATA",
        "INSUFFICIENT_HISTORY",
        "ERROR",
    }:
        print(f"operational_signal={raw_signal or route or 'REVIEW_DATA'}")
        print("gate_status=NOT_APPLICABLE")
        print("gate_reason=terminal_non_actionable_route")
        print("=" * 118)
        return 0

    engine_result = analysis.get("engine_result")
    if not isinstance(engine_result, Mapping):
        engine_result = dict(analysis)

    # Normalize authoritative Orchestrator fields at the Gate boundary.
    gate_input = dict(engine_result)

    normalized_fields = (
        "ticker",
        "signal",
        "score",
        "confidence",
        "trading_confidence",
        "match_quality",
        "fallback_ratio",
        "historical_probability_20d",
        "reward_risk_20d",
        "appt_x_20d",
        "expected_return_20d",
        "cycle",
        "regime",
        "levels",
        "current_price",
        "entry_price",
        "entry_zone_low",
        "entry_zone_high",
        "stop",
        "benchmark",
        "selected_benchmark",
        "benchmark_trust",
        "benchmark_trust_decision_grade",
        "engine",
        "engine_version",
        "route",
        "classification",
        "confirmation",
        "confirmation_required",
        "confirmed",
        "confirmation_score",
        "confirmation_max_score",
        "trading_days",
        "listing_age_days",
    )

    for key in normalized_fields:
        if analysis.get(key) is not None:
            gate_input[key] = analysis[key]

    # Preserve nested CCE/NLE structures when they are only present in the
    # Orchestrator boundary rather than directly in engine_result.
    for key in (
        "historical_pattern",
        "decision_context",
        "decision_quality",
        "dynamic_levels",
        "benchmark_resolution",
        "orchestrator_benchmark_resolution",
    ):
        if isinstance(analysis.get(key), Mapping):
            gate_input[key] = dict(analysis[key])

    try:
        gate = apply_decision_gate(gate_input)
    except Exception as exc:
        print(f"Decision Gate V1.5 error: {exc}")
        return 1

    print(json.dumps(gate, indent=2, default=str))
    print()
    print("=" * 118)
    print(
        f"FINAL: {symbol} | "
        f"engine={gate.get('engine_policy')} | "
        f"raw={gate.get('engine_signal')} | "
        f"operational={gate.get('operational_signal')} | "
        f"status={gate.get('gate_status')}"
    )
    print("=" * 118)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decision Gate V1.5")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo-tsm", action="store_true")
    parser.add_argument("--demo-spcx", action="store_true")
    parser.add_argument(
        "--ticker",
        type=str,
        help="run the real Orchestrator -> Decision Gate V1.5 flow for one ticker",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.ticker:
        return _run_real_ticker(args.ticker)

    if args.demo_tsm:
        _demo_tsm()
        return 0
    if args.demo_spcx:
        _demo_spcx()
        return 0

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
