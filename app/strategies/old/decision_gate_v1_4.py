"""
AI Trader - Decision Gate V1.4
==============================

Policy evolution from V1.3
---------------------------
V1.4 preserves trading_confidence and match_quality, but changes their role:

    V1.3:
        trading_confidence < 60  -> hard BUY blocker
        match_quality < GOOD     -> hard BUY blocker

    V1.4:
        They are QUALITY MODIFIERS.

A raw BUY can become operational BUY when:
    1. All hard opportunity/risk gates pass.
    2. Evidence quality is at least MEDIUM.
    3. The historical opportunity is sufficiently strong when quality is
       below the full-quality threshold.

This avoids allowing a weak statistical opportunity to become BUY merely
because confidence/match thresholds were relaxed.

Hard BUY gates:
    benchmark trust       -> HIGH/MEDIUM/LOW + valid decision grade
    fallback ratio        -> <= 50%
    historical probability -> >= 60%
    reward/risk            -> >= 1.25
    APPT_X                 -> >= 0
    dynamic entry/stop     -> valid
    regime                 -> not blocked

Full-quality BUY:
    trading confidence >= 60
    match quality >= GOOD

Strong-opportunity BUY with medium quality:
    APPT_X >= 0.50
    historical probability >= 65%
    reward/risk >= 1.50
    trading confidence >= 50
    match quality >= MODERATE
    fallback ratio <= 50%

Otherwise:
    BUY_ON_CONFIRMATION

The engine signal is never overwritten.
SELL / AVOID / WATCH are preserved.

The module is intentionally independent from the production pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


VERSION = "1.4"

DECISION_GRADE_TRUST = {"HIGH", "MEDIUM", "LOW"}
BLOCKED_BUY_TRUST = {"CONTEXT_ONLY", "NONE", ""}

BLOCKED_REGIMES = {
    "CAPITULATION",
    "DOWNTREND",
    "DISTRIBUTION",
    "EXTREME_MOMENTUM",
}

MATCH_QUALITY_RANK = {
    "NONE": 0,
    "WEAK": 1,
    "MODERATE": 2,
    "GOOD": 3,
    "STRONG": 4,
}


@dataclass(frozen=True)
class GateConfig:
    # Quality thresholds retained from V1.3.
    min_trading_confidence: float = 60.0
    min_match_quality: str = "GOOD"

    # Hard opportunity/risk gates.
    max_fallback_ratio: float = 0.50
    min_historical_probability: float = 0.60
    min_reward_risk: float = 1.25
    min_appt_x: float = 0.0

    # Medium-quality evidence required for the strong-opportunity path.
    min_medium_confidence: float = 50.0
    min_medium_match_quality: str = "MODERATE"

    # Strong historical opportunity required to compensate for medium
    # evidence quality.
    strong_opportunity_appt_x: float = 0.50
    strong_opportunity_probability: float = 0.65
    strong_opportunity_reward_risk: float = 1.50

    decision_grade_trust: frozenset[str] = field(
        default_factory=lambda: frozenset(DECISION_GRADE_TRUST)
    )
    blocked_regimes: frozenset[str] = field(
        default_factory=lambda: frozenset(BLOCKED_REGIMES)
    )


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


def _extract_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract CCE/NLE/Orchestrator-compatible metrics."""

    historical = _mapping(result.get("historical_pattern"))
    decision = _mapping(result.get("decision_context"))
    quality = _mapping(result.get("decision_quality"))
    levels = _mapping(result.get("levels"))

    # Support both normalized orchestrator levels and legacy engine levels.
    if not levels:
        levels = _mapping(result.get("legacy_levels"))
    if not levels:
        levels = _mapping(result.get("dynamic_levels"))

    metrics = {
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
            _get(
                result,
                "appt_x_20d",
                default=historical.get("appt_x_20d"),
            )
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
        "confirmation_required": _get(
            result,
            "confirmation_required",
            default=decision.get("confirmation_required"),
        ),
        "confirmed": _get(
            result,
            "confirmed",
            default=decision.get("confirmed"),
        ),
        "current_price": _float(
            _get(result, "current_price", default=levels.get("current_price"))
        ),
        "entry_price": _float(
            _get(result, "entry_price", default=levels.get("entry_price"))
        ),
        "entry_zone_low": _float(
            _get(result, "entry_zone_low", default=levels.get("entry_zone_low"))
        ),
        "entry_zone_high": _float(
            _get(
                result,
                "entry_zone_high",
                default=levels.get("entry_zone_high"),
            )
        ),
        "stop": _float(
            _get(
                result,
                "stop",
                "dynamic_stop",
                default=levels.get("stop", levels.get("dynamic_stop")),
            )
        ),
    }

    return metrics


def _dynamic_entry_stop_gate(m: Mapping[str, Any]) -> bool:
    """Validate bullish dynamic entry/stop geometry."""

    current = m["current_price"]
    entry = m["entry_price"]
    zone_low = m["entry_zone_low"]
    zone_high = m["entry_zone_high"]
    stop = m["stop"]

    values = (current, entry, zone_low, zone_high, stop)
    if any(value is None for value in values):
        return False

    if any(value <= 0 for value in values):
        return False

    if zone_low > zone_high:
        return False

    if not (zone_low <= entry <= zone_high):
        return False

    if stop >= entry:
        return False

    if stop > zone_high:
        return False

    # Executable BUY must currently be at/inside the allowed entry zone.
    if current < stop:
        return False

    if current > zone_high:
        return False

    return True


def _quality_evaluation(
    m: Mapping[str, Any],
    config: GateConfig,
) -> Dict[str, Any]:
    """Evaluate quality as a modifier, not as an unconditional blocker."""

    confidence = m["trading_confidence"]
    match_quality = m["match_quality"]

    confidence_full = (
        confidence is not None
        and confidence >= config.min_trading_confidence
    )

    min_full_rank = MATCH_QUALITY_RANK.get(
        _norm(config.min_match_quality), 3
    )
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
            "FULL"
            if full_quality
            else "MEDIUM"
            if medium_quality
            else "LOW"
        ),
    }


def evaluate_buy_gates(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Evaluate BUY gates.

    `trading_confidence` and `match_quality` are reported as quality gates,
    but are not hard blockers when strong historical opportunity exists and
    minimum medium-quality evidence is present.
    """

    m = _extract_metrics(result)

    trust = m["benchmark_trust"]
    grade = m["benchmark_grade"]

    trust_ok = trust in config.decision_grade_trust
    if grade is not None:
        trust_ok = trust_ok and bool(grade)

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

    dynamic_entry_stop_ok = _dynamic_entry_stop_gate(m)
    regime_ok = m["regime"] not in config.blocked_regimes

    quality = _quality_evaluation(m, config)

    # Hard gates. Quality is deliberately excluded from this AND.
    hard_gates = {
        "benchmark_trust": trust_ok,
        "fallback_ratio": fallback_ok,
        "historical_probability": probability_ok,
        "reward_risk": rr_ok,
        "appt_x": appt_ok,
        "dynamic_entry_stop": dynamic_entry_stop_ok,
        "regime": regime_ok,
    }

    quality_gates = {
        "trading_confidence": quality["confidence_full"],
        "match_quality": quality["match_full"],
        "medium_quality": quality["medium_quality"],
        "strong_opportunity": quality["strong_opportunity"],
    }

    hard_failed = [
        name for name, passed in hard_gates.items() if not passed
    ]

    quality_failed = [
        name for name, passed in {
            "trading_confidence": quality["confidence_full"],
            "match_quality": quality["match_full"],
        }.items()
        if not passed
    ]

    hard_passed = not hard_failed

    # Two BUY paths:
    # A) Full-quality evidence.
    # B) Medium-quality evidence + sufficiently strong historical opportunity.
    executable_buy = (
        hard_passed
        and (
            quality["full_quality"]
            or (
                quality["medium_quality"]
                and quality["strong_opportunity"]
            )
        )
    )

    if not hard_passed:
        decision_path = "HARD_GATE_BLOCK"
    elif quality["full_quality"]:
        decision_path = "FULL_QUALITY"
    elif quality["medium_quality"] and quality["strong_opportunity"]:
        decision_path = "STRONG_OPPORTUNITY_MEDIUM_QUALITY"
    else:
        decision_path = "QUALITY_CONFIRMATION"

    return {
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
        "decision_path": decision_path,
        "metrics": m,
    }


def build_decision(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Build the operational decision without changing the engine signal."""

    engine_signal = _norm(
        _get(
            result,
            "engine_signal",
            "signal",
            "operational_signal",
            default="WATCH",
        )
    )

    gate = evaluate_buy_gates(result, config=config)
    m = gate["metrics"]

    if engine_signal in {"BUY", "STRONG_BUY"}:
        if gate["all_passed"]:
            operational_signal = "BUY"
            gate_status = "PASSED"
            if gate["decision_path"] == "FULL_QUALITY":
                gate_reason = "all_buy_gates_passed_full_quality"
            else:
                gate_reason = "strong_opportunity_overcame_medium_quality"
        else:
            operational_signal = "BUY_ON_CONFIRMATION"
            gate_status = "BLOCKED"
            if gate["hard_failed_gates"]:
                gate_reason = "hard_buy_gate_failed"
            else:
                gate_reason = "quality_requires_confirmation"

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

    else:
        operational_signal = engine_signal or "WATCH"
        gate_status = "NOT_APPLICABLE"
        gate_reason = "unknown_signal_preserved"

    return {
        "version": VERSION,
        "engine_signal": engine_signal,
        "operational_signal": operational_signal,
        "signal": operational_signal,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "decision_path": gate["decision_path"],
        "benchmark": result.get("benchmark")
        or result.get("selected_benchmark"),
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
    """Public API alias."""
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


def _base_case() -> Dict[str, Any]:
    return {
        "ticker": "TEST",
        "signal": "BUY",
        "benchmark": "QQQ",
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


def _run_case(
    name: str,
    result: Dict[str, Any],
    expected: str,
) -> Dict[str, Any]:
    output = apply_decision_gate(result)
    actual = output["operational_signal"]
    passed = actual == expected
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "decision_path": output["decision_path"],
        "gate_status": output["gate_status"],
        "failed_gates": output["failed_gates"],
        "passed": passed,
    }


def run_self_test() -> bool:
    """Run V1.3 compatibility tests plus V1.4 policy tests."""

    cases = []

    # V1.3-compatible baseline / preservation cases.
    cases.append(("FULL_PASS", _base_case(), "BUY"))

    case = _base_case()
    case["benchmark_trust"] = "CONTEXT_ONLY"
    case["benchmark_trust_decision_grade"] = False
    cases.append(("CONTEXT_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["benchmark_trust"] = "NONE"
    case["benchmark_trust_decision_grade"] = False
    cases.append(("NONE_BLOCK", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["trading_confidence"] = 59.99
    cases.append(("LOW_CONFIDENCE_STRONG_OPP", case, "BUY"))

    case = _base_case()
    case["match_quality"] = "MODERATE"
    cases.append(("WEAK_MATCH_STRONG_OPP", case, "BUY"))

    case = _base_case()
    case["fallback_ratio"] = 0.51
    cases.append(("HIGH_FALLBACK", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["historical_probability_20d"] = 0.59
    cases.append(("LOW_PROBABILITY", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["reward_risk_20d"] = 1.24
    cases.append(("LOW_RR", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["appt_x_20d"] = -0.01
    cases.append(("NEGATIVE_APPT", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["cycle"] = "CAPITULATION"
    cases.append(("BLOCKED_REGIME", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["levels"]["stop"] = 105.0
    cases.append(("DYNAMIC_STOP_INVALID", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["levels"]["entry_price"] = 110.0
    cases.append(("DYNAMIC_ENTRY_ABOVE_ZONE", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    del case["levels"]
    cases.append(("DYNAMIC_LEVELS_MISSING", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["signal"] = "BUY_ON_CONFIRMATION"
    case["trading_confidence"] = 40.0
    case["match_quality"] = "WEAK"
    cases.append(("ENGINE_CONFIRMATION", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["signal"] = "WATCH"
    cases.append(("WATCH_PRESERVED", case, "WATCH"))

    case = _base_case()
    case["signal"] = "AVOID"
    cases.append(("AVOID_PRESERVED", case, "AVOID"))

    case = _base_case()
    case["signal"] = "SELL"
    cases.append(("SELL_PRESERVED", case, "SELL"))

    # V1.4-specific policy tests.
    case = _base_case()
    case["trading_confidence"] = 55.64
    case["match_quality"] = "MODERATE"
    case["appt_x_20d"] = 0.921634
    case["historical_probability_20d"] = 0.70
    case["reward_risk_20d"] = 2.828
    case["fallback_ratio"] = 0.05
    case["cycle"] = "CONFIRMED_BULL"
    case["ticker"] = "TSM"
    cases.append(("TSM_STRONG_OPPORTUNITY", case, "BUY"))

    case = _base_case()
    case["trading_confidence"] = 50.0
    case["match_quality"] = "MODERATE"
    case["appt_x_20d"] = 0.50
    case["historical_probability_20d"] = 0.65
    case["reward_risk_20d"] = 1.50
    cases.append(("MEDIUM_QUALITY_BOUNDARY", case, "BUY"))

    case = _base_case()
    case["trading_confidence"] = 49.99
    case["match_quality"] = "MODERATE"
    case["appt_x_20d"] = 0.80
    case["historical_probability_20d"] = 0.70
    case["reward_risk_20d"] = 2.00
    cases.append(("MEDIUM_CONFIDENCE_TOO_LOW", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["trading_confidence"] = 55.0
    case["match_quality"] = "MODERATE"
    case["appt_x_20d"] = 0.49
    case["historical_probability_20d"] = 0.70
    case["reward_risk_20d"] = 2.00
    cases.append(("OPPORTUNITY_NOT_STRONG_ENOUGH", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["trading_confidence"] = 55.0
    case["match_quality"] = "WEAK"
    case["appt_x_20d"] = 1.00
    case["historical_probability_20d"] = 0.80
    case["reward_risk_20d"] = 3.00
    cases.append(("QUALITY_TOO_LOW", case, "BUY_ON_CONFIRMATION"))

    case = _base_case()
    case["trading_confidence"] = 55.0
    case["match_quality"] = "MODERATE"
    case["fallback_ratio"] = 0.50
    cases.append(("FALLBACK_BOUNDARY", case, "BUY"))

    case = _base_case()
    case["trading_confidence"] = 55.0
    case["match_quality"] = "MODERATE"
    case["fallback_ratio"] = 0.5001
    cases.append(("FALLBACK_ABOVE_HARD_LIMIT", case, "BUY_ON_CONFIRMATION"))

    passed = 0
    failures = []

    print("=" * 104)
    print("AI TRADER - DECISION GATE V1.4 SELF-TEST")
    print("=" * 104)
    print()
    print(
        f"{'Case':34} | {'Expected':20} | {'Actual':20} | "
        f"{'Path':34} | Result"
    )
    print("-" * 104)

    for name, result, expected in cases:
        item = _run_case(name, result, expected)
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{name:34} | {expected:20} | {item['actual']:20} | "
            f"{item['decision_path']:34} | {status}"
        )
        if item["passed"]:
            passed += 1
        else:
            failures.append(item)

    print()
    print("=" * 104)
    print(f"RESULT: {passed}/{len(cases)} cases passed")
    print("=" * 104)

    if failures:
        print()
        print("FAILED CASES:")
        for failure in failures:
            print(json.dumps(failure, indent=2))
        print()
        print("DECISION GATE V1.4 SELF-TEST: FAIL")
        return False

    print("DECISION GATE V1.4 SELF-TEST: PASS")
    return True


def _demo() -> None:
    """Print the TSM policy example used for manual inspection."""

    result = _base_case()
    result.update(
        {
            "ticker": "TSM",
            "trading_confidence": 55.64,
            "match_quality": "MODERATE",
            "fallback_ratio": 0.05,
            "historical_probability_20d": 0.70,
            "reward_risk_20d": 2.828,
            "appt_x_20d": 0.921634,
            "cycle": "CONFIRMED_BULL",
        }
    )

    print(json.dumps(build_decision(result), indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decision Gate V1.4")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the independent V1.4 self-test",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print the TSM strong-opportunity example",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        _demo()
        return 0

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
