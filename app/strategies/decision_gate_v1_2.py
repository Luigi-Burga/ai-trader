"""
Decision Gate V1.2
==================

Independent decision-policy layer for AI Trader.

Purpose
-------
Convert an engine-level signal into an operational signal without changing
the underlying analysis engine.

The gate combines:
    1. Engine signal
    2. Benchmark Trust
    3. Trading confidence
    4. Match quality
    5. Historical positive probability
    6. Historical reward/risk
    7. Fallback-match ratio
    8. Market/asset regime
    9. Engine confirmation state

Policy
------
Benchmark Trust:
    HIGH          -> decision-grade
    MEDIUM        -> decision-grade
    LOW           -> decision-grade, lower-quality context
    CONTEXT_ONLY  -> cannot validate a BUY
    NONE          -> cannot validate a BUY

For a raw BUY/STRONG_BUY to become operational BUY, the following
confirmation gates must pass:

    trading confidence >= 60
    match quality      >= GOOD
    fallback ratio     <= 25%
    historical prob    >= 60%
    reward/risk        >= 1.25
    benchmark trust    in HIGH/MEDIUM/LOW
    regime             not in blocked regimes

If one or more critical conditions fail, the underlying engine signal is
preserved as `engine_signal`, while the operational signal becomes
BUY_ON_CONFIRMATION.

BUY_ON_CONFIRMATION is never promoted automatically to BUY by this gate.
SELL/AVOID/WATCH are preserved.

This module is intentionally independent. It does not import:
    - main.py
    - production signal_engine.py
    - watchlist_scanner.py
    - characteristic_curve.py
    - new_listing_engine.py
    - benchmark_resolver.py

It can therefore be tested before production integration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


VERSION = "1.2"

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
    min_trading_confidence: float = 60.0
    min_match_quality: str = "GOOD"
    max_fallback_ratio: float = 0.25
    min_historical_probability: float = 0.60
    min_reward_risk: float = 1.25
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
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _get(result: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    return default


def _extract_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract CCE/NLE-compatible metrics from a result dictionary."""

    historical = result.get("historical_pattern")
    if not isinstance(historical, Mapping):
        historical = {}

    decision = result.get("decision_context")
    if not isinstance(decision, Mapping):
        decision = {}

    quality = result.get("decision_quality")
    if not isinstance(quality, Mapping):
        quality = {}

    metrics = {
        "trading_confidence": _float(
            _get(
                result,
                "trading_confidence",
                "confidence",
                default=decision.get("confidence"),
            )
        ),
        "match_quality": _norm(
            _get(
                result,
                "match_quality",
                default=decision.get("match_quality"),
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
    }

    return metrics


def evaluate_buy_gates(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Evaluate all BUY gates independently.

    A missing critical metric is treated as a failed gate. This is deliberate:
    the Decision Gate must not infer missing evidence.
    """
    m = _extract_metrics(result)

    trust = m["benchmark_trust"]
    grade = m["benchmark_grade"]

    confidence = m["trading_confidence"]
    match_quality = m["match_quality"]
    fallback_ratio = m["fallback_ratio"]
    historical_probability = m["historical_probability"]
    reward_risk = m["reward_risk"]
    regime = m["regime"]

    trust_ok = trust in config.decision_grade_trust
    # If explicit grade is available, it must also be true.
    if grade is not None:
        trust_ok = trust_ok and bool(grade)

    confidence_ok = (
        confidence is not None
        and confidence >= config.min_trading_confidence
    )

    min_quality_rank = MATCH_QUALITY_RANK.get(
        _norm(config.min_match_quality), 3
    )
    match_ok = (
        match_quality in MATCH_QUALITY_RANK
        and MATCH_QUALITY_RANK[match_quality] >= min_quality_rank
    )

    fallback_ok = (
        fallback_ratio is not None
        and fallback_ratio <= config.max_fallback_ratio
    )

    probability_ok = (
        historical_probability is not None
        and historical_probability >= config.min_historical_probability
    )

    rr_ok = reward_risk is not None and reward_risk >= config.min_reward_risk

    regime_ok = regime not in config.blocked_regimes

    gates = {
        "benchmark_trust": trust_ok,
        "trading_confidence": confidence_ok,
        "match_quality": match_ok,
        "fallback_ratio": fallback_ok,
        "historical_probability": probability_ok,
        "reward_risk": rr_ok,
        "regime": regime_ok,
    }

    failed = [name for name, passed in gates.items() if not passed]

    return {
        "all_passed": not failed,
        "gates": gates,
        "failed_gates": failed,
        "metrics": m,
    }


def build_decision(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Build the operational decision.

    The engine signal is never overwritten. `operational_signal` is the
    signal allowed to leave the decision gate.
    """
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
            gate_reason = "all_buy_gates_passed"
        else:
            operational_signal = "BUY_ON_CONFIRMATION"
            gate_status = "BLOCKED"
            gate_reason = "buy_requires_confirmation"
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
        "benchmark": result.get("benchmark")
        or result.get("selected_benchmark"),
        "benchmark_trust": m["benchmark_trust"] or None,
        "benchmark_trust_decision_grade": (
            bool(m["benchmark_grade"])
            if m["benchmark_grade"] is not None
            else m["benchmark_trust"] in config.decision_grade_trust
        ),
        "buy_gates_passed": gate["all_passed"],
        "failed_gates": gate["failed_gates"],
        "gates": gate["gates"],
        "metrics": m,
    }


def apply_decision_gate(
    result: Mapping[str, Any],
    config: GateConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Public alias for build_decision()."""
    return build_decision(result, config=config)


# ---------------------------------------------------------------------------
# Independent self-test
# ---------------------------------------------------------------------------

def _case(
    name: str,
    *,
    signal: str,
    trust: str,
    grade: bool,
    confidence: Optional[float] = 75,
    match_quality: Optional[str] = "GOOD",
    fallback_ratio: Optional[float] = 0.10,
    historical_probability: Optional[float] = 0.70,
    reward_risk: Optional[float] = 2.0,
    regime: str = "CONSOLIDATION",
    expected: str,
) -> Dict[str, Any]:
    result = {
        "ticker": name,
        "signal": signal,
        "benchmark": "TEST",
        "benchmark_trust": trust,
        "benchmark_trust_decision_grade": grade,
        "confidence": confidence,
        "trading_confidence": confidence,
        "match_quality": match_quality,
        "fallback_ratio": fallback_ratio,
        "historical_probability_20d": historical_probability,
        "reward_risk_20d": reward_risk,
        "cycle": regime,
    }
    decision = build_decision(result)
    passed = decision["operational_signal"] == expected
    return {
        "name": name,
        "engine_signal": signal,
        "trust": trust,
        "expected": expected,
        "actual": decision["operational_signal"],
        "gate_status": decision["gate_status"],
        "failed_gates": decision["failed_gates"],
        "passed": passed,
    }


def run_self_test() -> bool:
    """
    Test boundary conditions and representative CCE/NLE scenarios.
    """
    cases = [
        _case(
            "FULL_PASS",
            signal="BUY",
            trust="HIGH",
            grade=True,
            expected="BUY",
        ),
        _case(
            "MEDIUM_PASS",
            signal="BUY",
            trust="MEDIUM",
            grade=True,
            expected="BUY",
        ),
        _case(
            "LOW_PASS",
            signal="BUY",
            trust="LOW",
            grade=True,
            expected="BUY",
        ),
        _case(
            "CONTEXT_BLOCK",
            signal="BUY",
            trust="CONTEXT_ONLY",
            grade=False,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "NONE_BLOCK",
            signal="BUY",
            trust="NONE",
            grade=False,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "LOW_CONFIDENCE",
            signal="BUY",
            trust="HIGH",
            grade=True,
            confidence=59.99,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "WEAK_MATCH",
            signal="BUY",
            trust="HIGH",
            grade=True,
            match_quality="MODERATE",
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "HIGH_FALLBACK",
            signal="BUY",
            trust="HIGH",
            grade=True,
            fallback_ratio=0.26,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "LOW_PROBABILITY",
            signal="BUY",
            trust="HIGH",
            grade=True,
            historical_probability=0.59,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "LOW_RR",
            signal="BUY",
            trust="HIGH",
            grade=True,
            reward_risk=1.24,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "BLOCKED_REGIME",
            signal="BUY",
            trust="HIGH",
            grade=True,
            regime="CAPITULATION",
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "ENGINE_CONFIRMATION",
            signal="BUY_ON_CONFIRMATION",
            trust="HIGH",
            grade=True,
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "WATCH_PRESERVED",
            signal="WATCH",
            trust="CONTEXT_ONLY",
            grade=False,
            expected="WATCH",
        ),
        _case(
            "AVOID_PRESERVED",
            signal="AVOID",
            trust="MEDIUM",
            grade=True,
            expected="AVOID",
        ),
        _case(
            "SELL_PRESERVED",
            signal="SELL",
            trust="CONTEXT_ONLY",
            grade=False,
            expected="SELL",
        ),
        # Realistic recent AI Trader cases.
        _case(
            "MSTU_REALISTIC",
            signal="BUY",
            trust="CONTEXT_ONLY",
            grade=False,
            confidence=79,
            match_quality=None,
            fallback_ratio=None,
            historical_probability=None,
            reward_risk=None,
            regime="",
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "PLTR_REALISTIC",
            signal="BUY_ON_CONFIRMATION",
            trust="MEDIUM",
            grade=True,
            confidence=52.25,
            match_quality="MODERATE",
            fallback_ratio=0.25,
            historical_probability=0.75,
            reward_risk=3.093,
            regime="CONSOLIDATION",
            expected="BUY_ON_CONFIRMATION",
        ),
        _case(
            "ASMG_REALISTIC",
            signal="AVOID",
            trust="MEDIUM",
            grade=True,
            expected="AVOID",
        ),
    ]

    print("=" * 92)
    print("AI TRADER - DECISION GATE V1.2 SELF-TEST")
    print("=" * 92)
    print()
    print(
        f"{'Case':22} | {'Engine':20} | {'Trust':12} | "
        f"{'Expected':20} | {'Actual':20} | Result"
    )
    print("-" * 92)

    failures = []
    for item in cases:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{item['name']:22} | "
            f"{item['engine_signal']:20} | "
            f"{item['trust']:12} | "
            f"{item['expected']:20} | "
            f"{item['actual']:20} | {status}"
        )
        if not item["passed"]:
            failures.append(item)

    print()
    print("=" * 92)
    print(f"RESULT: {len(cases) - len(failures)}/{len(cases)} cases passed")
    print("=" * 92)

    if failures:
        print()
        print("FAILED CASES:")
        for failure in failures:
            print(json.dumps(failure, indent=2))
        print()
        print("DECISION GATE V1.2 SELF-TEST: FAIL")
        return False

    print("DECISION GATE V1.2 SELF-TEST: PASS")
    return True


def _demo() -> None:
    """Small JSON demonstration for manual inspection."""
    sample = {
        "ticker": "MSTU",
        "signal": "BUY",
        "benchmark": "SPY",
        "benchmark_trust": "CONTEXT_ONLY",
        "benchmark_trust_decision_grade": False,
        "confidence": 79.0,
        "trading_confidence": 79.0,
        "cycle": "CONSOLIDATION",
    }
    print(json.dumps(build_decision(sample), indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decision Gate V1.2")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run independent Decision Gate self-test",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print a representative decision as JSON",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        _demo()
        return 0

    # Default execution is the self-test so `python -m ...` is safe.
    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
