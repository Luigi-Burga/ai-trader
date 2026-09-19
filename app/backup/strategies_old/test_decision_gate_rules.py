"""
Decision Gate V1.2 - Rule Test
==============================

Standalone unit/integration-style test for Decision Gate V1.2.

Purpose
-------
Validates the eight BUY-specific rules independently of market data:

1. BUY + decision-grade benchmark + all criteria OK -> BUY
2. BUY + CONTEXT_ONLY benchmark -> BUY_ON_CONFIRMATION
3. BUY + trading confidence below threshold -> BUY_ON_CONFIRMATION
4. BUY + match quality below GOOD -> BUY_ON_CONFIRMATION
5. BUY + fallback ratio above maximum -> BUY_ON_CONFIRMATION
6. BUY + historical probability below threshold -> BUY_ON_CONFIRMATION
7. BUY + reward/risk below threshold -> BUY_ON_CONFIRMATION
8. BUY + blocked regime -> BUY_ON_CONFIRMATION

This file does NOT modify production files.

Run from the project root:
    python -m app.strategies.test_decision_gate_rules

Or, if copied beside the module:
    python app/strategies/test_decision_gate_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Make direct execution from app/strategies work as well as module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from app.strategies.decision_gate_v1_2 import apply_decision_gate
except ImportError as exc:
    print("ERROR: Could not import Decision Gate V1.2.")
    print(f"       {exc}")
    print()
    print("Verify that this file exists:")
    print("  app/strategies/decision_gate_v1_2.py")
    raise SystemExit(2)


def base_buy_result() -> Dict[str, Any]:
    """
    Synthetic but realistic CCE-style bullish result.

    Field names match the current Decision Gate V1.2 input contract.
    Values intentionally satisfy every BUY gate.
    Individual tests modify only the criterion being tested.
    """
    return {
        "ticker": "TEST",
        "signal": "BUY",
        "action": "BUY",
        "score": 85.0,
        "confidence": 75.0,

        "benchmark": "QQQ",
        "benchmark_trust": "HIGH",
        "benchmark_trust_decision_grade": True,

        "trading_confidence": 75.0,

        "match_quality": "GOOD",
        "fallback_ratio": 0.10,
        "historical_probability_20d": 0.70,
        "reward_risk_20d": 2.00,
        "cycle": "CONSOLIDATION",

        # Include common nested structures so the test resembles
        # an actual orchestrator/CCE result.
    }


def extract_final_signal(result: Dict[str, Any]) -> str:
    """Extract the final decision signal from the Gate output."""
    for key in ("final_signal", "signal", "action", "decision"):
        value = result.get(key)
        if value is not None:
            return str(value).upper()
    return "UNKNOWN"


def extract_gate_details(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Gate diagnostic section when available."""
    for key in ("gate", "decision_gate", "checks", "gates"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def run_case(
    number: int,
    name: str,
    result: Dict[str, Any],
    expected: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Run one test case and print a compact diagnostic."""
    try:
        output = apply_decision_gate(result)
        actual = extract_final_signal(output)
        passed = actual == expected.upper()

        status = "PASS" if passed else "FAIL"

        print(f"TEST {number:02d} | {name}")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        print(f"Result:   {status}")

        # Print useful diagnostics if available.
        details = extract_gate_details(output)
        if details:
            failed = details.get("failed_gates")
            if failed:
                print(f"Failed gates: {failed}")

        print("-" * 78)

        return passed, output

    except Exception as exc:
        print(f"TEST {number:02d} | {name}")
        print(f"Expected: {expected}")
        print(f"Actual:   EXCEPTION")
        print(f"Result:   FAIL")
        print(f"Error:    {type(exc).__name__}: {exc}")
        print("-" * 78)
        return False, {"error": str(exc)}


def main() -> int:
    print("=" * 78)
    print("DECISION GATE V1.2 - RULE TEST")
    print("=" * 78)
    print("Synthetic controlled cases - no market data is used.")
    print("Production files are not modified.")
    print("-" * 78)

    cases: List[Tuple[str, Dict[str, Any], str]] = []

    # 1. All BUY gates pass.
    cases.append((
        "BUY + all criteria OK",
        base_buy_result(),
        "BUY",
    ))

    # 2. Benchmark trust blocks executable BUY.
    case = base_buy_result()
    case["benchmark_trust"] = "CONTEXT_ONLY"
    case["benchmark_trust_decision_grade"] = False
    cases.append((
        "BUY + CONTEXT_ONLY benchmark",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 3. Trading confidence below 60.
    case = base_buy_result()
    case["trading_confidence"] = 59.9
    case["confidence"] = 59.9
    cases.append((
        "BUY + trading confidence < 60",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 4. Match quality below GOOD.
    case = base_buy_result()
    case["match_quality"] = "MODERATE"
    cases.append((
        "BUY + match quality < GOOD",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 5. Fallback ratio above 25%.
    case = base_buy_result()
    case["fallback_ratio"] = 0.26
    cases.append((
        "BUY + fallback ratio > 25%",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 6. Historical probability below 60%.
    case = base_buy_result()
    case["historical_probability_20d"] = 0.59
    cases.append((
        "BUY + historical probability < 60%",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 7. Reward/risk below 1.25.
    case = base_buy_result()
    case["reward_risk_20d"] = 1.24
    cases.append((
        "BUY + reward/risk < 1.25",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    # 8. Blocked regime.
    case = base_buy_result()
    case["cycle"] = "DOWNTREND"
    cases.append((
        "BUY + blocked regime",
        case,
        "BUY_ON_CONFIRMATION",
    ))

    passed_count = 0
    failed_count = 0

    for index, (name, result, expected) in enumerate(cases, start=1):
        passed, _ = run_case(index, name, result, expected)
        if passed:
            passed_count += 1
        else:
            failed_count += 1

    total = len(cases)

    print("=" * 78)
    print(
        f"RESULT: {passed_count}/{total} PASS"
        if failed_count == 0
        else f"RESULT: {passed_count}/{total} PASS, {failed_count}/{total} FAIL"
    )

    if failed_count == 0:
        print("DECISION GATE V1.2 RULE TEST: PASS")
        print()
        print("All eight BUY decision rules behave as expected.")
        return 0

    print("DECISION GATE V1.2 RULE TEST: FAIL")
    print()
    print("At least one BUY decision rule did not produce the expected result.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
