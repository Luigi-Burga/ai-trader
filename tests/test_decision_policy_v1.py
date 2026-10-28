import unittest
from app.ai.decision_policy_v1 import DecisionInput, DecisionPolicyConfig, evaluate_decision


def candidate(**overrides):
    values = dict(
        symbol="TEST", analytical_signal="BUY", score=74,
        probability_20d=0.63, reward_risk=1.4, confidence=0.68,
        fallback_ratio=0.10, match_quality="GOOD", regime="CONFIRMED_BULL",
        appt=0.2, benchmark_trust="HIGH", benchmark_decision_grade=True,
        data_valid=True, entry_valid=True, stop_valid=True,
    )
    values.update(overrides)
    return DecisionInput(**values)


class DecisionPolicyTests(unittest.TestCase):
    def test_primary_buy(self):
        result = evaluate_decision(candidate())
        self.assertEqual(result.operational_signal, "BUY")
        self.assertEqual(result.decision_status, "AUTO_ELIGIBLE")

    def test_confirmation_is_not_auto_buy(self):
        result = evaluate_decision(candidate(
            confidence=0.54, probability_20d=0.67, reward_risk=1.6,
            fallback_ratio=0.40, match_quality="MODERATE", appt=0.6
        ))
        self.assertEqual(result.operational_signal, "BUY_ON_CONFIRMATION")
        self.assertEqual(result.decision_status, "REQUIRES_CONFIRMATION")
        self.assertIn("CONFIDENCE_BELOW_AUTO_MIN", result.failed_gates)

    def test_fallback_over_confirmation_limit_blocks(self):
        result = evaluate_decision(candidate(fallback_ratio=0.70))
        self.assertEqual(result.operational_signal, "WATCH")
        self.assertIn("FALLBACK_ABOVE_AUTO_MAX", result.failed_gates)

    def test_bad_benchmark_blocks(self):
        result = evaluate_decision(candidate(benchmark_trust="CONTEXT_ONLY"))
        self.assertEqual(result.operational_signal, "WATCH")
        self.assertIn("BENCHMARK_NOT_DECISION_GRADE", result.failed_gates)

    def test_non_buy_is_not_promoted(self):
        result = evaluate_decision(candidate(analytical_signal="WATCH"))
        self.assertEqual(result.operational_signal, "WATCH")
        self.assertEqual(result.decision_status, "NOT_BUY_CANDIDATE")

    def test_blocked_regime_does_not_auto_buy(self):
        result = evaluate_decision(candidate(regime="DOWNTREND"))
        self.assertEqual(result.operational_signal, "WATCH")
        self.assertIn("BLOCKED_REGIME", result.failed_gates)


if __name__ == "__main__":
    unittest.main()
