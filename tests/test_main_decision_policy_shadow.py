import unittest
from app.decision_policy_shadow_integration import _decision_policy_shadow
from pathlib import Path
MAIN_SOURCE = (Path(__file__).parents[1] / "app" / "main.py").read_text(encoding="utf-8")


class MainDecisionPolicyShadowTests(unittest.TestCase):
    def make_result(self, signal="BUY_ON_CONFIRMATION", confidence=54.09):
        return {
            "ticker": "PLTR", "status": "ANALYZED", "final_signal": signal,
            "signal": signal, "score": 80.68, "confidence": confidence,
            "current_price": 198.73, "regime": "CONFIRMED_BULL",
            "benchmark": "IGV", "benchmark_trust": "MEDIUM",
            "benchmark_trust_decision_grade": True, "appt_x_20d": 1.489448,
            "levels": {"entry_price": 198.73, "stop": 166.2105},
            "engine_result": {"historical_probability_20d": 0.75,
                "reward_risk_20d": 2.13, "fallback_ratio": 0.20,
                "match_quality": "GOOD", "cycle": "CONFIRMED_BULL",
                "appt_x_20d": 1.489448,
                "levels": {"entry_price": 198.73, "stop": 166.2105}},
        }

    def test_confirmation_signal_is_preserved_and_confidence_normalized(self):
        result = self.make_result()
        decision = _decision_policy_shadow("PLTR", result)
        self.assertEqual(result["final_signal"], "BUY_ON_CONFIRMATION")
        self.assertEqual(decision["original_signal"], "BUY_ON_CONFIRMATION")
        self.assertEqual(decision["eligibility_probe"], "HYPOTHETICAL_ONLY")
        self.assertAlmostEqual(decision["metrics"]["confidence"], 0.5409)
        self.assertEqual(decision["mode"], "SHADOW_ONLY")
        self.assertFalse(decision["orders_submitted"])

    def test_non_buy_signal_is_not_promoted(self):
        result = self.make_result(signal="WATCH", confidence=54.09)
        decision = _decision_policy_shadow("PLTR", result)
        self.assertEqual(result["final_signal"], "WATCH")
        self.assertEqual(decision["operational_signal"], "WATCH")
        self.assertEqual(decision["decision_status"], "NOT_BUY_CANDIDATE")

    def test_execution_interlock_is_disabled(self):
        self.assertIn("DECISION_POLICY_SHADOW_ONLY = False", MAIN_SOURCE)
        self.assertIn("NO EXECUTION: confirmation required.", MAIN_SOURCE)
        self.assertIn("risk_gate.evaluate(", MAIN_SOURCE)
        self.assertIn("executor.execute(", MAIN_SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
