import unittest
from decision_policy_shadow_v1 import map_scanner_result, summarize


def complete_watch_result():
    return {
        "ticker": "XYZ", "signal": "WATCH", "gated_signal": "WATCH",
        "final_signal": "WATCH", "status": "ANALYZED",
        "analysis": {
            "signal": "WATCH", "score": 75, "historical_probability_20d": .7,
            "reward_risk_20d": 1.6, "trading_confidence": 68,
            "fallback_ratio": .1, "match_quality": "GOOD",
            "cycle": "CONFIRMED_BULL", "appt_x_20d": .6,
            "benchmark_trust": "HIGH", "benchmark_trust_decision_grade": True,
            "current_price": 100,
            "levels": {"current_price":100,"entry_price":100,
                       "entry_zone_low":95,"entry_zone_high":105,"stop":90}
        }
    }


class ShadowAdapterTests(unittest.TestCase):
    def test_maps_percentage_confidence_to_fraction(self):
        mapped, meta = map_scanner_result(complete_watch_result())
        self.assertAlmostEqual(mapped.confidence, .68)
        self.assertTrue(mapped.entry_valid)
        self.assertTrue(mapped.stop_valid)
        self.assertEqual(meta["missing_required_metrics"], [])

    def test_maps_metrics_from_nested_engine_result(self):
        result = {
            "ticker": "XYZ", "signal": "BUY", "final_signal": "BUY", "status": "ANALYZED",
            "analysis": {"signal": "BUY", "engine": "CCE"},
            "engine_result": {
                "score": 76, "historical_probability_20d": .71,
                "reward_risk_20d": 1.7, "trading_confidence": 66,
                "fallback_ratio": .1, "match_quality": "GOOD",
                "cycle": "CONFIRMED_BULL", "appt_x_20d": .7,
                "benchmark_trust": "HIGH", "benchmark_trust_decision_grade": True,
                "levels": {"current_price":100,"entry_price":100,
                           "entry_zone_low":95,"entry_zone_high":105,"stop":90}
            }
        }
        mapped, meta = map_scanner_result(result)
        self.assertEqual(mapped.score, 76)
        self.assertAlmostEqual(mapped.probability_20d, .71)
        self.assertEqual(meta["missing_required_metrics"], [])
        self.assertIn("engine_result.historical_probability_20d", meta["field_paths"]["probability_20d"])

    def test_missing_levels_not_marked_valid(self):
        result={"ticker":"XYZ","signal":"BUY","final_signal":"BUY","status":"ANALYZED",
                "analysis":{"signal":"BUY","benchmark_trust":"HIGH"}}
        mapped, meta=map_scanner_result(result)
        self.assertFalse(mapped.entry_valid)
        self.assertFalse(mapped.stop_valid)
        self.assertTrue(meta["missing_required_metrics"])

    def test_summary_marks_missing_metrics_even_for_watch(self):
        result={"ticker":"XYZ","signal":"WATCH","final_signal":"WATCH","status":"ANALYZED",
                "analysis":{"signal":"WATCH"}}
        row=summarize(result)
        self.assertEqual(row["comparison"], "DATOS_INSUFICIENTES")
        self.assertEqual(row["audit_status"], "INCOMPLETE_METRICS")
        self.assertIn("probability_20d", row["missing_required_metrics"])

    def test_watch_still_runs_hypothetical_gates_without_promoting_actual_signal(self):
        row=summarize(complete_watch_result())
        self.assertEqual(row["analytical_signal_preserved"], "WATCH")
        self.assertEqual(row["eligibility_probe"], "HYPOTHETICAL_ONLY")
        self.assertEqual(row["proposed_signal"], "BUY")
        self.assertEqual(row["comparison"], "CAMBIO_DE_SEÑAL")

    def test_below_score_watch_has_gate_failure_reason(self):
        result=complete_watch_result()
        result["analysis"]["score"]=68
        row=summarize(result)
        self.assertEqual(row["analytical_signal_preserved"], "WATCH")
        self.assertEqual(row["proposed_signal"], "WATCH")
        self.assertIn("SCORE_BELOW_BUY_MIN", row["failed_gates"])


if __name__=="__main__":
    unittest.main()
