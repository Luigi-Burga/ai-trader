import unittest
import sys
import types

# Stub production-boundary dependencies so these unit tests do not fetch market data.
def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module

_stub_module("app.ai.asset_analysis_orchestrator_v1_5_1", analyze_ticker=lambda *a, **k: {})
_stub_module("app.strategies.decision_gate_v1_5", apply_decision_gate=lambda *a, **k: {})
_stub_module("app.strategies.signal_engine_v1_2", generate_signal=lambda *a, **k: {})
from app.scanners.watchlist_scanner_v2_2_2 import _decision_policy_shadow

class ShadowIntegrationTests(unittest.TestCase):
    def fixture(self, signal="BUY_ON_CONFIRMATION", **overrides):
        analysis = {
            "ticker":"PLTR", "signal":signal, "status":"ANALYZED", "score":81.97,
            "confidence":0.5822, "regime":"RECOVERY", "current_price":198.73,
            "benchmark_trust":"MEDIUM", "benchmark_trust_decision_grade":True,
            "appt_x_20d":2.000206,
        }
        engine = {
            "ticker":"PLTR", "historical_probability_20d":0.8, "reward_risk_20d":2.557,
            "fallback_ratio":0.1, "match_quality":"MODERATE",
            "levels":{"entry_price":198.73,"stop":180.0},
        }
        analysis.update(overrides)
        return analysis, engine

    def test_confirmation_probe_reports_automatic_blockers_without_overriding(self):
        a,e=self.fixture()
        r=_decision_policy_shadow(a,e,"BUY_ON_CONFIRMATION")
        self.assertEqual(r["decision_status"],"REQUIRES_CONFIRMATION")
        self.assertIn("CONFIDENCE_BELOW_AUTO_MIN",r["failed_gates"])
        self.assertIn("MATCH_QUALITY_BELOW_AUTO_MIN",r["failed_gates"])
        self.assertEqual(r["eligibility_probe"],"HYPOTHETICAL_ONLY")
        self.assertTrue(r["policy_does_not_override_pipeline"])
        self.assertEqual(r["original_final_signal"],"BUY_ON_CONFIRMATION")

    def test_actual_buy_is_evaluated_as_actual_signal(self):
        a,e=self.fixture(signal="BUY", confidence=0.68)
        e["match_quality"]="GOOD"
        r=_decision_policy_shadow(a,e,"BUY")
        self.assertEqual(r["eligibility_probe"],"ACTUAL_FINAL_SIGNAL")
        self.assertEqual(r["decision_status"],"AUTO_ELIGIBLE")

    def test_watch_is_never_promoted(self):
        a,e=self.fixture(signal="WATCH")
        r=_decision_policy_shadow(a,e,"WATCH")
        self.assertEqual(r["decision_status"],"NOT_BUY_CANDIDATE")
        self.assertEqual(r["operational_signal"],"WATCH")

if __name__ == "__main__": unittest.main()
