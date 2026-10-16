import importlib.util
import unittest
from pathlib import Path
MODULE_PATH = Path(__file__).with_name("decision_audit_v1_3.py")
spec = importlib.util.spec_from_file_location("decision_audit_v1_3", MODULE_PATH)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)

class DecisionAuditV13Tests(unittest.TestCase):
    def test_extreme_return_is_flagged_with_source_details(self):
        rows = [{"symbol":"AAA","timestamp":None,"raw_signal":"BUY","final_signal":"BUY_ON_CONFIRMATION","price":10,"benchmark":"SPY","engine":"TEST","regime":"BULLISH","reason":None,"source_file":"p.jsonl","source_line":3,"outcomes":{"1d":{"return_pct":5,"future_close":10.5}}}, {"symbol":"BBB","timestamp":None,"raw_signal":"BUY","final_signal":"BUY","price":1,"benchmark":"SPY","engine":"TEST","regime":"BULLISH","reason":None,"source_file":"p.jsonl","source_line":9,"outcomes":{"1d":{"return_pct":16212,"future_close":163.12}}}]
        result=audit.audit_outliers(rows,[1]); self.assertEqual(result["by_horizon"]["1d"]["flagged_N"],1); self.assertEqual(result["flagged_observations"][0]["source_line"],9); self.assertIn("ABS_RETURN_THRESHOLD",result["flagged_observations"][0]["flag_reasons"])
    def test_flagging_does_not_mutate_or_remove_data(self):
        rows=[{"symbol":"AAA","raw_signal":"BUY","final_signal":"BUY","outcomes":{"1d":{"return_pct":250.0}}}]; audit.audit_outliers(rows,[1]); self.assertEqual(rows[0]["outcomes"]["1d"]["return_pct"],250.0)
    def test_buy_audit_includes_reason_status_source_and_horizons(self):
        rows=[{"symbol":"AAA","timestamp":None,"raw_signal":"BUY","gated_signal":"BUY_ON_CONFIRMATION","final_signal":"BUY_ON_CONFIRMATION","score":90,"confidence":75,"appt":None,"price":10,"reason":None,"benchmark":"SPY","regime":"BULLISH","engine":"TEST","source_file":"p.jsonl","source_line":12,"outcomes":{"1d":{"future_close":9,"return_pct":-10,"benchmark_return_pct":1,"excess_return_pct":-11}}}]
        result=audit.buy_signal_audit(rows,[1,5]); record=result["records"][0]; self.assertEqual(result["raw_buy_records_N"],1); self.assertEqual(record["reason_status"],"UNAVAILABLE_IN_SOURCE_RECORD"); self.assertEqual(record["source_line"],12); self.assertEqual(record["outcomes"]["1d"]["return_pct"],-10); self.assertIn("5d",record["outcomes"]); self.assertEqual(result["reason_unavailable_N"],1)
    def test_available_reason_and_classification_are_counted(self):
        rows=[{"symbol":"AAA","raw_signal":"BUY","final_signal":"BUY","gated_signal":"BUY","reason":"risk gate passed","outcomes":{}}]; result=audit.buy_signal_audit(rows,[1]); self.assertEqual(result["classification_counts"]["BUY_EXECUTABLE"],1); self.assertEqual(result["reason_counts"][0]["reason"],"risk gate passed"); self.assertEqual(result["reason_unavailable_N"],0)
    def test_empty_outlier_report_is_explicit(self):
        result=audit.audit_outliers([],[1,5]); self.assertEqual(result["flagged_observations_N"],0); self.assertEqual(result["by_horizon"]["1d"]["evaluated_N"],0); self.assertIsNone(result["by_horizon"]["1d"]["iqr_upper_fence_pct"])
if __name__ == "__main__": unittest.main(verbosity=2)
