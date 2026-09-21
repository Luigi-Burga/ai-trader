"""Structural tests; no Telegram/network calls."""
from __future__ import annotations
import os, unittest
from unittest.mock import patch

class TelegramAnalysisV1Tests(unittest.TestCase):
    def test_formatter_core_fields(self):
        from app.telegram.formatter import format_analysis
        r={"ticker":"GDXU","final_signal":"WATCH","current_price":95.12,"score":61.5,"confidence":58.2,"regime":"BASE","benchmark":"GDX","benchmark_trust":"HIGH","route":"MATURE","engine":"CHARACTERISTIC_CURVE_ENGINE","engine_version":"1.6.1","decision_gate":{"gate_status":"NOT_APPLICABLE","gate_reason":"non_bullish_signal_preserved"},"engine_result":{"levels":{"entry_zone_low":91,"entry_zone_high":94,"dynamic_stop":87,"take_profit_1":101}}}
        m=format_analysis(r)
        for s in ["GDXU","WATCH","GDX","HIGH","91.00","94.00"]: self.assertIn(s,m)

    def test_service_no_fixed_target(self):
        import app.services.analysis_service as service
        with patch.object(service,"scan_buy_opportunity",return_value={"ticker":"GDXU","final_signal":"WATCH"}) as scanner:
            service.analyze_ticker("gdxu")
        self.assertEqual(scanner.call_args.args[0]["buy_target"],0.0)

    def test_allow_list(self):
        from app.telegram.handlers import _allowed_user_ids
        with patch.dict(os.environ,{"TELEGRAM_ALLOWED_USER_IDS":"123,456"},clear=False): self.assertEqual(_allowed_user_ids(),{123,456})

if __name__ == "__main__": unittest.main(verbosity=2)
