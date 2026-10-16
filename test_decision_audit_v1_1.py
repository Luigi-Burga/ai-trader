import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).with_name('decision_audit_v1_1.py')
spec = importlib.util.spec_from_file_location('decision_audit_v1_1', MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DecisionAuditV11Tests(unittest.TestCase):
    def test_prediction_timestamp_utc_alias_is_recognized(self):
        row = audit.normalize({
            'ticker': 'SPY',
            'prediction_timestamp_utc': '2026-10-01T15:30:00Z',
            'price_at_prediction': 100,
            '_audit_file': 'fixture.jsonl',
            '_audit_line': 1,
        })
        self.assertEqual(row['timestamp'], datetime(2026, 10, 1, 15, 30, tzinfo=timezone.utc))

    def test_horizon_counts_trading_sessions_not_calendar_days(self):
        series = pd.Series(
            [100.0, 101.0, 102.0, 103.0],
            index=pd.to_datetime(['2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07']),
        )
        when = datetime(2026, 10, 2, tzinfo=timezone.utc)
        self.assertEqual(audit.future_close(series, when, 1), 101.0)
        self.assertEqual(audit.future_close(series, when, 3), 103.0)
        self.assertIsNone(audit.future_close(series, when, 4))

    def test_benchmark_excess_return_is_calculated_on_same_horizon(self):
        idx = pd.to_datetime(['2026-10-01', '2026-10-02', '2026-10-05'])
        asset = pd.DataFrame({'Close': [100.0, 105.0, 110.0]}, index=idx)
        benchmark = pd.DataFrame({'Close': [200.0, 202.0, 204.0]}, index=idx)
        row = {
            'symbol': 'AAA', 'benchmark': 'SPY',
            'timestamp': datetime(2026, 10, 1, tzinfo=timezone.utc),
            'price': 100.0, 'outcomes': {},
        }
        audit.add_outcomes([row], {'AAA': asset, 'SPY': benchmark}, [1])
        outcome = row['outcomes']['1d']
        self.assertAlmostEqual(outcome['return_pct'], 5.0)
        self.assertAlmostEqual(outcome['benchmark_return_pct'], 1.0)
        self.assertAlmostEqual(outcome['excess_return_pct'], 4.0)

    def test_confirmation_rate_uses_raw_buy_to_confirmation_group(self):
        rows = [
            {'raw_signal': 'BUY', 'final_signal': 'BUY_ON_CONFIRMATION', 'confidence': None, 'score': None, 'appt': None, 'outcomes': {}},
            {'raw_signal': 'BUY', 'final_signal': 'BUY', 'confidence': None, 'score': None, 'appt': None, 'outcomes': {}},
            {'raw_signal': 'HOLD', 'final_signal': 'BUY_ON_CONFIRMATION', 'confidence': None, 'score': None, 'appt': None, 'outcomes': {}},
        ]
        summary = audit.aggregate(rows, [1])
        self.assertEqual(summary['raw_buy']['N'], 2)
        self.assertEqual(summary['raw_buy_to_confirmation']['N'], 1)
        self.assertEqual(summary['buy_on_confirmation']['N'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
