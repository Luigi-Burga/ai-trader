import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name('decision_audit_v1_2.py')
spec = importlib.util.spec_from_file_location('decision_audit_v1_2', MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DecisionAuditV12Tests(unittest.TestCase):
    def test_percentile_linear_interpolation(self):
        self.assertAlmostEqual(audit.percentile([0.0, 10.0, 20.0, 30.0], 0.5), 15.0)

    def test_robust_stats_reports_mean_median_and_outlier_resistant_means(self):
        stats = audit.robust_stats([-2.0, -1.0, 0.0, 1.0, 1000.0])
        self.assertEqual(stats['N'], 5)
        self.assertAlmostEqual(stats['median'], 0.0)
        self.assertAlmostEqual(stats['mean'], 199.6)
        self.assertIsNotNone(stats['trimmed_mean_5pct'])
        self.assertIsNotNone(stats['winsorized_mean_5pct'])
        self.assertLess(stats['winsorized_mean_5pct'], stats['mean'])

    def test_empty_robust_stats_are_explicit_nulls(self):
        stats = audit.robust_stats([])
        self.assertEqual(stats['N'], 0)
        self.assertIsNone(stats['median'])
        self.assertIsNone(stats['mean'])

    def test_group_stats_includes_unique_symbols_and_robust_horizon_metrics(self):
        rows = [
            {'symbol': 'AAA', 'confidence': 70, 'score': 80, 'appt': None,
             'outcomes': {'1d': {'return_pct': 10.0, 'positive': True, 'benchmark_return_pct': 2.0, 'excess_return_pct': 8.0}}},
            {'symbol': 'AAA', 'confidence': 60, 'score': 70, 'appt': None,
             'outcomes': {'1d': {'return_pct': -4.0, 'positive': False, 'benchmark_return_pct': 1.0, 'excess_return_pct': -5.0}}},
            {'symbol': 'BBB', 'confidence': 50, 'score': 60, 'appt': None,
             'outcomes': {'1d': {'return_pct': 2.0, 'positive': True, 'benchmark_return_pct': 0.0, 'excess_return_pct': 2.0}}},
        ]
        stats = audit.group_stats(rows, [1])
        self.assertEqual(stats['N'], 3)
        self.assertEqual(stats['unique_symbols_N'], 2)
        self.assertEqual(stats['1d_N'], 3)
        self.assertAlmostEqual(stats['1d_median_return_pct'], 2.0)
        self.assertIsNotNone(stats['1d_winsorized_mean_5pct_return_pct'])
        self.assertEqual(stats['1d_excess_N'], 3)

    def test_grouped_analysis_handles_missing_categories(self):
        rows = [
            {'engine': 'ENGINE_A', 'symbol': 'AAA', 'confidence': None, 'score': None, 'appt': None, 'outcomes': {}},
            {'engine': None, 'symbol': 'BBB', 'confidence': None, 'score': None, 'appt': None, 'outcomes': {}},
        ]
        grouped = audit.grouped_analysis(rows, 'engine', [1])
        self.assertIn('ENGINE_A', grouped)
        self.assertIn('<UNKNOWN>', grouped)
        self.assertEqual(grouped['ENGINE_A']['N'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
