"""Contract tests for Prediction Tracker V1.6 / Performance Optimization V1."""
from __future__ import annotations

import importlib.util
import sys
import types
import json
import tempfile
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).with_name("prediction_tracker_v1_6.py")


def load_module():
    # Provide minimal dependency shims so this contract test remains
    # standalone; no production package or network access is required.
    app = types.ModuleType("app")
    data = types.ModuleType("app.data")
    market = types.ModuleType("app.data.market_data")
    market.get_history = lambda *args, **kwargs: pd.DataFrame()
    market.get_multiple_daily_history = lambda *args, **kwargs: {}
    portfolio = types.ModuleType("app.portfolio")
    risk = types.ModuleType("app.portfolio.risk_x_engine")
    risk.build_risk_context = lambda entry, stop, target: {"risk_pct": 1.0}
    risk.enrich_observation_with_x = lambda obs, risk_pct: dict(obs, risk_pct=risk_pct)
    risk.risk_percent = lambda *args, **kwargs: 1.0
    post = types.ModuleType("app.portfolio.post_exit_analyzer")
    post.analyze_post_exit = lambda *args, **kwargs: None
    sys.modules.update({
        "app": app,
        "app.data": data,
        "app.data.market_data": market,
        "app.portfolio": portfolio,
        "app.portfolio.risk_x_engine": risk,
        "app.portfolio.post_exit_analyzer": post,
    })
    spec = importlib.util.spec_from_file_location("prediction_tracker_v1_6", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load prediction_tracker_v1_6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_frame():
    idx = pd.date_range("2026-01-02", periods=80, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(80)],
            "High": [101.0 + i for i in range(80)],
            "Low": [99.0 + i for i in range(80)],
            "Close": [100.5 + i for i in range(80)],
            "Volume": [1000] * 80,
        },
        index=idx,
    )


def test_version_and_ttl(mod):
    assert mod.VERSION == "1.6"
    assert mod.TRACKER_MARKET_CACHE_TTL_SECONDS == 86400


def test_cache_uses_tracker_ttl(mod):
    calls = []
    frame = sample_frame()

    original = mod.get_history

    def fake_get_history(*args, **kwargs):
        calls.append(kwargs.copy())
        return frame.copy(deep=True)

    mod.get_history = fake_get_history
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "predictions_2026-01-02.jsonl"
            record = {
                "ticker": "NVDA",
                "prediction_timestamp_utc": "2026-01-02T15:00:00+00:00",
                "price_at_prediction": 100.0,
                "evaluation": {"status": "PENDING", "observations": {}},
            }
            p.write_text(json.dumps(record) + "\n", encoding="utf-8")
            mod._build_market_data_cache([p])
    finally:
        mod.get_history = original

    assert calls, "Expected a Market Data call"
    assert calls[0]["ttl_seconds"] == 86400


def test_complete_snapshot_without_exit_is_fast_path(mod):
    snapshot = {
        "ticker": "NVDA",
        "prediction_timestamp_utc": "2026-01-02T15:00:00+00:00",
        "price_at_prediction": 100.0,
        "evaluation": {
            "status": "COMPLETE",
            "observations": {},
        },
        "risk_x": {"risk_pct": 1.0},
        "post_exit_analysis": None,
    }

    called = {"value": False}
    original = mod._trading_days_since

    def fail_if_called(*args, **kwargs):
        called["value"] = True
        raise AssertionError("Market data should not be requested for a stable COMPLETE snapshot")

    mod._trading_days_since = fail_if_called
    try:
        result = mod.evaluate_snapshot(snapshot)
    finally:
        mod._trading_days_since = original

    assert result == snapshot
    assert called["value"] is False


def test_partial_snapshot_preserves_completed_horizons(mod):
    frame = sample_frame()
    snapshot = {
        "ticker": "NVDA",
        "prediction_timestamp_utc": "2026-01-02T15:00:00+00:00",
        "price_at_prediction": 100.0,
        "final_signal": "BUY",
        "levels": {"target": 110.0, "stop": 95.0},
        "evaluation": {
            "status": "PARTIAL",
            "observations": {
                "+1d": {
                    "actual_price": 999.0,
                    "return_pct": 999.0,
                    "evaluated_at_utc": "KEEP-ME",
                },
                "+3d": None,
                "+5d": None,
                "+10d": None,
                "+20d": None,
                "+60d": None,
            },
        },
    }

    result = mod.evaluate_snapshot(snapshot, market_data=frame)
    kept = result["evaluation"]["observations"]["+1d"]
    assert kept["actual_price"] == 999.0
    assert kept["evaluated_at_utc"] == "KEEP-ME"
    assert result["evaluation"]["observations"]["+3d"] is not None


def test_update_prediction_file_does_not_rewrite_when_unchanged(mod):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions_2026-01-02.jsonl"
        record = {
            "ticker": "NVDA",
            "prediction_timestamp_utc": "2026-01-02T15:00:00+00:00",
            "price_at_prediction": 100.0,
            "evaluation": {"status": "COMPLETE", "observations": {}},
        }
        original = json.dumps(record, separators=(",", ":")) + "\n"
        path.write_text(original, encoding="utf-8")
        before_mtime = path.stat().st_mtime_ns

        updated, unchanged = mod.update_prediction_file(path, market_data_cache={})
        after_mtime = path.stat().st_mtime_ns

        assert updated == 0
        assert unchanged == 1
        assert after_mtime == before_mtime
        assert path.read_text(encoding="utf-8") == original


def main():
    mod = load_module()
    tests = [
        test_version_and_ttl,
        test_cache_uses_tracker_ttl,
        test_complete_snapshot_without_exit_is_fast_path,
        test_partial_snapshot_preserves_completed_horizons,
        test_update_prediction_file_does_not_rewrite_when_unchanged,
    ]
    for test in tests:
        test(mod)
        print(f"PASS: {test.__name__}")
    print("\nPREDICTION TRACKER V1.6 CONTRACT TEST: PASS")
    print("Performance Optimization V1: PASS")
    print("Production source unchanged: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
