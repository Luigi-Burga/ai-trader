#!/usr/bin/env python3
"""Contract test for Performance Diagnostic V2.4."""

from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("performance_diagnostic_v2_4.py")


def test_exists_and_compiles():
    source = TARGET.read_text(encoding="utf-8")
    ast.parse(source)
    compile(source, str(TARGET), "exec")


def test_research_only_scope():
    s = TARGET.read_text(encoding="utf-8").lower()
    assert "research-only" in s
    assert "production source files" in s
    assert "does not submit orders" in s


def test_no_trading_endpoints():
    s = TARGET.read_text(encoding="utf-8").lower()
    forbidden = [
        "submit_order(",
        "cancel_order(",
        "close_position(",
        "tradingclient(",
        "alpaca.trading",
    ]
    for token in forbidden:
        assert token not in s


def test_no_internal_cache_replacement():
    s = TARGET.read_text(encoding="utf-8")
    # These helpers may be called for shadow inspection, but must not be
    # assigned/replaced.
    assert "md._request_key =" not in s
    assert "md._cache_path =" not in s
    assert "md._read_cache =" not in s
    assert "md._write_cache =" not in s
    assert "yf.download =" not in s
    assert "MD._request_key =" not in s
    assert "MD._cache_path =" not in s
    assert "MD._read_cache =" not in s
    assert "MD._write_cache =" not in s


def test_real_flow_and_spy_scope():
    s = TARGET.read_text(encoding="utf-8")
    assert "production_main.main()" in s
    assert 'TARGET_TICKER = "SPY"' in s
    assert 'ticker != TARGET_TICKER' in s
    assert "MD.get_history = traced_get_history" in s
    assert "MD.get_daily_history = traced_get_daily_history" in s


def test_lineage_fields():
    s = TARGET.read_text(encoding="utf-8")
    for token in (
        "request_key",
        "cache_dir",
        "cache_path",
        "ttl_seconds",
        "cache_exists",
        "cache_age_seconds",
        "cache_fresh",
        "caller",
        "diagnostic_cache_classification",
    ):
        assert token in s


def test_safety_environment():
    s = TARGET.read_text(encoding="utf-8")
    assert 'AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"' in s
    assert 'AI_TRADER_LIVE_TRADING"] = "0"' in s



def test_v24_classification_and_metrics():
    s = TARGET.read_text(encoding="utf-8")
    for token in (
        "FRESH_CACHE_ELIGIBLE",
        "FORCE_REFRESH",
        "MISS_OR_STALE",
        "repeated_request_key_count",
        "caller_groups",
        "cache_mtime_changed",
    ):
        assert token in s


def test_v24_output_name():
    s = TARGET.read_text(encoding="utf-8")
    assert "performance_diagnostic_v2_4_results.json" in s

def main():
    tests = [
        test_exists_and_compiles,
        test_research_only_scope,
        test_no_trading_endpoints,
        test_no_internal_cache_replacement,
        test_real_flow_and_spy_scope,
        test_lineage_fields,
        test_safety_environment,
        test_v24_classification_and_metrics,
        test_v24_output_name,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
    print("PERFORMANCE DIAGNOSTIC V2.4 CONTRACT TEST: PASS")


if __name__ == "__main__":
    main()
