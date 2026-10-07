"""
Contract test for performance_diagnostic_v2.py.

Research-only. Does not execute production main.py, contact Alpaca, or place
orders. Validates that the diagnostic remains observational and that its
instrumentation targets the current production architecture.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "performance_diagnostic_v2.py"


def source() -> str:
    return TARGET.read_text(encoding="utf-8")


def test_exists_and_compiles() -> None:
    assert TARGET.exists(), f"Missing target: {TARGET}"
    py_compile.compile(str(TARGET), doraise=True)


def test_version_and_research_only_contract() -> None:
    tree = ast.parse(source())
    text = source()

    assert 'VERSION = "2.0"' in text
    assert "research_only" in text
    assert "production_changes" in text
    assert "orders_allowed" in text
    assert "AI_TRADER_AUTONOMOUS_EXECUTION" in text
    assert "DiagnosticSafetyError" in text

    # Must not expose or invoke an order endpoint from this diagnostic.
    forbidden = [
        "submit_order",
        "submit_order_request",
        "TradingClient.submit_order",
        "MarketOrderRequest",
        "LimitOrderRequest",
    ]
    for token in forbidden:
        assert token not in text, f"Forbidden trading token found: {token}"


def test_does_not_edit_production_source() -> None:
    text = source()

    assert "PRODUCTION_MAIN = PROJECT_ROOT / \"app\" / \"main.py\"" in text
    assert "read_text(encoding=\"utf-8\")" in text
    assert "write_text(" in text

    # write_text is permitted only for the diagnostic JSON output. The
    # production source itself must never be opened for writing.
    assert "PRODUCTION_MAIN.write_text" not in text
    assert "PRODUCTION_MAIN.unlink" not in text
    assert "PRODUCTION_MAIN.rename" not in text


def test_runtime_instrumentation_contract() -> None:
    text = source()

    required = [
        "app.main",
        "_run_prediction_tracker",
        "monitor_position",
        "scan_buy_opportunity",
        "_execution_sizing",
        "calculate_fundamental_score",
        "process_fundamental_alert",
        "app.data.market_data",
        "get_history",
        "_read_cache",
        "yfinance_download_calls",
        "app.prediction_tracker",
        "_build_market_data_cache",
        "app.ai.asset_analysis_orchestrator_v1_5_1",
        "analyze_ticker",
    ]

    for token in required:
        assert token in text, f"Missing instrumentation contract: {token}"


def test_cache_hit_miss_contract() -> None:
    text = source()

    assert '"cache_read_calls"' in text
    assert '"cache_hits"' in text
    assert '"cache_misses"' in text
    assert "result = original_read(*args, **kwargs)" in text
    assert "if result is None:" in text


def test_yfinance_boundary_contract() -> None:
    text = source()

    assert "yf.download" in text
    assert "yfinance_download_seconds" in text
    assert "yfinance_download_errors" in text


def test_ticker_breakdown_contract() -> None:
    text = source()

    assert "by_ticker" in text
    assert "_ticker_from_args" in text
    assert "TOP TICKERS BY INSTRUMENTED TIME" in text


def test_no_cprofile_dependency() -> None:
    text = source()
    # The diagnostic intentionally does not import or execute cProfile/pstats.
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "cProfile" not in imported
    assert "pstats" not in imported


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
    print()
    print("PERFORMANCE DIAGNOSTIC V2 CONTRACT TEST: PASS")
    print("Research-only guards: PASS")
    print("Production source unchanged contract: PASS")
    print("Stage instrumentation contract: PASS")
    print("Market-data cache/download contract: PASS")
