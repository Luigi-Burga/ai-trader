"""Contract test for performance_diagnostic_v2_1.py. Research-only."""
from __future__ import annotations
import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "performance_diagnostic_v2_1.py"

def source() -> str:
    return TARGET.read_text(encoding="utf-8")

def test_exists_and_compiles():
    assert TARGET.exists()
    py_compile.compile(str(TARGET), doraise=True)

def test_version_and_safety():
    text = source()
    assert 'VERSION = "2.1"' in text
    assert "research_only" in text
    assert "production_changes" in text
    assert "orders_allowed" in text
    assert "AI_TRADER_AUTONOMOUS_EXECUTION" in text
    assert "DiagnosticSafetyError" in text

def test_no_trading_endpoint():
    text = source()
    for token in ["submit_order", "submit_order_request", "TradingClient.submit_order",
                  "MarketOrderRequest", "LimitOrderRequest"]:
        assert token not in text

def test_production_read_only():
    text = source()
    assert 'PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"' in text
    assert "PRODUCTION_MAIN.read_text" in text
    assert "PRODUCTION_MAIN.write_text" not in text
    assert "PRODUCTION_MAIN.unlink" not in text
    assert "PRODUCTION_MAIN.rename" not in text

def test_watchlist_instrumentation():
    text = source()
    for token in [
        "watchlist_scanner_v2_2_2", "scan_buy_opportunity",
        "asset_analysis_orchestrator_v1_5_1", "analyze_ticker",
        "app.data.market_data", "_read_cache", "yf.download",
        "auto_adjust", "duplicate_yfinance_signatures",
        "market_requests", "yfinance_requests", "PER-TICKER WATCHLIST BREAKDOWN",
    ]:
        assert token in text

def test_no_cprofile():
    tree = ast.parse(source())
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "cProfile" not in imports
    assert "pstats" not in imports

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
    print("PERFORMANCE DIAGNOSTIC V2.1 CONTRACT TEST: PASS")
