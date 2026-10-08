from pathlib import Path
import ast, py_compile

TARGET = Path(__file__).with_name("performance_diagnostic_v2_3_1.py")

def test_exists_and_compiles():
    assert TARGET.exists()
    py_compile.compile(str(TARGET), doraise=True)

def test_version_and_safety():
    s = TARGET.read_text(encoding="utf-8")
    assert 'VERSION = "2.3.1"' in s
    assert "RESEARCH_ONLY = True" in s
    assert "PRODUCTION_CHANGES = False" in s
    assert "ORDERS_ALLOWED = False" in s

def test_no_trading_endpoints():
    s = TARGET.read_text(encoding="utf-8").lower()
    for token in ("submit_order(", "close_position(", "cancel_order(",
                  "tradingclient(", "orders.submit"):
        assert token not in s

def test_safe_instrumentation_contract():
    s = TARGET.read_text(encoding="utf-8")
    # V2.3.1 must not replace the internal cache helpers.
    assert "md._request_key =" not in s
    assert "md._cache_path =" not in s
    assert "md._read_cache =" not in s
    assert "md._write_cache =" not in s
    # Public boundaries and Yahoo remain instrumented.
    assert "md.get_history" in s
    assert "md.get_daily_history" in s
    assert "md.yf.download" in s

def test_lineage_contract():
    s = TARGET.read_text(encoding="utf-8")
    for token in (
        "request_key", "cache_path", "cache_exists",
        "cache_age_seconds", "public_failures",
        "benchmark_request_lineage", "all_public_failures",
        "traceback", "_cache_path",
    ):
        assert token in s

def test_ast():
    ast.parse(TARGET.read_text(encoding="utf-8"))

if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
            print("PASS:", name)
    print("PERFORMANCE DIAGNOSTIC V2.3.1 CONTRACT TEST: PASS")
