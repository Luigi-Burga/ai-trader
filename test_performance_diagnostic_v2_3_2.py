from pathlib import Path
import ast

TARGET = Path("/app/performance_diagnostic_v2_3_2.py")


def test_exists_and_compiles():
    s = TARGET.read_text(encoding="utf-8")
    ast.parse(s, filename=str(TARGET))


def test_minimal_scope():
    s = TARGET.read_text(encoding="utf-8")
    assert "def _run_one(" in s
    assert "r1 = _run_one(" in s
    assert "r2 = _run_one(" in s
    assert "get_history(" in s
    assert "yfinance" not in s.lower() or "production" in s.lower()


def test_no_production_replacement():
    s = TARGET.read_text(encoding="utf-8")
    for token in (
        "md._request_key =",
        "md._cache_path =",
        "md._read_cache =",
        "md._write_cache =",
        "md.yf.download =",
    ):
        assert token not in s


def test_no_trading_endpoints():
    s = TARGET.read_text(encoding="utf-8").lower()
    for token in ("tradingclient(", "orders.submit", "submit_order", "/v2/orders"):
        assert token not in s


def test_safety():
    s = TARGET.read_text(encoding="utf-8")
    assert 'AI_TRADER_AUTONOMOUS_EXECUTION"] = "0"' in s
    assert 'AI_TRADER_LIVE_TRADING"] = "0"' in s
    assert '"orders_allowed": False' in s


def test_exact_production_request():
    s = TARGET.read_text(encoding="utf-8")
    assert 'ticker=args.ticker' in s
    assert 'period=period' in s
    assert 'interval=interval' in s
    assert 'auto_adjust=auto_adjust' in s
    assert 'actions=actions' in s
    assert 'group_by=group_by' in s
    assert 'ttl_seconds=ttl_seconds' in s


def test_two_request_design():
    s = TARGET.read_text(encoding="utf-8")
    assert s.count("_run_one(") >= 3
    assert 'r1 = _run_one(' in s
    assert 'r2 = _run_one(' in s
    assert '"same_request_key"' in s


if __name__ == "__main__":
    tests = [
        test_exists_and_compiles,
        test_minimal_scope,
        test_no_production_replacement,
        test_no_trading_endpoints,
        test_safety,
        test_exact_production_request,
        test_two_request_design,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
    print("PERFORMANCE DIAGNOSTIC V2.3.2 CONTRACT TEST: PASS")
