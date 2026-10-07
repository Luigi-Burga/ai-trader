from pathlib import Path
import ast, py_compile
TARGET=Path(__file__).with_name("performance_diagnostic_v2_3.py")
def test_exists_and_compiles(): assert TARGET.exists(); py_compile.compile(str(TARGET),doraise=True)
def test_version_and_safety():
    s=TARGET.read_text(encoding="utf-8")
    assert 'VERSION = "2.3"' in s and "RESEARCH_ONLY = True" in s and "PRODUCTION_CHANGES = False" in s
def test_no_trading_endpoints():
    s=TARGET.read_text(encoding="utf-8").lower()
    for x in ("submit_order(","close_position(","cancel_order(","tradingclient(","orders.submit"): assert x not in s
def test_cache_lineage_contract():
    s=TARGET.read_text(encoding="utf-8")
    for x in ("_request_key","_cache_path","_read_cache","_write_cache","request_key","cache_path",
              "cache_age_seconds","benchmark_request_lineage","duplicate_exact_request_keys"): assert x in s
def test_request_contract():
    s=TARGET.read_text(encoding="utf-8")
    for x in ('"ticker"','"period"','"interval"','"start"','"end"','"auto_adjust"','"actions"','"group_by"'): assert x in s
def test_yahoo_contract(): assert "md.yf.download" in TARGET.read_text(encoding="utf-8")
def test_ast(): ast.parse(TARGET.read_text(encoding="utf-8"))
if __name__=="__main__":
    for n,v in sorted(globals().items()):
        if n.startswith("test_"): v();print("PASS:",n)
    print("PERFORMANCE DIAGNOSTIC V2.3 CONTRACT TEST: PASS")
