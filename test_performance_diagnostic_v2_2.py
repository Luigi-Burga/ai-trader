from pathlib import Path
import ast, py_compile
TARGET=Path(__file__).with_name("performance_diagnostic_v2_2.py")
def test_exists_and_compiles(): assert TARGET.exists(); py_compile.compile(str(TARGET),doraise=True)
def test_safety():
 s=TARGET.read_text(); assert 'VERSION="2.2"' in s; assert "RESEARCH_ONLY=True" in s; assert 'AI_TRADER_AUTONOMOUS_EXECUTION' in s
def test_no_trading(): 
 s=TARGET.read_text().lower()
 for x in ("submit_order(","close_position(","cancel_order(","tradingclient(","orders.submit"): assert x not in s
def test_benchmark_diagnostics():
 s=TARGET.read_text(); 
 for x in ("benchmark_by_symbol","duplicate_benchmark_signatures","selected_benchmarks","benchmark_yfinance_calls"): assert x in s
def test_market_data(): 
 s=TARGET.read_text(); assert "_read_cache" in s and "yf.download" in s and "market_data" in s
def test_ast(): ast.parse(TARGET.read_text())
if __name__=="__main__":
 for n,v in sorted(globals().items()):
  if n.startswith("test_"): v(); print("PASS:",n)
 print("PERFORMANCE DIAGNOSTIC V2.2 CONTRACT TEST: PASS")
