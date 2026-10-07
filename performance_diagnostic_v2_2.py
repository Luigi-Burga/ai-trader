"""AI Trader Performance Diagnostic V2.2 - Benchmark/Cache Deep Diagnostic.
Research-only: no production changes and no trading/orders.
"""
from __future__ import annotations
import argparse, hashlib, json, os, runpy, time
from collections import Counter, defaultdict
from pathlib import Path

VERSION="2.2"
RESEARCH_ONLY=True
PRODUCTION_CHANGES=False
ORDERS_ALLOWED=False
PROJECT_ROOT=Path("/app")
PRODUCTION_MAIN=PROJECT_ROOT/"app"/"main.py"
DEFAULT_OUTPUT=PROJECT_ROOT/"performance_diagnostic_v2_2_results.json"

class State:
    def __init__(self):
        self.calls=Counter(); self.seconds=Counter(); self.errors=Counter()
        self.cache=[]; self.yahoo=[]; self.resolver=[]; self.selected=[]
        self.asset=None; self.stage=None
    def benchmark(self,t):
        return str(t or "").upper() in {"SPY","QQQ","GDX","GDXJ","IGV","SMH","XLK","IWM","DIA","XLE","XLF","XLI","XLP","XLV"}
S=State()

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""): h.update(c)
    return h.hexdigest()

def req(args,kw):
    return {
        "ticker":kw.get("ticker",args[0] if args else None),
        "period":kw.get("period",args[1] if len(args)>1 else None),
        "interval":kw.get("interval",args[2] if len(args)>2 else None),
        "auto_adjust":kw.get("auto_adjust",args[3] if len(args)>3 else None)}

def instrument_market_data():
    import app.data.market_data as md
    original_cache=md._read_cache
    if not getattr(original_cache,"_v22",False):
        def cache(*args,**kwargs):
            t=time.perf_counter(); r=original_cache(*args,**kwargs)
            name=None
            for v in list(args)+list(kwargs.values()):
                if isinstance(v,(str,Path)) and ("market_" in str(v) or str(v).endswith(".pkl")):
                    name=Path(str(v)).name; break
            S.cache.append({"asset":S.asset,"stage":S.stage,"hit":r is not None,"cache":name,
                            "elapsed_seconds":time.perf_counter()-t})
            return r
        cache._v22=True; md._read_cache=cache
    yd=md.yf.download
    if not getattr(yd,"_v22",False):
        def download(*args,**kwargs):
            q=req(args,kwargs); t=time.perf_counter(); err=None
            try: return yd(*args,**kwargs)
            except Exception as e: err=repr(e); raise
            finally:
                S.yahoo.append({**q,"asset":S.asset,"stage":S.stage,
                                "is_benchmark":S.benchmark(q["ticker"]),
                                "elapsed_seconds":time.perf_counter()-t,"error":bool(err),"error_message":err or ""})
        download._v22=True; md.yf.download=download

def instrument_orchestrator():
    import app.ai.asset_analysis_orchestrator_v1_5_1 as m
    fn=m.analyze_ticker
    if getattr(fn,"_v22",False): return
    def wrap(*args,**kwargs):
        ticker=kwargs.get("ticker",args[0] if args else None)
        olda,olds=S.asset,S.stage; S.asset=str(ticker).upper(); S.stage="asset_analysis"
        t=time.perf_counter(); S.calls["orchestrator.analyze_ticker"]+=1
        try:
            r=fn(*args,**kwargs)
            if isinstance(r,dict):
                b=r.get("benchmark") or r.get("selected_benchmark")
                if b: S.selected.append({"asset":S.asset,"benchmark":b,"source":"orchestrator_result"})
            return r
        except Exception: S.errors["orchestrator.analyze_ticker"]+=1; raise
        finally:
            S.seconds["orchestrator.analyze_ticker"]+=time.perf_counter()-t; S.asset, S.stage=olda,olds
    wrap._v22=True; m.analyze_ticker=wrap

def instrument_scanner():
    import app.scanners.watchlist_scanner_v2_2_2 as m
    fn=m.scan_buy_opportunity
    if getattr(fn,"_v22",False): return
    def wrap(*args,**kwargs):
        ticker=kwargs.get("ticker",args[0] if args else None)
        olda,olds=S.asset,S.stage; S.asset=str(ticker).upper(); S.stage="watchlist_scanner"
        t=time.perf_counter(); S.calls["scanner.scan_buy_opportunity"]+=1
        try: return fn(*args,**kwargs)
        except Exception: S.errors["scanner.scan_buy_opportunity"]+=1; raise
        finally:
            S.seconds["scanner.scan_buy_opportunity"]+=time.perf_counter()-t; S.asset,S.stage=olda,olds
    wrap._v22=True; m.scan_buy_opportunity=wrap

def instrument_main():
    import app.main as m
    for n in ("_run_prediction_tracker","monitor_position","scan_buy_opportunity","_log_prediction_snapshot",
              "_execution_sizing","calculate_fundamental_score","process_fundamental_alert"):
        fn=getattr(m,n,None)
        if fn and not getattr(fn,"_v22",False):
            def make(fn,name):
                def wrap(*a,**k):
                    t=time.perf_counter(); S.calls["main."+name]+=1
                    try:return fn(*a,**k)
                    except Exception:S.errors["main."+name]+=1;raise
                    finally:S.seconds["main."+name]+=time.perf_counter()-t
                wrap._v22=True; return wrap
            setattr(m,n,make(fn,n))

def instrument_tracker():
    import app.prediction_tracker as m
    for n in ("update_all_predictions","_build_market_data_cache","update_prediction_file","evaluate_snapshot"):
        fn=getattr(m,n,None)
        if fn and not getattr(fn,"_v22",False):
            def make(fn,name):
                def wrap(*a,**k):
                    t=time.perf_counter(); S.calls["tracker."+name]+=1
                    try:return fn(*a,**k)
                    except Exception:S.errors["tracker."+name]+=1;raise
                    finally:S.seconds["tracker."+name]+=time.perf_counter()-t
                wrap._v22=True; return wrap
            setattr(m,n,make(fn,n))

def run(out,force=False):
    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"]="0"
    os.environ["AI_TRADER_LIVE_TRADING"]="0"
    if force: os.environ["AI_TRADER_FORCE_SCAN"]="1"
    before=cache_state()
    import app.main
    instrument_market_data(); instrument_orchestrator(); instrument_scanner(); instrument_main(); instrument_tracker()
    t=time.perf_counter(); status="OK"; code=0; err=None
    try: runpy.run_path(str(PRODUCTION_MAIN),run_name="__main__")
    except SystemExit as e:
        code=int(e.code) if isinstance(e.code,int) else 1
        if code: status="ERROR"; err=repr(e)
    except Exception as e: status="ERROR"; code=1; err=repr(e)
    elapsed=time.perf_counter()-t
    after=cache_state()
    benchmarks=[x for x in S.yahoo if x["is_benchmark"]]
    bmap=defaultdict(lambda:{"calls":0,"seconds":0.0,"errors":0,"assets":Counter(),"signatures":Counter()})
    for x in benchmarks:
        b=str(x["ticker"]).upper(); z=bmap[b]; z["calls"]+=1; z["seconds"]+=x["elapsed_seconds"]; z["errors"]+=int(x["error"])
        sig=json.dumps({k:x.get(k) for k in ("ticker","period","interval","auto_adjust")},sort_keys=True); z["signatures"][sig]+=1
        if x["asset"]: z["assets"][x["asset"]]+=1
    dup={b:{"calls":z["calls"],"seconds":z["seconds"],"duplicate_signatures":{k:v for k,v in z["signatures"].items() if v>1},"assets":dict(z["assets"])}
        for b,z in bmap.items()}
    report={"version":VERSION,"research_only":True,"production_changes":False,"orders_allowed":False,
            "production_sha256":sha256(PRODUCTION_MAIN),
            "run":{"status":status,"return_code":code,"elapsed_seconds":elapsed,
                   "safety":{"autonomous_execution":"0","orders_allowed":False,"live_trading":False},
                   "calls":dict(S.calls),"seconds":dict(S.seconds),"errors":dict(S.errors)},
            "market_data":{"cache_reads":len(S.cache),"cache_hits":sum(x["hit"] for x in S.cache),
                           "cache_misses":sum(not x["hit"] for x in S.cache),
                           "yfinance_calls":len(S.yahoo),
                           "yfinance_seconds":sum(x["elapsed_seconds"] for x in S.yahoo),
                           "benchmark_yfinance_calls":len(benchmarks),
                           "benchmark_yfinance_seconds":sum(x["elapsed_seconds"] for x in benchmarks)},
            "benchmark_by_symbol":{b:{**z,"assets":dict(z["assets"]),"signatures":dict(z["signatures"])} for b,z in bmap.items()},
            "duplicate_benchmark_signatures":dup,
            "selected_benchmarks":S.selected,
            "all_benchmark_yahoo_requests":benchmarks,
            "all_cache_reads":S.cache,
            "cache_state_before":before,"cache_state_after":after,"error":err}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print("PERFORMANCE DIAGNOSTIC V2.2"); print(f"Status: {status} | Elapsed: {elapsed:.2f}s")
    print(f"Yahoo: {len(S.yahoo)} calls / {sum(x['elapsed_seconds'] for x in S.yahoo):.2f}s")
    print("BENCHMARKS:")
    for b,z in sorted(bmap.items(),key=lambda kv:kv[1]["seconds"],reverse=True):
        print(f"  {b}: {z['calls']} calls / {z['seconds']:.2f}s")
    print(f"JSON: {out}")
    return code

def cache_state():
    d=Path("/app/data/market_cache")
    fs=list(d.glob("*")) if d.exists() else []
    return {"cache_dir":str(d),"files":len(fs),"size_bytes":sum(p.stat().st_size for p in fs if p.is_file())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",action="store_true"); p.add_argument("--force-scan",action="store_true")
    p.add_argument("--output",default=str(DEFAULT_OUTPUT)); a=p.parse_args()
    if not a.run: p.error("Use --run")
    return run(Path(a.output),a.force_scan)

if __name__=="__main__": raise SystemExit(main())
