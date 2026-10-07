"""AI Trader Performance Diagnostic V2.3 — Cache Key / Request Lineage.
Research-only: no production changes, no orders, live/autonomous execution disabled.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, os, runpy, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION = "2.3"
RESEARCH_ONLY = True
PRODUCTION_CHANGES = False
ORDERS_ALLOWED = False
PROJECT_ROOT = Path("/app")
PRODUCTION_MAIN = PROJECT_ROOT / "app" / "main.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "performance_diagnostic_v2_3_results.json"
BENCHMARKS = {"SPY","QQQ","GDX","GDXJ","IGV","SMH","XLK","IWM","DIA","XLE","XLF","XLI","XLP","XLV"}

class State:
    def __init__(self):
        self.calls=Counter(); self.seconds=Counter(); self.errors=Counter()
        self.requests=[]; self.key_events=[]; self.cache_reads=[]; self.cache_writes=[]; self.yahoo=[]
        self.request_by_key={}; self.asset=None; self.stage=None
    @staticmethod
    def benchmark(t): return str(t or "").strip().upper() in BENCHMARKS
S=State()

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""): h.update(c)
    return h.hexdigest()

def caller():
    for fi in inspect.stack()[2:]:
        mod=str(fi.frame.f_globals.get("__name__",""))
        if mod.startswith("app.") and mod != __name__:
            return {"module":mod,"function":fi.function,"line":fi.lineno}
    return {"module":None,"function":None,"line":None}

def payload(args,kw):
    def pick(n,i,d=None): return kw[n] if n in kw else (args[i] if len(args)>i else d)
    return {"ticker":str(pick("ticker",0,"") or "").strip().upper(),
            "period":pick("period",1,None),"interval":pick("interval",2,"1d"),
            "start":pick("start",3,None),"end":pick("end",4,None),
            "auto_adjust":bool(pick("auto_adjust",5,True)),
            "actions":bool(pick("actions",6,False)),
            "group_by":pick("group_by",7,"column")}

def key_for(p):
    raw=json.dumps(p,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()

def remember(source,p,k):
    r={"source":source,"request_key":k,"request":p,"is_benchmark":S.benchmark(p["ticker"]),
       "asset_context":S.asset,"stage":S.stage,"caller":caller(),"timestamp":time.time()}
    S.requests.append(r); S.request_by_key.setdefault(k,r)

def instrument_market_data():
    import app.data.market_data as md

    orig=md._request_key
    if not getattr(orig,"_v23",False):
        def rk(ticker,*,period,interval,start,end,auto_adjust,actions,group_by):
            p={"ticker":str(ticker or "").strip().upper(),"period":period,"interval":interval,
               "start":start,"end":end,"auto_adjust":bool(auto_adjust),
               "actions":bool(actions),"group_by":group_by}
            k=orig(ticker,period=period,interval=interval,start=start,end=end,
                   auto_adjust=auto_adjust,actions=actions,group_by=group_by)
            S.request_by_key.setdefault(k,{"source":"_request_key","request_key":k,"request":p,
                "is_benchmark":S.benchmark(p["ticker"]),"asset_context":S.asset,"stage":S.stage,
                "caller":caller(),"timestamp":time.time()})
            S.key_events.append({"event":"request_key","request_key":k,"request":p,
                "asset_context":S.asset,"stage":S.stage,"caller":caller()})
            return k
        rk._v23=True; md._request_key=rk

    orig=md._cache_path
    if not getattr(orig,"_v23",False):
        def cp(cache_dir,k):
            p=orig(cache_dir,k); known=S.request_by_key.get(k,{})
            S.key_events.append({"event":"cache_path","request_key":k,"cache_path":str(p),
                "cache_exists":p.exists(),"request":known.get("request"),
                "is_benchmark":known.get("is_benchmark",False),"asset_context":S.asset,
                "stage":S.stage,"caller":caller()})
            return p
        cp._v23=True; md._cache_path=cp

    orig=md._read_cache
    if not getattr(orig,"_v23",False):
        def rc(path,ttl):
            t=time.perf_counter(); r=orig(path,ttl); e=time.perf_counter()-t
            p=Path(path); k=p.stem.removeprefix("market_")
            try: age=time.time()-p.stat().st_mtime
            except OSError: age=None
            known=S.request_by_key.get(k,{})
            S.cache_reads.append({"request_key":k,"cache_path":str(p),"cache_exists":p.exists(),
                "cache_age_seconds":age,"ttl_seconds":ttl,"hit":r is not None,"elapsed_seconds":e,
                "request":known.get("request"),"is_benchmark":known.get("is_benchmark",False),
                "asset_context":S.asset,"stage":S.stage,"caller":caller()})
            return r
        rc._v23=True; md._read_cache=rc

    orig=md._write_cache
    if not getattr(orig,"_v23",False):
        def wc(path,data,request):
            t=time.perf_counter(); orig(path,data,request); e=time.perf_counter()-t
            p=Path(path); k=p.stem.removeprefix("market_")
            S.cache_writes.append({"request_key":k,"cache_path":str(p),
                "cache_exists_after":p.exists(),"request":request,
                "is_benchmark":S.benchmark(request.get("ticker")),"asset_context":S.asset,
                "stage":S.stage,"elapsed_seconds":e,"caller":caller()})
        wc._v23=True; md._write_cache=wc

    for name in ("get_history","get_daily_history"):
        orig=getattr(md,name)
        if getattr(orig,"_v23",False): continue
        def make_get(fn,n):
            def get(*a,**kw):
                p=payload(a,kw); k=key_for(p); remember(n,p,k)
                oa,os_=S.asset,S.stage
                if p["ticker"]: S.asset=p["ticker"]
                S.stage=S.stage or "market_data."+n
                t=time.perf_counter()
                try: return fn(*a,**kw)
                except Exception: S.errors["market_data."+n]+=1; raise
                finally:
                    S.calls["market_data."+n]+=1; S.seconds["market_data."+n]+=time.perf_counter()-t
                    S.asset,S.stage=oa,os_
            get._v23=True; return get
        setattr(md,name,make_get(orig,name))

    orig=md.yf.download
    if not getattr(orig,"_v23",False):
        def yd(*a,**kw):
            ticker=str(kw.get("tickers",kw.get("ticker",a[0] if a else "")) or "").strip().upper()
            q={k:kw.get(k) for k in ("period","interval","start","end","auto_adjust","actions","group_by","threads")}
            q["ticker"]=ticker; t=time.perf_counter(); err=None
            try: return orig(*a,**kw)
            except Exception as ex: err=f"{type(ex).__name__}: {ex}"; raise
            finally:
                S.yahoo.append({**q,"is_benchmark":S.benchmark(ticker),"asset_context":S.asset,
                    "stage":S.stage,"elapsed_seconds":time.perf_counter()-t,"error":bool(err),
                    "error_message":err or "","caller":caller()})
        yd._v23=True; md.yf.download=yd

def instrument_orchestrator():
    import app.ai.asset_analysis_orchestrator_v1_5_1 as m
    fn=m.analyze_ticker
    if getattr(fn,"_v23",False): return
    def w(*a,**kw):
        old=S.asset,S.stage; S.asset=str(kw.get("ticker",a[0] if a else "")).upper(); S.stage="asset_analysis"
        t=time.perf_counter()
        try:
            r=fn(*a,**kw)
            if isinstance(r,dict):
                b=r.get("benchmark") or r.get("selected_benchmark")
                if b: S.key_events.append({"event":"selected_benchmark","asset":S.asset,"benchmark":str(b).upper()})
            return r
        except Exception: S.errors["orchestrator.analyze_ticker"]+=1; raise
        finally:
            S.calls["orchestrator.analyze_ticker"]+=1; S.seconds["orchestrator.analyze_ticker"]+=time.perf_counter()-t
            S.asset,S.stage=old
    w._v23=True; m.analyze_ticker=w

def instrument_scanner():
    import app.scanners.watchlist_scanner_v2_2_2 as m
    fn=m.scan_buy_opportunity
    if getattr(fn,"_v23",False): return
    def w(*a,**kw):
        old=S.asset,S.stage; S.asset=str(kw.get("ticker",a[0] if a else "")).upper(); S.stage="watchlist_scanner"
        t=time.perf_counter()
        try: return fn(*a,**kw)
        except Exception: S.errors["scanner.scan_buy_opportunity"]+=1; raise
        finally:
            S.calls["scanner.scan_buy_opportunity"]+=1; S.seconds["scanner.scan_buy_opportunity"]+=time.perf_counter()-t
            S.asset,S.stage=old
    w._v23=True; m.scan_buy_opportunity=w

def instrument_main():
    import app.main as m
    for n in ("_run_prediction_tracker","monitor_position","scan_buy_opportunity","_log_prediction_snapshot",
              "_execution_sizing","calculate_fundamental_score","process_fundamental_alert"):
        fn=getattr(m,n,None)
        if fn is None or getattr(fn,"_v23",False): continue
        def make(f,name):
            def w(*a,**kw):
                t=time.perf_counter()
                try:return f(*a,**kw)
                except Exception:S.errors["main."+name]+=1;raise
                finally:S.calls["main."+name]+=1;S.seconds["main."+name]+=time.perf_counter()-t
            w._v23=True;return w
        setattr(m,n,make(fn,n))

def instrument_tracker():
    import app.prediction_tracker as m
    for n in ("update_all_predictions","_build_market_data_cache","update_prediction_file","evaluate_snapshot"):
        fn=getattr(m,n,None)
        if fn is None or getattr(fn,"_v23",False): continue
        def make(f,name):
            def w(*a,**kw):
                t=time.perf_counter()
                try:return f(*a,**kw)
                except Exception:S.errors["tracker."+name]+=1;raise
                finally:S.calls["tracker."+name]+=1;S.seconds["tracker."+name]+=time.perf_counter()-t
            w._v23=True;return w
        setattr(m,n,make(fn,n))

def cache_state():
    d=Path("/app/data/market_cache"); fs=list(d.glob("*")) if d.exists() else []
    return {"cache_dir":str(d),"files":len(fs),"size_bytes":sum(p.stat().st_size for p in fs if p.is_file())}

def lineage():
    groups={}
    for r in S.requests:
        if not r["is_benchmark"]: continue
        k=r["request_key"]; g=groups.setdefault(k,{"request_key":k,"request":r["request"],
            "assets":Counter(),"sources":Counter(),"cache_reads":[],"cache_writes":[],"yahoo_calls":[]})
        if r.get("asset_context"): g["assets"][r["asset_context"]]+=1
        g["sources"][r["source"]]+=1
    for x in S.cache_reads:
        if x["request_key"] in groups: groups[x["request_key"]]["cache_reads"].append(x)
    for x in S.cache_writes:
        if x["request_key"] in groups: groups[x["request_key"]]["cache_writes"].append(x)
    for y in S.yahoo:
        if not y["is_benchmark"]: continue
        matches=[]
        for k,g in groups.items():
            q=g["request"]; same=all(q.get(n)==y.get(n) for n in
                ("ticker","period","interval","start","end","auto_adjust","actions","group_by"))
            if same: matches.append(k)
        if len(matches)==1: groups[matches[0]]["yahoo_calls"].append(y)
        else: y["lineage_match"]="AMBIGUOUS" if matches else "NONE"
    out=[]
    for k,g in groups.items():
        out.append({"request_key":k,"request":g["request"],"assets":dict(g["assets"]),
            "sources":dict(g["sources"]),
            "cache_paths":sorted({x["cache_path"] for x in g["cache_reads"]+g["cache_writes"]}),
            "cache_reads":g["cache_reads"],"cache_writes":g["cache_writes"],"yahoo_calls":g["yahoo_calls"],
            "counts":{"request_events":sum(g["sources"].values()),"cache_reads":len(g["cache_reads"]),
                      "cache_writes":len(g["cache_writes"]),"yahoo_calls":len(g["yahoo_calls"])}})
    return sorted(out,key=lambda x:(-x["counts"]["yahoo_calls"],x["request"]["ticker"],x["request_key"]))

def run(out,force=False):
    os.environ["AI_TRADER_AUTONOMOUS_EXECUTION"]="0"; os.environ["AI_TRADER_LIVE_TRADING"]="0"
    if force: os.environ["AI_TRADER_FORCE_SCAN"]="1"
    before=cache_state()
    import app.main
    instrument_market_data();instrument_orchestrator();instrument_scanner();instrument_main();instrument_tracker()
    t=time.perf_counter();status="OK";code=0;error=None
    try: runpy.run_path(str(PRODUCTION_MAIN),run_name="__main__")
    except SystemExit as e:
        code=int(e.code) if isinstance(e.code,int) else 1
        if code: status="ERROR";error=repr(e)
    except Exception as e: status="ERROR";code=1;error=repr(e)
    elapsed=time.perf_counter()-t
    after=cache_state()
    by=defaultdict(lambda:{"calls":0,"seconds":0.0,"errors":0,"assets":Counter(),"signatures":Counter()})
    for x in S.yahoo:
        if not x["is_benchmark"]: continue
        z=by[x["ticker"]];z["calls"]+=1;z["seconds"]+=x["elapsed_seconds"];z["errors"]+=int(x["error"])
        if x.get("asset_context"):z["assets"][x["asset_context"]]+=1
        z["signatures"][json.dumps({k:x.get(k) for k in ("ticker","period","interval","start","end","auto_adjust","actions","group_by","threads")},sort_keys=True)]+=1
    keys=Counter(x["request_key"] for x in S.requests if x["is_benchmark"])
    report={"version":VERSION,"method":"Cache Key / Request Lineage Diagnostic","research_only":True,
        "production_changes":False,"orders_allowed":False,"production_sha256":sha256(PRODUCTION_MAIN),
        "run":{"status":status,"return_code":code,"elapsed_seconds":elapsed,
               "safety":{"autonomous_execution":"0","live_trading":"0","orders_allowed":False},
               "calls":dict(S.calls),"seconds":dict(S.seconds),"errors":dict(S.errors)},
        "market_data":{"cache_reads":len(S.cache_reads),"cache_hits":sum(x["hit"] for x in S.cache_reads),
            "cache_misses":sum(not x["hit"] for x in S.cache_reads),"cache_writes":len(S.cache_writes),
            "yfinance_calls":len(S.yahoo),"yfinance_seconds":sum(x["elapsed_seconds"] for x in S.yahoo),
            "benchmark_yfinance_calls":sum(x["is_benchmark"] for x in S.yahoo),
            "benchmark_yfinance_seconds":sum(x["elapsed_seconds"] for x in S.yahoo if x["is_benchmark"])},
        "duplicate_exact_request_keys":{k:v for k,v in keys.items() if v>1},
        "benchmark_summary":{k:{**v,"assets":dict(v["assets"]),"signatures":dict(v["signatures"])} for k,v in by.items()},
        "benchmark_request_lineage":lineage(),"all_requests":S.requests,"all_key_events":S.key_events,
        "all_cache_reads":S.cache_reads,"all_cache_writes":S.cache_writes,"all_yahoo_calls":S.yahoo,
        "cache_state_before":before,"cache_state_after":after,"error":error}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print("="*72);print("AI TRADER - PERFORMANCE DIAGNOSTIC V2.3");print("Cache Key / Request Lineage Diagnostic")
    print("Research-only | production source unchanged | orders blocked");print("="*72)
    print(f"Status: {status} | Elapsed: {elapsed:.2f}s");print(f"Production SHA256: {report['production_sha256']}")
    print(f"Cache reads={len(S.cache_reads)} hits={sum(x['hit'] for x in S.cache_reads)} misses={sum(not x['hit'] for x in S.cache_reads)} writes={len(S.cache_writes)}")
    print(f"Yahoo={len(S.yahoo)} calls / {sum(x['elapsed_seconds'] for x in S.yahoo):.2f}s")
    print("BENCHMARKS:")
    for k,v in sorted(by.items(),key=lambda kv:kv[1]["seconds"],reverse=True):
        print(f"  {k}: {v['calls']} calls / {v['seconds']:.2f}s / assets={len(v['assets'])}")
    print("DUPLICATE EXACT REQUEST KEYS:",sum(v>1 for v in keys.values()))
    print(f"JSON: {out}");print("="*72)
    return code

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");p.add_argument("--force-scan",action="store_true")
    p.add_argument("--output",default=str(DEFAULT_OUTPUT));a=p.parse_args()
    if not a.run:p.error("Use --run")
    return run(Path(a.output),a.force_scan)

if __name__=="__main__":raise SystemExit(main())
