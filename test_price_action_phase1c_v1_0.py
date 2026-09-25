"""AI Trader - Price Action Phase 1C Statistical Validation V1.0."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from app.data.market_data import get_daily_history
from app.ai.price_action_engine_v1_0 import PriceActionConfig, build_features

TICKERS = ["NVDA", "TSM", "TQQQ", "SPCX", "PLTU"]
HORIZONS = (1, 5, 10, 20)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "price_action_validation"
DISTRIBUTION_JSON = OUTPUT_DIR / "price_action_phase1c_distribution.json"
FORWARD_CSV = OUTPUT_DIR / "price_action_phase1c_forward_returns.csv"
REPORT_TXT = OUTPUT_DIR / "price_action_phase1c_report.txt"

def _safe_float(v: Any):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def _stats(s: pd.Series):
    s=pd.to_numeric(s, errors="coerce").dropna()
    return {"count":int(len(s)),"mean":_safe_float(s.mean()),"median":_safe_float(s.median()),"std":_safe_float(s.std(ddof=1)) if len(s)>1 else None,"min":_safe_float(s.min()),"max":_safe_float(s.max())}

def _dist(s: pd.Series):
    s=s.dropna(); n=len(s); out={}
    for k,v in s.value_counts().items(): out[str(k)]={"count":int(v),"pct":round(v/n*100,4) if n else None}
    return {"count":n,"values":out}

def _forward(df,event,h):
    c=pd.to_numeric(df["Close"],errors="coerce"); hi=pd.to_numeric(df["High"],errors="coerce"); lo=pd.to_numeric(df["Low"],errors="coerce")
    rets=[]; mfes=[]; maes=[]
    for i in np.flatnonzero(event.fillna(False).to_numpy()):
        j=i+h
        if j>=len(df) or not np.isfinite(c.iloc[i]) or c.iloc[i]==0 or not np.isfinite(c.iloc[j]): continue
        entry=c.iloc[i]; rets.append(float(c.iloc[j]/entry-1))
        fh=hi.iloc[i+1:j+1]; fl=lo.iloc[i+1:j+1]
        if len(fh): mfes.append(float(fh.max()/entry-1))
        if len(fl): maes.append(float(fl.min()/entry-1))
    r=pd.Series(rets,dtype="float64")
    out=_stats(r); out.update({"positive_rate":_safe_float((r>0).mean()) if len(r) else None,"mfe_mean":_safe_float(np.mean(mfes)) if mfes else None,"mfe_median":_safe_float(np.median(mfes)) if mfes else None,"mae_mean":_safe_float(np.mean(maes)) if maes else None,"mae_median":_safe_float(np.median(maes)) if maes else None,"sample_forward_complete":len(r)})
    return out

def analyze_ticker(ticker):
    raw=get_daily_history(ticker,period="5y",auto_adjust=False)
    if raw is None or raw.empty: raise RuntimeError("No historical OHLCV data returned")
    features=build_features(raw,cfg=PriceActionConfig())
    required={"close_above_breakout_high","close_below_breakout_low","outside_bar","candle_close_state"}
    if features.empty or not required.issubset(features.columns): raise RuntimeError("Price Action feature contract validation failed")
    close_state=features["candle_close_state"]; outside_state=features["outside_bar_state"]
    sc=features["candle_strong_close"].astype(bool); wc=features["candle_weak_close"].astype(bool)
    ob=features["outside_bar"].astype(bool); bull=features["outside_bar_bullish"].astype(bool); bear=features["outside_bar_bearish"].astype(bool); neutral=features["outside_bar_neutral"].astype(bool); exp=features["outside_bar_expanded_range"].astype(bool)
    events={"close_near_high":close_state.eq("CLOSE_NEAR_HIGH"),"close_near_low":close_state.eq("CLOSE_NEAR_LOW"),"close_mid_range":close_state.eq("CLOSE_MID_RANGE"),"strong_close":sc,"weak_close":wc,"close_above_prev_high":features["close_above_prev_high"].astype(bool),"close_below_prev_low":features["close_below_prev_low"].astype(bool),"outside_bar":ob,"outside_bar_bullish":bull,"outside_bar_bearish":bear,"outside_bar_neutral":neutral,"outside_bar_expanded":exp,"outside_bar_bullish_strong_close":bull&sc,"outside_bar_bearish_strong_close":bear&sc}
    result={"ticker":ticker,"rows":len(raw),"first_date":str(raw.index.min().date()),"last_date":str(raw.index.max().date()),"engine":"Price Action Engine","engine_version":"1.0","lookahead_contract":True,"candle_close":{"distribution":_dist(close_state),"strong_close":_dist(features["candle_strong_close"].map({True:"TRUE",False:"FALSE"})),"weak_close":_dist(features["candle_weak_close"].map({True:"TRUE",False:"FALSE"}))},"outside_bar":{"distribution":_dist(features["outside_bar"].map({True:"TRUE",False:"FALSE"})),"direction":_dist(outside_state.astype(str).where(outside_state.notna(),"NONE")),"state":_dist(outside_state)},"event_analysis":[]}
    baseline=pd.Series(True,index=raw.index)
    for name,mask in events.items():
        e={"event":name,"event_count":int(mask.sum()),"horizons":{}}
        for h in HORIZONS:
            s=_forward(raw,mask,h); b=_forward(raw,baseline,h); s["baseline_mean_return"]=b["mean"]; s["delta_vs_baseline"]=_safe_float(s["mean"]-b["mean"]) if s["mean"] is not None and b["mean"] is not None else None; e["horizons"][f"{h}d"]=s
        result["event_analysis"].append(e)
    return result

def flatten(ticker,r):
    rows=[]
    for e in r["event_analysis"]:
        for h,s in e["horizons"].items(): rows.append({"ticker":ticker,"event":e["event"],"horizon":h,"event_count":e["event_count"],"forward_complete":s["sample_forward_complete"],"mean_return":s["mean"],"median_return":s["median"],"std_return":s["std"],"positive_rate":s["positive_rate"],"baseline_mean_return":s["baseline_mean_return"],"delta_vs_baseline":s["delta_vs_baseline"],"mfe_mean":s["mfe_mean"],"mfe_median":s["mfe_median"],"mae_mean":s["mae_mean"],"mae_median":s["mae_median"]})
    return rows

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); results={}; errors={}; rows=[]
    print("AI TRADER - PRICE ACTION PHASE 1C STATISTICAL VALIDATION V1.0\n")
    print("Production engines modified: NO\nOrchestrator modified: NO\nBenchmark Resolver modified: NO\nDecision Gate modified: NO\nSignal Engine modified: NO\nPrice Action generates trading signals: NO\nData source: Market Data V2 persistent cache / data/market_cache\nHorizons: 1d, 5d, 10d, 20d\n")
    for t in TICKERS:
        try:
            r=analyze_ticker(t); results[t]=r; rows+=flatten(t,r); print(f"{t} | PASS | rows={r['rows']} | {r['first_date']} -> {r['last_date']}")
        except Exception as e: errors[t]=str(e); print(f"{t} | ERROR | {e}")
    DISTRIBUTION_JSON.write_text(json.dumps({"version":"1.0","engine":"Price Action Engine","engine_version":"1.0","tickers":TICKERS,"horizons":list(HORIZONS),"results":results,"errors":errors},indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame(rows).to_csv(FORWARD_CSV,index=False)
    lines=["AI TRADER - PRICE ACTION PHASE 1C STATISTICAL VALIDATION V1.0","="*72,"","Production engines modified: NO","Orchestrator modified: NO","Benchmark Resolver modified: NO","Decision Gate modified: NO","Signal Engine modified: NO","Price Action generates trading signals: NO","Data source: Market Data V2 persistent cache / data/market_cache","Horizons: 1d, 5d, 10d, 20d",""]
    for t in TICKERS:
        if t not in results: lines.append(f"{t} | ERROR | {errors[t]}"); continue
        r=results[t]; lines.append(f"{t} | PASS | rows={r['rows']} | {r['first_date']} -> {r['last_date']}")
        lines.append("  Candle Close distribution:")
        for k,v in r["candle_close"]["distribution"]["values"].items(): lines.append(f"    {k}: {v['count']} ({v['pct']:.2f}%)")
        lines.append("  Outside Bar direction:")
        for k,v in r["outside_bar"]["direction"]["values"].items(): lines.append(f"    {k}: {v['count']} ({v['pct']:.2f}%)")
        lines.append("  Forward-return events:")
        for e in r["event_analysis"]:
            if e["event"] not in {"close_near_high","close_near_low","strong_close","outside_bar","outside_bar_bullish","outside_bar_bearish","outside_bar_bullish_strong_close","outside_bar_bearish_strong_close"}: continue
            p=[]
            for h in ("1d","5d","10d","20d"):
                s=e["horizons"][h]; fmt=lambda x:"N/A" if x is None else f"{x*100:.2f}%"; p.append(f"{h}: n={s['sample_forward_complete']}, ret={fmt(s['mean'])}, pos={fmt(s['positive_rate'])}, delta={fmt(s['delta_vs_baseline'])}")
            lines.append(f"    {e['event']}: " + " | ".join(p))
        lines.append("")
    lines += ["="*72,f"Successful tickers: {len(results)}/{len(TICKERS)}",f"Errors: {len(errors)}","","Interpretation note: descriptive/statistical validation only; positive historical deltas do not by themselves establish predictive or causal value and do not authorize production integration."]
    REPORT_TXT.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nSuccessful tickers: {len(results)}/{len(TICKERS)}\nErrors: {len(errors)}")
    print(f"JSON report: {DISTRIBUTION_JSON}\nCSV report:  {FORWARD_CSV}\nTXT report:  {REPORT_TXT}")
    return 0 if not errors else 1

if __name__ == "__main__": raise SystemExit(main())
