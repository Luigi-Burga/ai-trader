from __future__ import annotations

"""AI Trader - New Listing Engine (NLE) V1.4.

V1.4 routes all market-data acquisition through Market Data V2 and preserves
unadjusted Close semantics. Existing NLE scoring/decision logic is unchanged.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import argparse, math, json
import numpy as np
import pandas as pd
from app.data.market_data import get_history

@dataclass
class NewListingConfig:
    period: str = "1y"
    interval: str = "1d"
    benchmark: Optional[str] = None
    minimum_days_for_nle: int = 60
    mature_days_for_cce: int = 320
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_window: int = 20
    buy_score: float = 70.0
    watch_score: float = 50.0
    avoid_score: float = 35.0
    min_buy_confidence: float = 55.0
    pullback_atr: float = 1.0
    stop_atr: float = 2.0
    max_entry_extension_atr: float = 1.5

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV; Adj Close is never mapped to Close."""
    if df is None or df.empty: return pd.DataFrame()
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        selected={}
        for field in ("Open","High","Low","Close","Volume"):
            for col in x.columns:
                if field in [str(part).strip() for part in col]:
                    selected[field]=col; break
        if selected:
            x=x[[selected[k] for k in selected]]; x.columns=list(selected.keys())
    rename={}
    for c in x.columns:
        key=str(c).strip().lower()
        if key in {"open","high","low","close","volume"}: rename[c]=key.capitalize()
    x=x.rename(columns=rename); x=x.loc[:,~x.columns.duplicated(keep="first")]
    for c in ("Open","High","Low","Close","Volume"):
        if c not in x.columns:
            if c=="Volume": x[c]=0.0
            else: return pd.DataFrame()
    x=x[["Open","High","Low","Close","Volume"]].copy()
    for c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=["High","Low","Close"])
    return x[~x.index.duplicated(keep="last")].sort_index()

def download_history(ticker:str,cfg:NewListingConfig,period:Optional[str]=None)->pd.DataFrame:
    if not ticker: return pd.DataFrame()
    try:
        return _clean_ohlcv(get_history(ticker,period=period or cfg.period,interval=cfg.interval,
                                        auto_adjust=False,actions=False,group_by="column"))
    except Exception: return pd.DataFrame()

def _rsi(close,period):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    al=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    rs=ag/al.replace(0,np.nan); return 100-100/(1+rs)

def _atr(data,period):
    prev=data["Close"].shift(1)
    tr=pd.concat([data["High"]-data["Low"],(data["High"]-prev).abs(),(data["Low"]-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean()

def _zscore(s,window):
    m=s.rolling(window).mean(); sd=s.rolling(window).std(ddof=0).replace(0,np.nan); return (s-m)/sd

def build_features(data,benchmark_df=None,cfg=None):
    cfg=cfg or NewListingConfig(); x=data.copy(); c=x["Close"]
    x["ema20"]=c.ewm(span=cfg.ema_fast,adjust=False).mean(); x["ema50"]=c.ewm(span=cfg.ema_slow,adjust=False).mean()
    x["rsi"]=_rsi(c,cfg.rsi_period); x["atr"]=_atr(x,cfg.atr_period); x["atr_pct"]=x["atr"]/c
    x["return_5d"]=c.pct_change(5); x["return_20d"]=c.pct_change(20); x["return_since_listing"]=c/c.iloc[0]-1
    x["listing_high"]=c.cummax(); x["listing_low"]=c.cummin(); x["drawdown_from_high"]=c/x["listing_high"]-1; x["recovery_from_low"]=c/x["listing_low"]-1
    vol=x["Volume"].replace(0,np.nan); x["volume_z"]=_zscore(vol,cfg.volume_window); x["avg_volume_20"]=x["Volume"].rolling(cfg.volume_window).mean(); x["volume_ratio"]=x["Volume"]/x["avg_volume_20"]
    x["ema20_distance_atr"]=(c-x["ema20"])/x["atr"]; x["ema50_distance_atr"]=(c-x["ema50"])/x["atr"]; x["rs_20d"]=0.0
    if benchmark_df is not None and not benchmark_df.empty:
        bdf=_clean_ohlcv(benchmark_df)
        if not bdf.empty:
            b=bdf["Close"].reindex(x.index).ffill(); rs=c/b; x["rs_20d"]=rs/rs.shift(20)-1
    return x.replace([np.inf,-np.inf],np.nan)

def _age_class(days,cfg):
    if days>=cfg.mature_days_for_cce:return "MATURE"
    if days>=120:return "NEW_LISTING_MATURE"
    if days>=cfg.minimum_days_for_nle:return "NEW_LISTING_EARLY"
    if days>=30:return "EARLY_LISTING"
    return "INSUFFICIENT_DATA"

def _f(v):
    try:
        v=float(v); return v if math.isfinite(v) else np.nan
    except Exception:return np.nan

def _regime(r):
    p,e20,e50,rsi,dd=map(_f,[r["Close"],r["ema20"],r["ema50"],r["rsi"],r["drawdown_from_high"]])
    if not all(math.isfinite(v) for v in (p,e20,e50,rsi,dd)):return "UNKNOWN"
    if p>e20>e50 and rsi>=55:return "BULLISH"
    if p<e20<e50 and rsi<=45:return "BEARISH"
    if dd<=-0.20 and p>e20:return "RECOVERY"
    return "CONSOLIDATION"

def _score(r):
    score=50.0; b={}
    def add(k,v):
        nonlocal score; score+=v; b[k]=v
    p,e20,e50,rsi,r20,rs,vr,dd=map(_f,[r["Close"],r["ema20"],r["ema50"],r["rsi"],r["return_20d"],r["rs_20d"],r["volume_ratio"],r["drawdown_from_high"]])
    add("trend",12 if p>e20 else -8); add("long_trend",10 if p>e50 else -8)
    if math.isfinite(rsi):add("momentum",10 if 55<=rsi<=70 else 2 if 45<=rsi<55 else -5 if rsi<35 else -8)
    if math.isfinite(r20):add("price_momentum",8 if r20>0.05 else 3 if r20>0 else -7)
    if math.isfinite(rs):add("relative_strength",8 if rs>0.03 else 3 if rs>=0 else -6)
    if math.isfinite(vr):add("volume_confirmation",5 if 1.2<=vr<=3 else 2 if vr>=0.8 else -3)
    if math.isfinite(dd):add("drawdown_structure",5 if -0.15<=dd<=-0.05 else 2 if dd>-0.05 else -4)
    return max(0,min(100,score)),b

def _data_quality(days,r):
    if days<30:return "INSUFFICIENT"
    required=["ema20","ema50","rsi","atr","volume_ratio"]; available=sum(math.isfinite(_f(r[k])) for k in required)
    if days<60 or available<len(required):return "LIMITED"
    if days<120:return "MODERATE"
    return "GOOD"

def _confidence(r,days,score):
    age_factor=min(1,max(0,(days-60)/260)); data_conf=35+30*age_factor; rsi,atrp=_f(r["rsi"]),_f(r["atr_pct"])
    quality=(10 if math.isfinite(rsi) and 35<=rsi<=70 else 3 if math.isfinite(rsi) else 0); quality+=(10 if math.isfinite(atrp) and atrp<=0.08 else 4 if math.isfinite(atrp) else 0)
    score_quality=max(0,15-abs(score-60)*0.10); return round(min(79,data_conf+quality+score_quality),2)

def _levels(r,cfg):
    p,atr,e20=map(_f,[r["Close"],r["atr"],r["ema20"]])
    if not all(math.isfinite(v) for v in (p,atr)) or atr<=0:return {k:None for k in ["current_price","zone_low","zone_high","preferred_entry","stop_loss","tp_1","tp_2","tp_3"]}
    low=p-cfg.pullback_atr*atr; high=p+0.25*atr; preferred=p-0.5*atr
    if math.isfinite(e20) and p>e20+cfg.max_entry_extension_atr*atr:high=e20+0.25*atr; preferred=e20
    stop=low-cfg.stop_atr*atr
    return {"current_price":round(p,4),"zone_low":round(max(0,low),4),"zone_high":round(max(0,high),4),"preferred_entry":round(max(0,preferred),4),"stop_loss":round(max(0,stop),4),"tp_1":round(p+1.5*atr,4),"tp_2":round(p+2.5*atr,4),"tp_3":round(p+4*atr,4)}

def analyze_dataframe(ticker,df,benchmark_df=None,cfg=None,listing_price=None):
    cfg=cfg or NewListingConfig(); data=_clean_ohlcv(df); days=len(data)
    if data.empty:return {"engine":"New Listing Engine","version":"1.4","ticker":ticker,"signal":"NO_DATA","decision":"WAIT","score":0.0,"confidence":0.0}
    listing_date=data.index[0].date().isoformat(); first_trading_close=_f(data["Close"].iloc[0]); cls=_age_class(days,cfg)
    if cls=="MATURE":return {"engine":"New Listing Engine","version":"1.4","ticker":ticker,"signal":"ROUTE_TO_CCE","decision":"USE_CCE","trading_days":days,"classification":cls,"listing_date":listing_date,"first_trading_close":round(first_trading_close,4) if math.isfinite(first_trading_close) else None,"score":0.0,"confidence":0.0,"data_quality":"GOOD"}
    if cls in {"INSUFFICIENT_DATA","EARLY_LISTING"}:return {"engine":"New Listing Engine","version":"1.4","ticker":ticker,"signal":"INSUFFICIENT_DATA","decision":"WAIT","trading_days":days,"classification":cls,"listing_date":listing_date,"first_trading_close":round(first_trading_close,4) if math.isfinite(first_trading_close) else None,"score":0.0,"confidence":0.0,"data_quality":"INSUFFICIENT" if cls=="INSUFFICIENT_DATA" else "LIMITED"}
    f=build_features(data,benchmark_df,cfg); r=f.iloc[-1]; score,breakdown=_score(r); conf=_confidence(r,days,score); regime=_regime(r); decision="WATCH"
    if regime=="BEARISH" and score<cfg.watch_score:decision="AVOID"
    elif score>=cfg.buy_score and conf>=cfg.min_buy_confidence and regime in {"BULLISH","RECOVERY"}:decision="BUY_ON_CONFIRMATION"
    elif score<cfg.watch_score:decision="WAIT"
    if cls=="NEW_LISTING_EARLY" and decision=="BUY_ON_CONFIRMATION":decision="BUY_ON_CONFIRMATION"
    p,atr,e20=_f(r["Close"]),_f(r["atr"]),_f(r["ema20"]); extended=all(math.isfinite(v) for v in (p,atr,e20)) and p>e20+cfg.max_entry_extension_atr*atr
    if extended and decision=="BUY_ON_CONFIRMATION":decision="BUY_ON_PULLBACK"
    effective_listing_price=_f(listing_price) if listing_price is not None else np.nan
    return {"engine":"New Listing Engine","version":"1.4","ticker":ticker,"as_of":str(data.index[-1].date()),"trading_days":days,"classification":cls,"signal":decision,"decision":decision,"score":round(score,2),"confidence":conf,"data_quality":_data_quality(days,r),"regime":regime,"listing_date":listing_date,"listing_price":round(effective_listing_price,4) if math.isfinite(effective_listing_price) else None,"first_trading_close":round(first_trading_close,4) if math.isfinite(first_trading_close) else None,"metrics":{"current_price":round(_f(r["Close"]),4),"ema20":round(_f(r["ema20"]),4),"ema50":round(_f(r["ema50"]),4),"rsi":round(_f(r["rsi"]),2),"atr":round(_f(r["atr"]),4),"atr_pct":round(_f(r["atr_pct"])*100,2),"return_5d":round(_f(r["return_5d"])*100,2),"return_20d":round(_f(r["return_20d"])*100,2),"return_since_listing":round(_f(r["return_since_listing"])*100,2),"drawdown_from_high":round(_f(r["drawdown_from_high"])*100,2),"recovery_from_low":round(_f(r["recovery_from_low"])*100,2),"volume_ratio":round(_f(r["volume_ratio"]),2),"relative_strength_20d":round(_f(r["rs_20d"])*100,2)},"entry":_levels(r,cfg),"score_breakdown":{k:round(v,2) for k,v in breakdown.items()},"evidence":{"method":"post-listing price/volume/volatility/relative-strength","historical_pattern_matches":0,"note":"CCE historical matches are intentionally not used for a new listing; listing_price is optional and never inferred as IPO offer price."},"config":asdict(cfg)}

def analyze(ticker,cfg=None,benchmark=None,listing_price=None):
    cfg=cfg or NewListingConfig(benchmark=benchmark); benchmark=benchmark or cfg.benchmark; df=download_history(ticker,cfg)
    if df.empty:return analyze_dataframe(ticker,df,None,cfg,listing_price)
    bdf=download_history(benchmark,cfg) if benchmark else None
    return analyze_dataframe(ticker,df,bdf if bdf is not None and not bdf.empty else None,cfg,listing_price)

def _synthetic_data(n=140,seed=7):
    rng=np.random.default_rng(seed); dates=pd.bdate_range("2026-01-01",periods=n); ret=rng.normal(0.001,0.025,n); ret[-25:]+=0.0015; close=100*np.exp(np.cumsum(ret)); high=close*(1+rng.uniform(0.005,0.03,n)); low=close*(1-rng.uniform(0.005,0.03,n)); open_=close*(1+rng.normal(0,0.008,n)); vol=rng.integers(800_000,2_000_000,n).astype(float); return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol},index=dates)

def self_test():
    cfg=NewListingConfig(); df=_synthetic_data(); result=analyze_dataframe("TEST",df,cfg=cfg,listing_price=100)
    assert result["classification"]=="NEW_LISTING_MATURE"; assert result["decision"] in {"WAIT","WATCH","AVOID","BUY_ON_CONFIRMATION","BUY_ON_PULLBACK"}; assert 0<=result["score"]<=100 and 0<=result["confidence"]<=79
    assert result["listing_date"]==df.index[0].date().isoformat(); assert result["first_trading_close"] is not None; assert result["data_quality"] in {"MODERATE","GOOD"}
    for k in ("current_price","zone_low","zone_high","preferred_entry","stop_loss","tp_1","tp_2","tp_3"):assert result["entry"][k] is not None and math.isfinite(result["entry"][k])
    blocked=analyze_dataframe("SHORT",_synthetic_data(30),cfg=cfg); assert blocked["signal"]=="INSUFFICIENT_DATA"
    early_result=analyze_dataframe("EARLY",_synthetic_data(45),cfg=cfg); assert early_result["classification"]=="EARLY_LISTING" and early_result["signal"]=="INSUFFICIENT_DATA"
    routed=analyze_dataframe("MATURE",_synthetic_data(330),cfg=cfg); assert routed["signal"]=="ROUTE_TO_CCE"
    lowvol=df.copy(); lowvol["High"]=lowvol["Close"]*1.005; lowvol["Low"]=lowvol["Close"]*0.995; lowvol_result=analyze_dataframe("LOWVOL",lowvol,cfg=cfg); assert lowvol_result["entry"]["zone_low"]!=result["entry"]["zone_low"]
    print("SELF-TEST: PASS")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AI Trader New Listing Engine V1.4"); parser.add_argument("--ticker",default="SPCX"); parser.add_argument("--benchmark",default=None); parser.add_argument("--period",default="1y"); parser.add_argument("--listing-price",type=float,default=None); parser.add_argument("--self-test",action="store_true"); args=parser.parse_args()
    if args.self_test:self_test()
    else: print(json.dumps(analyze(args.ticker,NewListingConfig(period=args.period,benchmark=args.benchmark),args.benchmark,args.listing_price),indent=2,default=str))
