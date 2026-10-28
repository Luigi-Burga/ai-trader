"""Live PAPER adapter for BREAKOUT_VOL_FILTER.

Signals are computed only from fully completed daily bars using the v3.2
walk-forward rules. Entries use a MARKET BUY with Alpaca-native bracket exits.
This live adapter intentionally caps each position at 10% of strategy capital
and risks at most 0.5% of the $100,000 capital base using the ATR stop distance.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Mapping
import numpy as np
import pandas as pd

STRATEGY = "BREAKOUT_VOL_FILTER"
CAPITAL_BASE = 100_000.0
RISK_FRACTION = 0.005
MAX_POSITION_FRACTION = 0.10
MAX_HOLD_SESSIONS = 15
STATE_PATH = Path("app/data/breakout_vol_filter_state.json")
NY = ZoneInfo("America/New_York")


def flatten(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1): df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0): df = df.xs(ticker, axis=1, level=0)
        else: return pd.DataFrame()
    df = df.copy(); df.columns = [str(c).title() for c in df.columns]
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols): return pd.DataFrame()
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open","High","Low","Close"])
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["sma20"] = x.Close.rolling(20).mean()
    x["sma50"] = x.Close.rolling(50).mean()
    x["prior_high20"] = x.High.rolling(20).max().shift(1)
    x["vol_avg20"] = x.Volume.rolling(20).mean()
    prev = x.Close.shift(1)
    tr = pd.concat([x.High-x.Low, (x.High-prev).abs(), (x.Low-prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    delta = x.Close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    x["rsi14"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    ret = x.Close.pct_change()
    x["realized_vol20"] = ret.rolling(20).std() * np.sqrt(252)
    x["volatility_limit80"] = x.realized_vol20.rolling(100, min_periods=60).quantile(0.80).shift(1)
    return x


def signal_from_completed_bars(ticker_df: pd.DataFrame, spy_df: pd.DataFrame) -> dict[str, Any]:
    """Pure signal calculation for tests; dataframes must contain completed bars."""
    if ticker_df is None or spy_df is None or ticker_df.empty or spy_df.empty:
        return {"signal": "WATCH", "reason": "INSUFFICIENT_DATA"}
    x, spy = indicators(ticker_df), indicators(spy_df)
    common = x.index.intersection(spy.index)
    if len(common) < 2: return {"signal":"WATCH", "reason":"INSUFFICIENT_DATA"}
    d = common[-1]; row = x.loc[d]; sr = spy.loc[d]
    needed = [row.get(k, np.nan) for k in ("Close","sma20","sma50","prior_high20","vol_avg20","Volume","atr14","rsi14")]
    if any(pd.isna(v) for v in needed) or pd.isna(sr.get("sma50")) or pd.isna(sr.get("realized_vol20")) or pd.isna(sr.get("volatility_limit80")):
        return {"signal":"WATCH", "reason":"INDICATORS_NOT_READY", "signal_date":d.strftime("%Y-%m-%d")}
    close=float(row.Close); avgvol=float(row.vol_avg20); volume=float(row.Volume); atr=float(row.atr14); rsi=float(row.rsi14)
    checks = {
        "trend": close > float(row.sma50) and float(row.sma20) > float(row.sma50) and float(sr.Close) > float(sr.sma50),
        "breakout_20d": close > float(row.prior_high20),
        "volume_1_2x": avgvol > 0 and volume >= 1.2*avgvol,
        "rsi_50": rsi >= 50,
        "spy_volatility_filter": float(sr.realized_vol20) <= float(sr.volatility_limit80),
        "atr_valid": atr > 0,
    }
    return {"signal":"BUY" if all(checks.values()) else "WATCH", "reason":"ALL_FILTERS_PASSED" if all(checks.values()) else "FILTERS_NOT_CONFIRMED",
            "checks":checks,"signal_date":d.strftime("%Y-%m-%d"),"signal_close":close,"atr14":atr,"rsi14":rsi,
            "volume":volume,"volume_avg20":avgvol,"spy_realized_vol20":float(sr.realized_vol20),"spy_volatility_limit80":float(sr.volatility_limit80)}


def _state() -> dict:
    try:
        if STATE_PATH.exists():
            data=json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data,dict) else {"symbols":{}}
    except Exception:
        pass
    return {"symbols":{}}


def _save_state(data: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp=STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2),encoding="utf-8")
    tmp.replace(STATE_PATH)


def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # Lazy import keeps pure signal tests independent of the complete app package.
    # Production uses AI Trader's shared cache, not direct Yahoo downloads.
    from app.data.market_data import get_history
    return get_history(ticker, period=period, interval=interval, auto_adjust=True,
                       actions=False, group_by="column", threads=False)


def _current_price(ticker: str) -> float | None:
    try:
        raw = _download_history(ticker, "1d", "1m")
        frame = flatten(raw, ticker)
        if not frame.empty:
            value = float(pd.to_numeric(frame["Close"].iloc[-1], errors="coerce"))
            return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None
    return None


def _positions(executor) -> list[dict]:
    return executor.client.get_positions()


def scan_buy_opportunity(stock: Mapping[str, Any], *, executor=None) -> dict[str, Any]:
    ticker=str(stock.get("ticker") or "").strip().upper()
    if not ticker: return {"ticker":"", "signal":"ERROR", "final_signal":"ERROR", "status":"ERROR", "error":"Missing ticker"}
    now=datetime.now(NY); today=now.date()
    try:
        raw=flatten(_download_history(ticker,"2y","1d"),ticker)
        spy_raw=flatten(_download_history("SPY","2y","1d"),"SPY")
        # Ignore today's partial daily candle: decisions use the prior completed session only.
        raw=raw.loc[raw.index.date < today]
        spy_raw=spy_raw.loc[spy_raw.index.date < today]
        result=signal_from_completed_bars(raw,spy_raw)
        result.update({"ticker":ticker,"scanner":"BREAKOUT_VOL_FILTER_LIVE","version":"1.0.0","strategy":STRATEGY,
                       "engine":STRATEGY,"confidence":75.0,"status":"ANALYZED","benchmark":"SPY"})
        if result.get("signal") != "BUY":
            result["final_signal"]="WATCH"; result["current_price"]=_current_price(ticker); return result
        if executor is None:
            result["final_signal"]="WATCH"; result["reason"]="EXECUTOR_CONTEXT_REQUIRED"; return result
        positions=_positions(executor)
        if any(str(p.get("symbol","")).upper()==ticker and abs(float(p.get("qty") or 0))>0 for p in positions):
            result["final_signal"]="WATCH"; result["reason"]="POSITION_ALREADY_OPEN"; result["current_price"]=_current_price(ticker); return result
        if executor.orders.has_open_order_for_symbol(ticker):
            result["final_signal"]="WATCH"; result["reason"]="OPEN_ORDER_EXISTS"; result["current_price"]=_current_price(ticker); return result
        state=_state(); prior=state.setdefault("symbols",{}).get(ticker,{})
        if prior.get("last_submitted_signal_date")==result.get("signal_date"):
            result["final_signal"]="WATCH"; result["reason"]="SIGNAL_ALREADY_SUBMITTED"; result["current_price"]=_current_price(ticker); return result
        price=_current_price(ticker)
        if price is None or not np.isfinite(price) or price<=0:
            result["final_signal"]="WATCH"; result["reason"]="CURRENT_PRICE_UNAVAILABLE"; return result
        atr=float(result["atr14"]); stop=round(price-2*atr,2); target=round(price+4*atr,2)
        if stop<=0 or target<=price:
            result["final_signal"]="WATCH"; result["reason"]="INVALID_BRACKET_LEVELS"; result["current_price"]=price; return result
        risk_budget=CAPITAL_BASE*RISK_FRACTION
        qty_by_risk=math.floor(risk_budget/(price-stop))
        qty_by_capital=math.floor(CAPITAL_BASE*MAX_POSITION_FRACTION/price)
        qty=min(qty_by_risk,qty_by_capital)
        if qty<1:
            result["final_signal"]="WATCH"; result["reason"]="SIZING_BELOW_ONE_SHARE"; result["current_price"]=price; return result
        result.update({"final_signal":"BUY","current_price":price,"quantity":int(qty),"stop_loss_price":stop,
                       "take_profit_price":target,"strategy_capital":CAPITAL_BASE,"risk_budget_usd":risk_budget,
                       "estimated_notional":round(qty*price,2),"reason":";".join(k for k,v in result["checks"].items() if v)})
        return result
    except Exception as exc:
        return {"ticker":ticker,"scanner":"BREAKOUT_VOL_FILTER_LIVE","version":"1.0.0","strategy":STRATEGY,
                "signal":"ERROR","final_signal":"ERROR","status":"ERROR","error":f"{type(exc).__name__}: {exc}"}


def record_submission(result: Mapping[str, Any]) -> None:
    ticker=str(result.get("ticker") or "").upper(); signal_date=str(result.get("signal_date") or "")
    if not ticker or not signal_date: return
    data=_state(); symbols=data.setdefault("symbols",{})
    symbols[ticker]={"last_submitted_signal_date":signal_date,"entry_date":datetime.now(NY).date().isoformat(),
                     "strategy":STRATEGY,"stop_loss_price":result.get("stop_loss_price"),"take_profit_price":result.get("take_profit_price")}
    _save_state(data)


def manage_time_exit(ticker: str, *, risk_gate, executor) -> None:
    """Close strategy-owned positions after 15 sessions, after canceling bracket children."""
    ticker=str(ticker or "").upper()
    try:
        positions=[p for p in _positions(executor) if str(p.get("symbol","")).upper()==ticker and abs(float(p.get("qty") or 0))>0]
        if not positions: return
        state=_state(); item=state.setdefault("symbols",{}).get(ticker)
        if not item or not item.get("entry_date"):
            print(f"{ticker} | BREAKOUT TIME EXIT | no local entry metadata; leaving broker bracket active")
            return
        entry_date=pd.Timestamp(item["entry_date"])
        bars=flatten(_download_history(ticker,"1y","1d"),ticker)
        today=datetime.now(NY).date()
        bars=bars.loc[bars.index.date < today]
        sessions=int(((bars.index > entry_date) & (bars.index.date < today)).sum())
        if sessions < MAX_HOLD_SESSIONS: return
        qty=abs(float(positions[0].get("qty") or 0))
        px=_current_price(ticker)
        if px is None or px<=0:
            print(f"{ticker} | TIME EXIT deferred: current price unavailable; bracket remains active")
            return
        from app.broker.alpaca_trade_intent import TradeIntent
        intent=TradeIntent(symbol=ticker,side="sell",quantity=qty,order_type="market",time_in_force="day",
                           reason=f"{STRATEGY} max holding {MAX_HOLD_SESSIONS} sessions",confidence=100.0,
                           strategy=STRATEGY,signal="SELL")
        # Validate account, position, capital context and reconciliation while
        # protective bracket children are still active. Only defer the duplicate
        # SELL-order check; the executor repeats it after cancellation.
        decision=risk_gate.evaluate(intent,market_price=float(px),ignore_open_order_check=True)
        if not decision.approved:
            print(f"{ticker} | TIME EXIT RISK REJECTED; bracket remains active: {decision.summary()}")
            return
        cancelled=executor.cancel_open_orders_for_symbol(ticker)
        result=executor.execute(intent=intent,risk_decision=decision)
        print(f"{ticker} | TIME EXIT | cancelled_orders={cancelled} | {result.summary()}")
        if result.submitted:
            state["symbols"].pop(ticker,None); _save_state(state)
    except Exception as exc:
        print(f"{ticker} | BREAKOUT TIME EXIT ERROR: {type(exc).__name__}: {exc}")
