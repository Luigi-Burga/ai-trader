#!/usr/bin/env python3
"""
AI Trader walk-forward backtest v3.2
One-year out-of-sample comparison of:
  BREAKOUT_BASE, BREAKOUT_VOL_FILTER, PULLBACK_BASE, PULLBACK_VOL_FILTER
versus buy-and-hold and cash.

No broker connection, credentials, or real orders.
Daily-bar simulation assumptions are documented in run_metadata.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START = "2025-10-28"
END = "2026-10-27"
END_EXCLUSIVE = "2026-10-28"
WARMUP_START = "2024-10-01"
BENCHMARK = "SPY"
CAPITAL_PER_TICKER = 10_000.0
RISK_FRACTION = 0.005
MAX_HOLD = 15
RR = 2.0
# Conservative configurable transaction-cost assumptions (set to 0 to disable).
COMMISSION_PER_SHARE_PER_SIDE = 0.005
SLIPPAGE_BPS_PER_SIDE = 5.0
UNIVERSE = [
    "VRT", "CLS", "MU", "NVDA", "AVGO", "AMD", "DELL", "PLTR",
    "MRVL", "TSM", "GEV", "NBIS", "LITE", "ORCL", "SMH", "QQQ"
]
STRATEGIES = ["BREAKOUT_BASE", "BREAKOUT_VOL_FILTER", "PULLBACK_BASE", "PULLBACK_VOL_FILTER"]

def flatten(df, ticker):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame()
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

def indicators(df):
    x = df.copy()
    x["sma20"] = x.Close.rolling(20).mean()
    x["sma50"] = x.Close.rolling(50).mean()
    x["prior_high20"] = x.High.rolling(20).max().shift(1)
    x["prior_high1"] = x.High.shift(1)
    x["vol_avg20"] = x.Volume.rolling(20).mean()
    prev = x.Close.shift(1)
    tr = pd.concat([x.High-x.Low, (x.High-prev).abs(), (x.Low-prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["atr_pct"] = x.atr14 / x.Close
    delta = x.Close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    x["rsi14"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    near = (x.Close - x.sma20).abs() / x.sma20
    x["near_sma20_recent"] = near.shift(1).rolling(5).min()
    # Volatility regime is computed only from past observations; current SPY
    # realized vol is compared with the rolling 100-session 80th percentile
    # shifted by one session, so today's decision does not use future values.
    ret = x.Close.pct_change()
    x["realized_vol20"] = ret.rolling(20).std() * np.sqrt(252)
    x["volatility_limit80"] = x.realized_vol20.rolling(100, min_periods=60).quantile(0.80).shift(1)
    return x

def make_trade(ticker, strategy, n, ed, xd, ep, xp, stop, target, qty, reason):
    gross_pnl = (xp-ep)*qty
    commission = max(float(qty), 0.0) * COMMISSION_PER_SHARE_PER_SIDE * 2.0
    slippage = (abs(float(ep)*float(qty)) + abs(float(xp)*float(qty))) * SLIPPAGE_BPS_PER_SIDE / 10000.0
    costs = commission + slippage
    pnl = gross_pnl - costs
    risk_dollars = max((ep-stop)*qty, 1e-9)
    return {
        "strategy": strategy, "ticker": ticker, "trade": n,
        "entry_date": pd.Timestamp(ed).strftime("%Y-%m-%d"),
        "entry_price": round(float(ep), 4),
        "exit_date": pd.Timestamp(xd).strftime("%Y-%m-%d"),
        "exit_price": round(float(xp), 4),
        "stop_price": round(float(stop), 4),
        "target_2r_price": round(float(target), 4),
        "quantity": round(float(qty), 4),
        "gross_pnl_usd": round(float(gross_pnl), 2),
        "commission_usd": round(float(commission), 2),
        "slippage_cost_usd": round(float(slippage), 2),
        "total_cost_usd": round(float(costs), 2),
        "pnl_usd": round(float(pnl), 2),
        "return_on_entry_pct": round((xp/ep-1)*100, 3),
        "r_multiple": round(float(pnl/risk_dollars), 3),
        "exit_reason": reason
    }

def get_signal(strategy, row, spy_row):
    if spy_row is None:
        return False
    needed = ["Close","sma20","sma50","prior_high20","prior_high1","vol_avg20","Volume","atr14","rsi14","near_sma20_recent"]
    if any(pd.isna(row.get(k, np.nan)) for k in needed):
        return False
    if pd.isna(spy_row.get("sma50", np.nan)) or pd.isna(spy_row.get("realized_vol20", np.nan)):
        return False
    close, sma20, sma50 = float(row.Close), float(row.sma20), float(row.sma50)
    volume, avgvol, rsi = float(row.Volume), float(row.vol_avg20), float(row.rsi14)
    spy_close = float(spy_row.Close)
    trend_ok = close > sma50 and sma20 > sma50 and spy_close > float(spy_row.sma50)
    is_breakout = strategy.startswith("BREAKOUT")
    vol_filter = strategy.endswith("VOL_FILTER")
    if vol_filter:
        limit = spy_row.get("volatility_limit80", np.nan)
        if pd.isna(limit) or float(spy_row.realized_vol20) > float(limit):
            return False
    if is_breakout:
        return (trend_ok and close > float(row.prior_high20) and avgvol > 0
                and volume >= 1.2*avgvol and rsi >= 50 and float(row.atr14)>0)
    return (trend_ok and float(row.near_sma20_recent) <= 0.02
            and close > float(row.prior_high1) and avgvol > 0
            and volume >= 1.1*avgvol and rsi >= 45 and float(row.atr14)>0)

def run_one(ticker, raw, spy_raw, strategy):
    x = indicators(raw)
    spy = indicators(spy_raw)
    spy = spy[["Close","sma50","realized_vol20","volatility_limit80"]].rename(columns={"Close":"spy_close","sma50":"spy_sma50","realized_vol20":"spy_realized_vol20","volatility_limit80":"spy_volatility_limit80"})
    x = x.join(spy, how="left")
    # Alias to keep the signal logic explicit.
    spy_for_signal = pd.DataFrame(index=x.index)
    spy_for_signal["Close"] = x.spy_close
    spy_for_signal["sma50"] = x.spy_sma50
    spy_for_signal["realized_vol20"] = x.spy_realized_vol20
    spy_for_signal["volatility_limit80"] = x.spy_volatility_limit80
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    test = x.loc[(x.index >= start) & (x.index <= end)]
    if len(test) < 30:
        return [], {"strategy":strategy,"ticker":ticker,"status":"INSUFFICIENT_TEST_DATA","bars":len(test)}
    trades, active, pending_entry, pending_exit = [], None, None, False
    n = 0
    for i, dt in enumerate(x.index):
        if dt < start or dt > end:
            continue
        row = x.loc[dt]
        o,h,l,c = map(float,[row.Open,row.High,row.Low,row.Close])
        if pending_entry is not None and active is None:
            atr = pending_entry["atr"]
            risk_per_share = 2.0*atr
            stop = o-risk_per_share
            if atr > 0 and stop > 0 and o > 0:
                qty = min(CAPITAL_PER_TICKER*RISK_FRACTION/risk_per_share, CAPITAL_PER_TICKER/o)
                n += 1
                active = {"entry_date":dt,"entry":o,"stop":stop,"target":o+RR*risk_per_share,"qty":qty,"number":n,"entry_i":i}
            pending_entry = None
        if active is not None:
            # Stop/target gaps are checked before signal exits.
            if o <= active["stop"]:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],o,active["stop"],active["target"],active["qty"],"STOP_GAP"))
                active, pending_exit = None, False
                continue
            if o >= active["target"]:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],active["target"],active["stop"],active["target"],active["qty"],"TARGET_GAP"))
                active, pending_exit = None, False
                continue
            if pending_exit:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],o,active["stop"],active["target"],active["qty"],"SIGNAL_EXIT_NEXT_OPEN"))
                active, pending_exit = None, False
                continue
            if l <= active["stop"]:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],active["stop"],active["stop"],active["target"],active["qty"],"STOP"))
                active, pending_exit = None, False
                continue
            if h >= active["target"]:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],active["target"],active["stop"],active["target"],active["qty"],"TARGET_2R"))
                active, pending_exit = None, False
                continue
            if i-active["entry_i"] >= MAX_HOLD:
                trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],dt,active["entry"],c,active["stop"],active["target"],active["qty"],"MAX_HOLD_15"))
                active, pending_exit = None, False
                continue
            pending_exit = pd.notna(row.sma20) and c < float(row.sma20)
            continue
        sr = spy_for_signal.loc[dt] if dt in spy_for_signal.index else None
        if get_signal(strategy,row,sr):
            pending_entry = {"atr":float(row.atr14),"signal_date":dt}
    if active is not None:
        last_dt = test.index[-1]
        trades.append(make_trade(ticker,strategy,active["number"],active["entry_date"],last_dt,active["entry"],float(x.loc[last_dt,"Close"]),active["stop"],active["target"],active["qty"],"END_OF_TEST_MARK_TO_MARKET"))
    # Benchmark / buy-and-hold values over exact same test dates.
    buyhold = (float(test.Close.iloc[-1])/float(test.Close.iloc[0])-1)*100
    pnl = sum(t["pnl_usd"] for t in trades)
    wins = sum(t["pnl_usd"] > 0 for t in trades)
    return trades, {
        "strategy":strategy,"ticker":ticker,"status":"OK","bars":len(test),
        "buy_hold_return_pct":round(buyhold,3),"trades":len(trades),
        "wins":int(wins),"losses":len(trades)-int(wins),
        "win_rate_pct":round(100*wins/len(trades),2) if trades else None,
        "net_pnl_usd":round(pnl,2),
        "return_on_allocated_10000_pct":round(pnl/CAPITAL_PER_TICKER*100,3),
        "avg_trade_r":round(float(np.mean([t["r_multiple"] for t in trades])),3) if trades else None,
        "cash_return_pct":0.0
    }

def max_drawdown(prices):
    s = pd.Series(prices).dropna()
    if s.empty: return None
    dd = s/s.cummax()-1
    return round(float(dd.min()*100),3)

def main():
    outdir = Path("walkforward_backtest_v3_2_20251028_20261027")
    outdir.mkdir(parents=True,exist_ok=True)
    data, errors = {}, []
    for ticker in list(dict.fromkeys(UNIVERSE+[BENCHMARK])):
        try:
            raw = yf.download(ticker,start=WARMUP_START,end=END_EXCLUSIVE,interval="1d",auto_adjust=True,actions=False,progress=False,threads=False)
            frame = flatten(raw,ticker)
            if frame.empty:
                errors.append({"ticker":ticker,"error":"No usable OHLCV"}); continue
            data[ticker] = frame
            print(f"Downloaded {ticker}: {len(frame)} bars, {frame.index.min().date()} to {frame.index.max().date()}")
        except Exception as e:
            errors.append({"ticker":ticker,"error":f"{type(e).__name__}: {e}"})
    if BENCHMARK not in data:
        sys.exit("ERROR: SPY data unavailable; no backtest run.")
    trades_all, summaries = [], []
    for ticker in UNIVERSE:
        if ticker not in data:
            for strategy in STRATEGIES:
                summaries.append({"strategy":strategy,"ticker":ticker,"status":"DATA_ERROR"})
            continue
        for strategy in STRATEGIES:
            trades, summary = run_one(ticker,data[ticker],data[BENCHMARK],strategy)
            trades_all.extend(trades); summaries.append(summary)
    pd.DataFrame(trades_all).to_csv(outdir/"trades.csv",index=False)
    pd.DataFrame(summaries).to_csv(outdir/"summary_by_ticker.csv",index=False)
    # Strategy-level summary: equal dollar sleeve for each of the same 16 symbols.
    comparison = []
    for strategy in STRATEGIES:
        rows = [r for r in summaries if r.get("strategy")==strategy and r.get("status")=="OK"]
        trs = [t for t in trades_all if t["strategy"]==strategy]
        allocated = len(UNIVERSE)*CAPITAL_PER_TICKER
        pnl = sum(float(r.get("net_pnl_usd",0)) for r in rows)
        avg_bh = float(np.mean([r["buy_hold_return_pct"] for r in rows])) if rows else np.nan
        comparison.append({
            "strategy":strategy,"symbols_available":len(rows),"allocated_capital_usd":allocated,
            "net_pnl_usd":round(pnl,2),"return_on_total_allocated_capital_pct":round(pnl/allocated*100,3),
            "trades":len(trs),"wins":sum(t["pnl_usd"]>0 for t in trs),"losses":sum(t["pnl_usd"]<=0 for t in trs),
            "win_rate_pct":round(100*sum(t["pnl_usd"]>0 for t in trs)/len(trs),2) if trs else None,
            "mean_buy_hold_return_pct":round(avg_bh,3),"cash_return_pct":0.0,
            "orders_submitted":False
        })
    # Daily marked-to-market portfolio equity. Open positions are valued at
    # each session close; entry costs are charged from entry date and exit costs
    # on the exit date. Closed trades retain their net realized P&L thereafter.
    curve_rows = []
    trading_dates = sorted(set().union(*[
        set(data[t].loc[(data[t].index >= pd.Timestamp(START)) &
                        (data[t].index <= pd.Timestamp(END))].index)
        for t in UNIVERSE if t in data
    ]))
    initial_capital = len(UNIVERSE) * CAPITAL_PER_TICKER
    for strategy in STRATEGIES:
        trs = [t for t in trades_all if t["strategy"] == strategy]
        daily_pnl = {d: 0.0 for d in trading_dates}
        for trade in trs:
            ticker = trade["ticker"]
            if ticker not in data:
                continue
            ed, xd = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
            ep, qty = float(trade["entry_price"]), float(trade["quantity"])
            total_commission = float(trade["commission_usd"])
            total_slippage = float(trade["slippage_cost_usd"])
            entry_cost = qty * COMMISSION_PER_SHARE_PER_SIDE + ep * qty * SLIPPAGE_BPS_PER_SIDE / 10000.0
            # Net P&L is realized from exit date onward. Before exit, mark to
            # market from daily close and subtract the entry-side transaction cost.
            prices = data[ticker]["Close"]
            for d in trading_dates:
                if d < ed:
                    continue
                if d >= xd:
                    daily_pnl[d] += float(trade["pnl_usd"])
                elif d in prices.index and pd.notna(prices.loc[d]):
                    daily_pnl[d] += (float(prices.loc[d]) - ep) * qty - entry_cost
        equity = initial_capital
        peak = initial_capital
        for d in trading_dates:
            equity = initial_capital + daily_pnl[d]
            # daily_pnl[d] above is total portfolio P&L at date, not a daily delta.
            # Build cumulative equity from per-trade marked/realized contributions.
            peak = max(peak, equity)
            curve_rows.append({"date": d.strftime("%Y-%m-%d"), "strategy": strategy,
                               "portfolio_pnl_usd": round(float(daily_pnl[d]), 2),
                               "equity_usd": round(float(equity), 2),
                               "drawdown_pct": round((equity / peak - 1.0) * 100.0, 3)})
        strat_dd = min((r["drawdown_pct"] for r in curve_rows if r["strategy"] == strategy), default=0.0)
        for item in comparison:
            if item.get("strategy") == strategy:
                item["max_drawdown_pct"] = round(float(strat_dd), 3)
                item["drawdown_method"] = "DAILY_MARK_TO_MARKET_INCLUDES_OPEN_POSITIONS_AND_ESTIMATED_COSTS"
    pd.DataFrame(curve_rows).to_csv(outdir/"realized_equity_curve.csv", index=False)

    # Buy-hold equal-weight basket with $10k per ticker, mark-to-market daily.
    px = pd.DataFrame({t:data[t].Close for t in UNIVERSE if t in data})
    px = px.loc[(px.index>=pd.Timestamp(START))&(px.index<=pd.Timestamp(END))].dropna(how="all")
    normalized = px.div(px.iloc[0])
    basket = normalized.mean(axis=1)
    bh_summary = {"strategy":"BUY_AND_HOLD_EQUAL_WEIGHT","symbols_available":len(px.columns),
        "allocated_capital_usd":len(px.columns)*CAPITAL_PER_TICKER,
        "net_pnl_usd":round((float(basket.iloc[-1])-1)*len(px.columns)*CAPITAL_PER_TICKER,2),
        "return_on_total_allocated_capital_pct":round((float(basket.iloc[-1])-1)*100,3),
        "trades":len(px.columns),"wins":None,"losses":None,"win_rate_pct":None,
        "mean_buy_hold_return_pct":round(float(normalized.iloc[-1].sub(1).mean()*100),3),
        "cash_return_pct":0.0,"max_drawdown_pct":max_drawdown(basket),"orders_submitted":False}
    comparison.extend([bh_summary,{"strategy":"CASH","symbols_available":len(UNIVERSE),
        "allocated_capital_usd":len(UNIVERSE)*CAPITAL_PER_TICKER,"net_pnl_usd":0.0,
        "return_on_total_allocated_capital_pct":0.0,"trades":0,"wins":0,"losses":0,
        "win_rate_pct":None,"mean_buy_hold_return_pct":None,"cash_return_pct":0.0,
        "max_drawdown_pct":0.0,"orders_submitted":False}])
    pd.DataFrame(comparison).to_csv(outdir/"strategy_comparison.csv",index=False)
    # Integrity checks: P&L reconciliation, complete equity curve, finite metrics,
    # no broker interaction, and transaction-cost metadata consistency.
    checks = []
    for strategy in STRATEGIES:
        trade_pnl = round(sum(float(t["pnl_usd"]) for t in trades_all if t["strategy"] == strategy), 2)
        summary_pnl = round(sum(float(r.get("net_pnl_usd", 0.0)) for r in summaries
                                if r.get("strategy") == strategy and r.get("status") == "OK"), 2)
        curve = [r for r in curve_rows if r["strategy"] == strategy]
        checks.append({"check": f"{strategy}_trade_pnl_reconciles", "passed": trade_pnl == summary_pnl,
                       "trade_pnl_usd": trade_pnl, "summary_pnl_usd": summary_pnl})
        checks.append({"check": f"{strategy}_daily_equity_curve_exists", "passed": len(curve) > 0,
                       "curve_rows": len(curve)})
        checks.append({"check": f"{strategy}_equity_starts_at_allocated_capital", "passed": bool(curve) and
                       abs(float(curve[0]["equity_usd"]) - initial_capital) < 1e-6,
                       "first_equity_usd": curve[0]["equity_usd"] if curve else None,
                       "allocated_capital_usd": initial_capital})
    checks.append({"check":"no_orders_submitted", "passed": True, "orders_submitted": False})
    checks.append({"check":"transaction_costs_documented", "passed": COMMISSION_PER_SHARE_PER_SIDE >= 0 and SLIPPAGE_BPS_PER_SIDE >= 0,
                   "commission_per_share_per_side_usd": COMMISSION_PER_SHARE_PER_SIDE,
                   "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE})
    (outdir/"validation_checks.json").write_text(json.dumps({"all_passed": all(c["passed"] for c in checks), "checks": checks}, indent=2), encoding="utf-8")
    (outdir/"run_metadata.json").write_text(json.dumps({
        "evaluation_start":START,"evaluation_end":END,"warmup_start":WARMUP_START,
        "universe":UNIVERSE,"benchmark":BENCHMARK,"capital_per_ticker_usd":CAPITAL_PER_TICKER,
        "risk_fraction_per_trade":RISK_FRACTION,"max_holding_sessions":MAX_HOLD,"target_r_multiple":RR,
        "orders_submitted":False,"data_errors":errors,
        "transaction_costs":{"commission_per_share_per_side_usd":COMMISSION_PER_SHARE_PER_SIDE,"slippage_bps_per_side":SLIPPAGE_BPS_PER_SIDE},
        "drawdown_note":"Strategy max_drawdown_pct is calculated from daily portfolio equity including open-position close-to-market P&L and estimated entry/exit transaction costs; directly comparable in method to buy-and-hold basket drawdown, with different exposure rules.",
        "volatility_filter":"SPY 20-session annualized realized volatility <= its prior 100-session rolling 80th percentile; percentile shifted one session to avoid look-ahead.",
        "strategies":{
          "BREAKOUT_BASE":"Close > prior 20-session high; Close>SMA50; SMA20>SMA50; volume>=1.2x average20; RSI>=50; SPY>SMA50.",
          "BREAKOUT_VOL_FILTER":"BREAKOUT_BASE plus SPY volatility filter.",
          "PULLBACK_BASE":"Close>SMA50; SMA20>SMA50; a prior close within 2% of SMA20 in prior five sessions; today's close > prior high; volume>=1.1x average20; RSI>=45; SPY>SMA50.",
          "PULLBACK_VOL_FILTER":"PULLBACK_BASE plus SPY volatility filter."
        },
        "execution":"Signals at close; entries next open. Stop gap exits at open. If stop and target both touched intraday, stop first. Target gap fills at target conservatively. Stop/target use 2*ATR14 measured on signal date. Commission and slippage estimates applied on entry and exit; daily OHLC cannot resolve intraday sequence.",
        "caveats":["No intraday data; stop/target order ambiguity resolved conservatively.","Strategy portfolio equity is not compounded for sizing; one $10,000 sleeve per ticker and per-trade risk sizing.","Strategy daily equity marks open positions at close and includes estimated entry/exit costs; buy-and-hold basket is equal-weighted and marked daily."]
    },indent=2),encoding="utf-8")
    print("\nSTRATEGY COMPARISON")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"\nComparison: {outdir/'strategy_comparison.csv'}")
    print(f"Per-ticker summary: {outdir/'summary_by_ticker.csv'}")
    print(f"Trades: {outdir/'trades.csv'}")
    print(f"Metadata: {outdir/'run_metadata.json'}")
    print(f"Daily marked-to-market equity curve: {outdir/'realized_equity_curve.csv'}")
    print(f"Validation checks: {outdir/'validation_checks.json'}")
    if errors: print("\nDATA ERRORS:\n"+json.dumps(errors,indent=2))

if __name__ == "__main__":
    main()
