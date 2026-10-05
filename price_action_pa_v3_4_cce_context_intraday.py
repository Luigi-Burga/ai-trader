"""
PA V3.4 — CCE Context -> Intraday PA

Research-only methodological experiment.

Question
--------
After a production CCE V1.7 BUY/STRONG_BUY context is frozen at the close
of trading day D, does the next trading session D+1 intraday Price Action
add information beyond the CCE context itself?

Design
------
CCE(D) -> frozen context -> PA observations during next session D+1.

Critical causal rule:
- CCE(D) uses information through daily close D only.
- PA(D+1, candle close) uses information through that 20m candle only.
- No D+1 information enters CCE(D).
- TRAIN/VAL/TEST are split by independent CCE context sessions, never by
  individual candles.

Frozen PA V2.1 candidates:
- SOXL PA-01 LARGE_BEAR, 40m
- SOXL PA-01 LARGE_BEAR, 120m
- NVDA PA-06 ACCEPT_BELOW_PREV_LOW, 240m

The experiment is deliberately separate from production. It does not
modify CCE, main, scanner, risk gate, executor, Alpaca orders, or Telegram.

Data:
- Alpaca historical SIP 5m
- exact complete regular-session 20m candles from 4 consecutive 5m bars
- production CCE V1.7 + its actual benchmark resolver
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


VERSION = "3.4"
PRODUCTION_CCE_VERSION = "1.7"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
ALPACA_FEED = "sip"
ALPACA_TIMEFRAME = "5Min"
REGULAR_TZ = "America/New_York"

# Frozen from PA V2.1. No candidate discovery occurs in V3.4.
FROZEN_CANDIDATES = [
    {"ticker": "SOXL", "group": "PA-01", "state": "LARGE_BEAR", "horizon_min": 40, "class": "MODERATE"},
    {"ticker": "SOXL", "group": "PA-01", "state": "LARGE_BEAR", "horizon_min": 120, "class": "STRONG"},
    {"ticker": "NVDA", "group": "PA-06", "state": "ACCEPT_BELOW_PREV_LOW", "horizon_min": 240, "class": "MODERATE"},
]

CCE_BUY_SIGNALS = {"BUY", "STRONG_BUY"}
SPLIT_RATIOS = (0.60, 0.20, 0.20)


def parse_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
ENV_FILE = APP_ROOT / ".env"


def get_credentials() -> Tuple[str, str]:
    file_env = parse_env(ENV_FILE)
    key = os.environ.get("ALPACA_API_KEY") or file_env.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or file_env.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(f"Missing Alpaca credentials in {ENV_FILE}")
    return key, secret


def _is_recent_sip_403(d: Dict[str, Any]) -> bool:
    return (
        d.get("http_status") == 403
        and "subscription does not permit querying recent sip data"
        in str(d.get("message", "")).lower()
    )


def alpaca_request_5m_window(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    key: str,
    secret: str,
) -> Tuple[str, Any]:
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    url = ALPACA_DATA_URL.format(symbol=ticker)
    rows: List[Dict[str, Any]] = []
    token: Optional[str] = None
    pages = 0

    while True:
        pages += 1
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": ALPACA_TIMEFRAME,
            "feed": ALPACA_FEED,
            "limit": 10000,
        }
        if token:
            params["page_token"] = token
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = {"raw_message": r.text.strip()}
            return "ERROR", {
                "ticker": ticker,
                "http_status": r.status_code,
                "message": body.get("message") if isinstance(body, dict) else str(body),
                "body": body,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "page": pages,
            }
        payload = r.json()
        rows.extend(payload.get("bars", []))
        token = payload.get("next_page_token")
        if not token:
            break

    return "PASS", {"bars": rows, "pages": pages}


def alpaca_get_5m_adaptive(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    key: str,
    secret: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Discover the historical SIP boundary by moving END backwards when the
    known recent-SIP subscription restriction is returned. No IEX fallback.
    """
    offsets = [0, 1, 2, 3, 5, 7, 14, 30, 60]
    probes: List[Dict[str, Any]] = []
    accepted_end: Optional[pd.Timestamp] = None

    for days in offsets:
        probe_end = end - pd.Timedelta(days=days)
        status, payload = alpaca_request_5m_window(ticker, probe_end - pd.Timedelta(days=30), probe_end, key, secret)
        if status == "PASS":
            accepted_end = probe_end
            probes.append({"end_offset_days": days, "status": "OK", "rows": len(payload["bars"])})
            break
        probes.append({
            "end_offset_days": days,
            "status": "403_RECENT_SIP" if _is_recent_sip_403(payload) else "ERROR",
            "http_status": payload.get("http_status"),
            "message": payload.get("message"),
        })
        if not _is_recent_sip_403(payload):
            raise RuntimeError(
                f"Alpaca SIP historical request failed for {ticker}: "
                f"{payload.get('http_status')} {payload.get('message')}"
            )

    if accepted_end is None:
        raise RuntimeError(f"Unable to discover Alpaca SIP historical boundary for {ticker}")

    final_start = start
    status, payload = alpaca_request_5m_window(ticker, final_start, accepted_end, key, secret)
    if status != "PASS":
        raise RuntimeError(
            f"Historical SIP retrieval failed after boundary discovery for {ticker}: "
            f"{payload.get('http_status')} {payload.get('message')}"
        )

    bars = payload["bars"]
    df = pd.DataFrame(bars).rename(columns={
        "t": "timestamp", "o": "open", "h": "high",
        "l": "low", "c": "close", "v": "volume",
    })
    if df.empty:
        raise RuntimeError(f"No Alpaca 5m bars returned for {ticker}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
          .drop_duplicates("timestamp")
          .sort_values("timestamp")
          .set_index("timestamp")
    )

    return df, {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "accepted_end": accepted_end.isoformat(),
        "boundary_discovery_method": "403_recent_sip_move_end_backward",
        "end_offsets_days": offsets,
        "probe_results": probes,
        "feed": ALPACA_FEED,
        "timeframe": ALPACA_TIMEFRAME,
        "rows": int(len(df)),
        "effective_start": df.index.min().isoformat(),
        "effective_end": df.index.max().isoformat(),
    }


def build_20m_exact(df5: pd.DataFrame) -> pd.DataFrame:
    """
    Build exact regular-session 20m candles:
    09:30-09:50 ... 15:40-16:00.

    A candle is accepted only if all 4 constituent 5m timestamps exist and
    are exactly 5 minutes apart. No fill/interpolation.
    """
    if df5.empty:
        return pd.DataFrame()

    ny = df5.tz_convert(REGULAR_TZ)
    mask = (
        (((ny.index.hour > 9) | ((ny.index.hour == 9) & (ny.index.minute >= 30))))
        & (ny.index.hour < 16)
    )
    x = ny.loc[mask].copy().sort_index()

    rows: List[Dict[str, Any]] = []
    for session_date, day in x.groupby(x.index.date):
        day = day.sort_index()
        base = pd.Timestamp(session_date).tz_localize(REGULAR_TZ) + pd.Timedelta(hours=9, minutes=30)
        for n in range(20):
            start = base + pd.Timedelta(minutes=20 * n)
            end = start + pd.Timedelta(minutes=20)
            b = day[(day.index >= start) & (day.index < end)]
            if len(b) != 4:
                continue
            if not bool((b.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()):
                continue
            rows.append({
                "timestamp": end,
                "session_date": str(session_date),
                "open": float(b["open"].iloc[0]),
                "high": float(b["high"].max()),
                "low": float(b["low"].min()),
                "close": float(b["close"].iloc[-1]),
                "volume": float(b["volume"].sum()),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out


def pa01_body_state(row: pd.Series) -> str:
    rng = float(row["high"] - row["low"])
    if rng <= 0:
        return "FLAT"
    ratio = float((row["close"] - row["open"]) / rng)
    if ratio >= 0.65:
        return "LARGE_BULL"
    if ratio >= 0.25:
        return "MEDIUM_BULL"
    if ratio > -0.25:
        return "SMALL_BULL" if ratio > 0 else "SMALL_BEAR"
    if ratio > -0.65:
        return "MEDIUM_BEAR"
    return "LARGE_BEAR"


def pa06_state(df: pd.DataFrame, i: int) -> str:
    if i == 0:
        return "UNDEFINED"
    cur = df.iloc[i]
    prev = df.iloc[i - 1]
    if cur["close"] > prev["high"]:
        return "ACCEPT_ABOVE_PREV_HIGH"
    if cur["close"] < prev["low"]:
        return "ACCEPT_BELOW_PREV_LOW"
    if cur["high"] > prev["high"] and cur["close"] <= prev["high"]:
        return "REJECT_PREV_HIGH"
    if cur["low"] < prev["low"] and cur["close"] >= prev["low"]:
        return "REJECT_PREV_LOW"
    return "INSIDE_PREVIOUS_RANGE"


def add_pa_states(df20: pd.DataFrame) -> pd.DataFrame:
    x = df20.copy()
    x["PA-01"] = [pa01_body_state(r) for _, r in x.iterrows()]
    x["PA-06"] = [pa06_state(x, i) for i in range(len(x))]
    return x


def load_production_cce():
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ai.characteristic_curve_v1_7 import CurveConfig, _clean_ohlcv, analyze_dataframe
    from app.ai.benchmark_resolver_v1_6_1 import resolve_benchmarks
    from app.data.market_data import get_daily_history
    return CurveConfig, _clean_ohlcv, analyze_dataframe, resolve_benchmarks, get_daily_history


def build_daily_cce_population(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Exact production CCE V1.7 API:
      _clean_ohlcv(daily)
      resolve_benchmarks(ticker, benchmark=None, period=..., asset_df=..., return_selected_data=True)
      analyze_dataframe(...)
    """
    CurveConfig, _clean_ohlcv, analyze_dataframe, resolve_benchmarks, get_daily_history = load_production_cce()

    raw = get_daily_history(
        ticker,
        period=period,
        auto_adjust=True,
        actions=False,
        group_by="column",
    )
    asset = _clean_ohlcv(raw)
    if asset.empty:
        raise RuntimeError(f"No usable normalized daily data for {ticker}")

    resolution = resolve_benchmarks(
        ticker,
        benchmark=None,
        period=period,
        asset_df=asset,
        return_selected_data=True,
    )
    benchmark_df = resolution.get("_selected_benchmark_data")
    benchmark_name = resolution.get("selected_benchmark")
    cfg = CurveConfig(period=period)

    rows: List[Dict[str, Any]] = []
    for ts in asset.index:
        hist = asset.loc[:ts].copy()
        bhist = None
        if benchmark_df is not None and not getattr(benchmark_df, "empty", True):
            bhist = benchmark_df.loc[:ts].copy()

        try:
            r = analyze_dataframe(
                ticker=ticker,
                df=hist,
                benchmark_df=bhist,
                cfg=cfg,
                benchmark_resolution=resolution,
            )
            rows.append({
                "date": pd.Timestamp(ts).date().isoformat(),
                "signal": str(r.get("signal", "UNKNOWN")),
                "score": r.get("score"),
                "confidence": r.get("confidence"),
                "benchmark": benchmark_name,
            })
        except Exception as exc:
            rows.append({
                "date": pd.Timestamp(ts).date().isoformat(),
                "signal": "ERROR",
                "score": None,
                "confidence": None,
                "benchmark": benchmark_name,
                "error": str(exc),
            })

    return pd.DataFrame(rows)


def next_trading_session_map(df20: pd.DataFrame) -> Dict[str, str]:
    dates = sorted(df20["session_date"].unique())
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def split_contexts(contexts: pd.DataFrame) -> pd.DataFrame:
    x = contexts.sort_values("context_date").reset_index(drop=True)
    n = len(x)
    if n == 0:
        return x
    train_n = int(math.floor(n * SPLIT_RATIOS[0]))
    val_n = int(math.floor(n * SPLIT_RATIOS[1]))
    x["split"] = "TEST"
    if train_n:
        x.loc[:train_n - 1, "split"] = "TRAIN"
    if val_n:
        x.loc[train_n:train_n + val_n - 1, "split"] = "VAL"
    return x


def forward_return_same_session(day: pd.DataFrame, idx: int, minutes: int) -> Optional[float]:
    """
    Forward close-to-close return only if the requested horizon ends inside
    the SAME trading session. This prevents a late-session observation from
    silently crossing into the following session.
    """
    t0 = day.index[idx]
    target = t0 + pd.Timedelta(minutes=minutes)
    future = day[day.index >= target]
    if future.empty:
        return None
    row = future.iloc[0]
    if str(row["session_date"]) != str(day.iloc[idx]["session_date"]):
        return None
    p0 = float(day["close"].iloc[idx])
    p1 = float(row["close"])
    if p0 == 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


def metrics(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive_pct": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "positive_pct": float((a > 0).mean() * 100.0),
    }


def session_aggregate(rows: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    if rows.empty:
        return {"sessions": 0, "mean_session_return": None, "median_session_return": None, "positive_session_pct": None}
    per_session = rows.groupby("context_date")[value_col].mean()
    return {
        "sessions": int(len(per_session)),
        "mean_session_return": float(per_session.mean()),
        "median_session_return": float(per_session.median()),
        "positive_session_pct": float((per_session > 0).mean() * 100.0),
    }


def evaluate_candidate(
    obs: pd.DataFrame,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    horizon = int(candidate["horizon_min"])
    state = candidate["state"]

    out: Dict[str, Any] = {
        **candidate,
        "comparison": "CCE_favorable_context_plus_PA_state_vs_CCE_favorable_context_other_PA",
        "horizon": f"{horizon}m",
        "splits": {},
    }

    for split in ["TRAIN", "VAL", "TEST"]:
        base = obs[obs["split"] == split]
        selected = base[base["pa_state"] == state]

        bm = metrics(base["forward_return"].dropna().tolist())
        sm = metrics(selected["forward_return"].dropna().tolist())

        base_s = session_aggregate(base.dropna(subset=["forward_return"]), "forward_return")
        selected_s = session_aggregate(selected.dropna(subset=["forward_return"]), "forward_return")

        out["splits"][split] = {
            "independent_context_sessions": int(base["context_date"].nunique()),
            "eligible_candles": int(len(base)),
            "candidate_candles": int(len(selected)),
            "cce_context_baseline_candle_level": bm,
            "pa_candidate_candle_level": sm,
            "delta_mean_pp": (
                sm["mean"] - bm["mean"]
                if sm["mean"] is not None and bm["mean"] is not None else None
            ),
            "delta_positive_pp": (
                sm["positive_pct"] - bm["positive_pct"]
                if sm["positive_pct"] is not None and bm["positive_pct"] is not None else None
            ),
            "cce_context_baseline_session_level": base_s,
            "pa_candidate_session_level": selected_s,
            "delta_session_mean_pp": (
                selected_s["mean_session_return"] - base_s["mean_session_return"]
                if selected_s["mean_session_return"] is not None and base_s["mean_session_return"] is not None
                else None
            ),
        }

    # Explicit methodological classification; not a production decision.
    test = out["splits"]["TEST"]
    test_sessions = test["independent_context_sessions"]
    dmean = test["delta_session_mean_pp"]
    dpos = test["delta_positive_pp"]
    if test_sessions < 3:
        classification = "INSUFFICIENT_INDEPENDENT_CONTEXTS"
    elif dmean is not None and dpos is not None and dmean > 0 and dpos > 0:
        classification = "POSITIVE_INCREMENTAL_EVIDENCE"
    elif dmean is not None and dpos is not None:
        classification = "NO_CLEAR_INCREMENTAL_EVIDENCE"
    else:
        classification = "INSUFFICIENT_DATA"
    out["test_classification"] = classification
    return out


def run_ticker(ticker: str, years: int, key: str, secret: str) -> Dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=365 * years)

    print(f"\n=== {ticker} ===")
    print("Downloading Alpaca 5m SIP...")
    df5, alpaca_report = alpaca_get_5m_adaptive(ticker, start, now, key, secret)
    df20 = build_20m_exact(df5)
    if df20.empty:
        raise RuntimeError(f"No exact 20m candles for {ticker}")
    df20 = add_pa_states(df20)
    print(f"5m rows: {len(df5):,}")
    print(f"20m complete candles: {len(df20):,}")

    print("Building production CCE V1.7 daily population...")
    cce = build_daily_cce_population(ticker, period="5y")
    buy = cce[cce["signal"].isin(CCE_BUY_SIGNALS)].copy()
    print(f"CCE BUY/STRONG_BUY contexts: {len(buy)}")

    session_dates = sorted(df20["session_date"].unique())
    next_map = next_trading_session_map(df20)

    contexts = []
    for _, r in buy.iterrows():
        d = str(r["date"])
        next_d = next_map.get(d)
        if not next_d:
            continue
        if next_d not in session_dates:
            continue
        contexts.append({
            "context_date": d,
            "next_session_date": next_d,
            "cce_signal": r["signal"],
            "cce_score": r.get("score"),
            "cce_confidence": r.get("confidence"),
        })

    contexts = split_contexts(pd.DataFrame(contexts))
    if contexts.empty:
        return {
            "alpaca": alpaca_report,
            "cce_buy_contexts": 0,
            "contexts": [],
            "candidates": [],
        }

    context_lookup = {
        str(r["next_session_date"]): r
        for _, r in contexts.iterrows()
    }

    candidate_defs = [c for c in FROZEN_CANDIDATES if c["ticker"] == ticker]
    candidate_results = []

    for cand in candidate_defs:
        horizon = int(cand["horizon_min"])
        rows: List[Dict[str, Any]] = []
        state_col = cand["group"]

        for session_date, day in df20.groupby("session_date", sort=True):
            if session_date not in context_lookup:
                continue
            ctx = context_lookup[session_date]
            day = day.sort_index()

            for i in range(len(day)):
                # State is known at candle close.
                state = str(day.iloc[i][state_col])
                ret = forward_return_same_session(day, i, horizon)
                if ret is None:
                    continue
                rows.append({
                    "context_date": str(ctx["context_date"]),
                    "session_date": session_date,
                    "cce_signal": str(ctx["cce_signal"]),
                    "cce_score": ctx["cce_score"],
                    "cce_confidence": ctx["cce_confidence"],
                    "split": str(ctx["split"]),
                    "candle_close": day.index[i].isoformat(),
                    "pa_group": cand["group"],
                    "pa_state": state,
                    "forward_horizon": f"{horizon}m",
                    "forward_return": float(ret),
                })

        obs = pd.DataFrame(rows)
        if obs.empty:
            evaluation = {
                **cand,
                "observations": 0,
                "independent_context_sessions": 0,
                "test_classification": "INSUFFICIENT_DATA",
                "splits": {},
            }
        else:
            evaluation = evaluate_candidate(obs, cand)
        candidate_results.append(evaluation)

    # Context inventory is saved for auditability, but no credentials.
    return {
        "alpaca": alpaca_report,
        "reconstructed_20m": {
            "rows": int(len(df20)),
            "sessions": int(df20["session_date"].nunique()),
            "first": df20.index.min().isoformat(),
            "last": df20.index.max().isoformat(),
        },
        "cce_buy_contexts_before_next_session_alignment": int(len(buy)),
        "cce_buy_contexts_with_next_session": int(len(contexts)),
        "context_split_counts": contexts["split"].value_counts().to_dict(),
        "context_inventory": contexts.to_dict(orient="records"),
        "candidates": candidate_results,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SOXL", "NVDA"])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--output", default="price_action_pa_v3_4_results.json")
    args = p.parse_args()

    key, secret = get_credentials()

    requested = [x.upper() for x in args.tickers]
    supported = sorted({c["ticker"] for c in FROZEN_CANDIDATES})
    bad = [x for x in requested if x not in supported]
    if bad:
        raise SystemExit(f"No frozen V2.1 PA candidate for: {bad}")

    result: Dict[str, Any] = {
        "version": VERSION,
        "method": "CCE Context -> Intraday PA",
        "production_cce_version": PRODUCTION_CCE_VERSION,
        "research_only": True,
        "production_modified": False,
        "orders_submitted": False,
        "alpaca_trading_endpoints_used": False,
        "causal": True,
        "lookahead": False,
        "context_rule": "CCE(D) BUY/STRONG_BUY frozen after daily close; evaluate PA only on next trading session D+1",
        "split_unit": "independent CCE context sessions",
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "baseline": "all eligible intraday candles inside the same frozen CCE favorable context",
        "candidate_source": "PA V2.1 STRONG/MODERATE frozen candidates",
        "candidates": FROZEN_CANDIDATES,
        "tickers": {},
    }

    for ticker in requested:
        result["tickers"][ticker] = run_ticker(ticker, args.years, key, secret)

    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("\nPA V3.4 COMPLETE")
    print(f"Output: {args.output}")
    for ticker, data in result["tickers"].items():
        print(
            f"{ticker}: CCE contexts={data.get('cce_buy_contexts_before_next_session_alignment', 0)}, "
            f"next-session contexts={data.get('cce_buy_contexts_with_next_session', 0)}"
        )
        for c in data.get("candidates", []):
            print(
                f"  {c.get('group')} {c.get('state')} {c.get('horizon_min')}m: "
                f"{c.get('test_classification')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
