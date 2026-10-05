"""
PA V3.3.5 — CCE BUY Population Diagnostic

Research-only diagnostic.

Purpose
-------
Quantify exactly where CCE BUY/STRONG_BUY observations are lost when the
production CCE V1.7 daily population is aligned with the Alpaca historical
5-minute -> exact 20-minute intraday dataset.

Safety
------
- Does NOT modify CCE V1.7.
- Does NOT modify production files.
- Does NOT submit/cancel/modify Alpaca orders.
- Uses Alpaca historical market-data endpoint only.
- Does not print or persist Alpaca credentials.
- Does not change PA V2.1 candidates.
- Diagnostic only: no trading decision and no production integration.

The diagnostic reports the population at each stage:
1. Daily CCE V1.7 attempts.
2. CCE result status distribution.
3. CCE BUY/STRONG_BUY before PA alignment.
4. Dates where valid PA 20m data exists.
5. CCE BUY observations with a same-day closed 20m candle.
6. CCE BUY observations after exact common-period intersection.
7. Split feasibility for 60/20/20.
8. Candidate-specific PA-state matches (informational only).

This is deliberately separate from V3.3.4 incremental evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(APP_ROOT / ".env")

from app.ai.benchmark_resolver_v1_6_1 import resolve_benchmarks
from app.ai.characteristic_curve_v1_7 import (
    CurveConfig,
    _clean_ohlcv,
    analyze_dataframe,
)
from app.data.market_data import get_daily_history


VERSION = "3.3.5-R1"

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks"
TIMEFRAME = "5Min"
FEED = "sip"

# Frozen PA V2.1 candidates. This list is diagnostic only.
FROZEN_CANDIDATES = [
    {
        "ticker": "SOXL",
        "group": "PA-01",
        "state": "LARGE_BEAR",
        "horizon": "120m",
        "class": "STRONG",
    },
    {
        "ticker": "NVDA",
        "group": "PA-06",
        "state": "ACCEPT_BELOW_PREV_LOW",
        "horizon": "240m",
        "class": "MODERATE",
    },
    {
        "ticker": "SOXL",
        "group": "PA-01",
        "state": "LARGE_BEAR",
        "horizon": "40m",
        "class": "MODERATE",
    },
]

CCE_BUY_SIGNALS = {"BUY", "STRONG_BUY"}
CCE_STATUSES = {
    "INSUFFICIENT_HISTORY",
    "WATCH",
    "SELL",
    "REDUCE",
    "BUY",
    "STRONG_BUY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SOXL", "NVDA"])
    p.add_argument("--period", default="5y")
    p.add_argument("--output", default="price_action_pa_v3_3_5_cce_buy_population_diagnostic_results.json")
    p.add_argument("--min-split-observations", type=int, default=20)
    return p.parse_args()


def require_credentials() -> Tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in app/.env"
        )
    return key, secret


def parse_alpaca_bar(item: dict) -> dict:
    # Alpaca stock bar fields: t,o,h,l,c,v,vw,n
    return {
        "timestamp": pd.Timestamp(item["t"], tz="UTC"),
        "Open": float(item["o"]),
        "High": float(item["h"]),
        "Low": float(item["l"]),
        "Close": float(item["c"]),
        "Volume": float(item["v"]),
    }


def alpaca_5m_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    key: str,
    secret: str,
) -> pd.DataFrame:
    url = f"{ALPACA_DATA_URL}/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    rows: List[dict] = []
    token: Optional[str] = None

    while True:
        params = {
            "timeframe": TIMEFRAME,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": FEED,
            "limit": 10000,
            "sort": "asc",
        }
        if token:
            params["page_token"] = token

        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(
                f"Alpaca historical data failed for {ticker}: "
                f"HTTP {r.status_code} {r.text[:300]}"
            )

        payload = r.json()
        for item in payload.get("bars", []):
            rows.append(parse_alpaca_bar(item))

        token = payload.get("next_page_token")
        if not token:
            break

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_alpaca_window(
    ticker: str,
    period: str,
    key: str,
    secret: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Discover the earliest SIP 5m date permitted by the current subscription.

    Alpaca may reject a broad historical query with HTTP 403:
      "subscription does not permit querying recent SIP data"

    Therefore R1 does not assume that a 5-year request can be accepted.
    It probes progressively smaller recent windows until one is accepted,
    then uses the earliest timestamp returned as the effective boundary.

    The probe itself is data-only and never touches trading endpoints.
    """
    now = pd.Timestamp.now(tz="UTC")
    requested_start = now - pd.Timedelta(days=365 * 5)

    # Recent-window probes. A successful probe gives us a valid API request
    # shape and an approximate accessible boundary. We then request from a
    # conservative boundary derived from the successful probe.
    probe_days = [30, 15, 7, 3, 1]
    probe_results = []

    accepted = None
    last_403 = None

    for days in probe_days:
        probe_start = now - pd.Timedelta(days=days)
        try:
            df = alpaca_5m_history(ticker, probe_start, now, key, secret)
            probe_results.append({
                "days": days,
                "status": "OK",
                "rows": int(len(df)),
                "first": df.index.min().isoformat() if not df.empty else None,
                "last": df.index.max().isoformat() if not df.empty else None,
            })
            if not df.empty:
                accepted = (days, df)
                break
        except RuntimeError as exc:
            msg = str(exc)
            if "HTTP 403" in msg and "recent SIP" in msg:
                last_403 = msg
                probe_results.append({
                    "days": days,
                    "status": "403_RECENT_SIP",
                })
                continue
            raise

    if accepted is None:
        detail = last_403 or "No accepted Alpaca SIP 5m probe window."
        raise RuntimeError(
            f"Unable to discover Alpaca SIP historical boundary for {ticker}: {detail}"
        )

    probe_days_used, probe_df = accepted
    probe_first = probe_df.index.min()

    # The returned first timestamp is the authoritative accessible boundary
    # for this account. Start slightly before it to avoid an edge truncation;
    # the server will simply return the permitted data.
    effective_start = probe_first - pd.Timedelta(days=1)

    try:
        full = alpaca_5m_history(
            ticker,
            effective_start,
            now,
            key,
            secret,
        )
    except RuntimeError as exc:
        # If the one-day cushion itself triggers the subscription boundary,
        # retry exactly from the server-observed first timestamp.
        if "HTTP 403" in str(exc) and "recent SIP" in str(exc):
            full = alpaca_5m_history(
                ticker,
                probe_first,
                now,
                key,
                secret,
            )
        else:
            raise

    if full.empty:
        raise RuntimeError(
            f"Alpaca SIP returned no usable 5m bars for {ticker} "
            f"after adaptive boundary discovery."
        )

    meta = {
        "requested_start": requested_start.isoformat(),
        "requested_end": now.isoformat(),
        "feed": FEED,
        "timeframe": TIMEFRAME,
        "adaptive_boundary": True,
        "probe_windows_days": probe_days,
        "probe_days_used": probe_days_used,
        "probe_results": probe_results,
        "probe_first_timestamp": probe_first.isoformat(),
        "effective_start": full.index.min().isoformat(),
        "effective_end": full.index.max().isoformat(),
        "rows": int(len(full)),
        "boundary_discovery_method": "403_recent_sip_progressive_probe",
    }
    return full, meta


def build_20m_exact(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "session_date"]
        )

    x = df5.copy().sort_index()
    x = x.tz_convert("America/New_York")
    x = x.between_time("09:30", "16:00", inclusive="left")

    rows = []
    for session_date, day in x.groupby(x.index.date):
        day = day.sort_index()
        for i in range(0, len(day), 4):
            chunk = day.iloc[i:i + 4]
            if len(chunk) != 4:
                continue

            # Exact 5-minute cadence. No filling and no synthetic prices.
            diffs = chunk.index.to_series().diff().dropna()
            if not (diffs == pd.Timedelta(minutes=5)).all():
                continue

            start = chunk.index[0]
            end = chunk.index[-1] + pd.Timedelta(minutes=5)

            if start.hour == 9 and start.minute == 30:
                pass
            # Buckets are anchored at 09:30 by construction.
            if end > pd.Timestamp(
                f"{session_date} 16:00", tz="America/New_York"
            ):
                continue

            rows.append(
                {
                    "timestamp": end,
                    "Open": float(chunk["Open"].iloc[0]),
                    "High": float(chunk["High"].max()),
                    "Low": float(chunk["Low"].min()),
                    "Close": float(chunk["Close"].iloc[-1]),
                    "Volume": float(chunk["Volume"].sum()),
                    "session_date": str(session_date),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "session_date"]
        )

    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out


def classify_pa01_large_bear(candle: pd.Series) -> str:
    rng = float(candle["High"] - candle["Low"])
    if rng <= 0:
        return "FLAT"

    body_ratio = abs(float(candle["Close"] - candle["Open"])) / rng
    if body_ratio < 0.25:
        return "FLAT"

    if candle["Close"] > candle["Open"]:
        return "LARGE_BULL" if body_ratio >= 0.65 else (
            "MEDIUM_BULL" if body_ratio >= 0.25 else "SMALL_BULL"
        )

    return "LARGE_BEAR" if body_ratio >= 0.65 else (
        "MEDIUM_BEAR" if body_ratio >= 0.25 else "SMALL_BEAR"
    )


def classify_pa06(candle: pd.Series, prev: Optional[pd.Series]) -> str:
    if prev is None:
        return "NO_PREVIOUS"

    if candle["High"] > prev["High"] and candle["Low"] < prev["Low"]:
        # PA V1.2 differentiates outside bars elsewhere; this diagnostic
        # only needs the frozen candidate's acceptance/rejection state.
        if candle["Close"] > prev["High"]:
            return "ACCEPT_ABOVE_PREV_HIGH"
        if candle["Close"] < prev["Low"]:
            return "ACCEPT_BELOW_PREV_LOW"
        return "INSIDE_PREVIOUS_RANGE"

    if candle["Close"] < prev["Low"]:
        return "ACCEPT_BELOW_PREV_LOW"
    if candle["Close"] > prev["High"]:
        return "ACCEPT_ABOVE_PREV_HIGH"
    if candle["High"] > prev["High"] and candle["Close"] <= prev["High"]:
        return "REJECT_PREV_HIGH"
    if candle["Low"] < prev["Low"] and candle["Close"] >= prev["Low"]:
        return "REJECT_PREV_LOW"
    return "INSIDE_PREVIOUS_RANGE"


def candidate_state_counts(
    ticker: str, pa20: pd.DataFrame
) -> Dict[str, dict]:
    candidates = [x for x in FROZEN_CANDIDATES if x["ticker"] == ticker]
    result = {}

    for c in candidates:
        mask = pd.Series(False, index=pa20.index)

        if c["group"] == "PA-01" and c["state"] == "LARGE_BEAR":
            states = pa20.apply(classify_pa01_large_bear, axis=1)
            mask = states.eq("LARGE_BEAR")
        elif c["group"] == "PA-06":
            states = []
            prev = None
            for _, row in pa20.iterrows():
                states.append(classify_pa06(row, prev))
                prev = row
            states = pd.Series(states, index=pa20.index)
            mask = states.eq(c["state"])

        result[f'{c["group"]}:{c["state"]}:{c["horizon"]}'] = {
            "candidate": c,
            "candidate_candle_count": int(mask.sum()),
            "candidate_first_timestamp": (
                pa20.index[mask][0].isoformat() if mask.any() else None
            ),
            "candidate_last_timestamp": (
                pa20.index[mask][-1].isoformat() if mask.any() else None
            ),
        }

    return result


def run_cce_population(ticker: str, period: str) -> Tuple[List[dict], dict]:
    cfg = CurveConfig(period=period)

    daily_raw = get_daily_history(
        ticker,
        period=cfg.period,
        auto_adjust=True,
        actions=False,
        group_by="column",
    )
    daily_df = _clean_ohlcv(daily_raw)

    if daily_df.empty:
        raise RuntimeError(f"No usable daily OHLCV after CCE normalization: {ticker}")

    resolution = resolve_benchmarks(
        ticker,
        benchmark=None,
        period=cfg.period,
        asset_df=daily_df,
        return_selected_data=True,
    )
    benchmark_df = resolution.pop("_selected_benchmark_data", None)

    rows = []
    failures = Counter()

    for ts in daily_df.index:
        hist = daily_df.loc[:ts].copy()

        # Reproduce CCE's causal point-in-time analysis as closely as possible
        # through the production public dataframe API. Benchmark data is
        # similarly truncated when available.
        bdf = None
        if benchmark_df is not None and not benchmark_df.empty:
            bdf = benchmark_df.loc[benchmark_df.index <= ts].copy()

        try:
            result = analyze_dataframe(
                ticker=ticker,
                df=hist,
                benchmark_df=bdf,
                cfg=cfg,
                benchmark_resolution=resolution,
            )
            signal = result.get("signal")
            rows.append(
                {
                    "date": pd.Timestamp(ts).date().isoformat(),
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "signal": signal,
                    "score": result.get("score"),
                    "confidence": result.get("confidence"),
                }
            )
        except Exception as exc:
            failures[type(exc).__name__] += 1
            rows.append(
                {
                    "date": pd.Timestamp(ts).date().isoformat(),
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "signal": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )

    return rows, {
        "daily_rows": int(len(daily_df)),
        "cce_failures": dict(failures),
        "benchmark_selected": resolution.get("selected_benchmark"),
        "benchmark_resolution_status": resolution.get("status"),
    }


def diagnose_alignment(
    cce_rows: List[dict],
    pa20: pd.DataFrame,
    min_split: int,
) -> dict:
    cce_df = pd.DataFrame(cce_rows)
    if cce_df.empty:
        return {"status": "EMPTY"}

    cce_df["date"] = pd.to_datetime(cce_df["date"]).dt.date

    signal_counts = cce_df["signal"].value_counts().to_dict()
    buy = cce_df[cce_df["signal"].isin(CCE_BUY_SIGNALS)].copy()

    pa_dates = set(pa20["session_date"].astype(str).unique()) if not pa20.empty else set()
    buy["has_pa_same_day"] = buy["date"].astype(str).isin(pa_dates)

    # For a daily CCE observation, use the latest closed 20m candle of the
    # same session. This is the alignment used by the existing V3 experiment.
    latest_by_date = {}
    if not pa20.empty:
        for session_date, day in pa20.groupby("session_date"):
            day = day.sort_index()
            latest_by_date[str(session_date)] = day.iloc[-1]

    buy["aligned_pa"] = buy["date"].astype(str).map(latest_by_date)
    aligned = buy[buy["aligned_pa"].notna()].copy()

    common_start = None
    common_end = None
    if not pa20.empty:
        common_start = pa20.index.min().date().isoformat()
        common_end = pa20.index.max().date().isoformat()

    split_info = {}
    n = len(aligned)
    if n >= 3 * min_split:
        train_n = int(np.floor(n * 0.60))
        val_n = int(np.floor(n * 0.20))
        test_n = n - train_n - val_n
        split_info = {
            "status": "FEASIBLE",
            "total": n,
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "min_required_per_split": min_split,
        }
    else:
        split_info = {
            "status": "INSUFFICIENT",
            "total": n,
            "train": int(np.floor(n * 0.60)),
            "val": int(np.floor(n * 0.20)),
            "test": n - int(np.floor(n * 0.60)) - int(np.floor(n * 0.20)),
            "min_required_per_split": min_split,
            "minimum_total_required": 3 * min_split,
        }

    return {
        "daily_cce_rows": int(len(cce_df)),
        "cce_signal_counts": signal_counts,
        "cce_buy_before_alignment": int(len(buy)),
        "pa20_candle_count": int(len(pa20)),
        "pa20_session_count": int(pa20["session_date"].nunique()) if not pa20.empty else 0,
        "pa20_first_session": common_start,
        "pa20_last_session": common_end,
        "cce_buy_with_same_day_pa": int(buy["has_pa_same_day"].sum()),
        "cce_buy_aligned_with_latest_same_day_20m": int(len(aligned)),
        "alignment_loss": int(len(buy) - len(aligned)),
        "aligned_date_first": (
            aligned["date"].min().isoformat() if not aligned.empty else None
        ),
        "aligned_date_last": (
            aligned["date"].max().isoformat() if not aligned.empty else None
        ),
        "split": split_info,
    }


def main() -> int:
    args = parse_args()
    key, secret = require_credentials()

    print(f"PA V{VERSION} — CCE BUY POPULATION DIAGNOSTIC")
    print("Research-only | no orders | no production changes")
    print(f"Tickers: {', '.join(args.tickers)}")
    print()

    output = {
        "version": VERSION,
        "production_cce_version": "1.7",
        "period": args.period,
        "tickers": args.tickers,
        "safety": {
            "research_only": True,
            "production_modified": False,
            "orders_submitted": False,
            "alpaca_trading_endpoints_used": False,
            "frozen_pa_v21_candidates": True,
        },
        "results": [],
    }

    for ticker in args.tickers:
        print(f"=== {ticker} ===")

        try:
            pa5, pa_meta = get_alpaca_window(ticker, args.period, key, secret)
            print(f"Alpaca 5m rows: {len(pa5):,}")
            if pa_meta.get("adaptive_boundary"):
                print("Adaptive SIP historical boundary detected.")
            print(f"Alpaca effective start: {pa_meta['effective_start']}")
            print(f"Alpaca effective end:   {pa_meta['effective_end']}")

            pa20 = build_20m_exact(pa5)
            print(f"Exact 20m complete candles: {len(pa20):,}")

            cce_rows, cce_meta = run_cce_population(ticker, args.period)
            print(f"Daily CCE rows: {cce_meta['daily_rows']:,}")

            diagnosis = diagnose_alignment(
                cce_rows=cce_rows,
                pa20=pa20,
                min_split=args.min_split_observations,
            )

            candidates = candidate_state_counts(ticker, pa20)

            counts = diagnosis["cce_signal_counts"]
            print(f"CCE signals: {counts}")
            print(
                "CCE BUY/STRONG_BUY before alignment: "
                f"{diagnosis['cce_buy_before_alignment']}"
            )
            print(
                "CCE BUY with same-day PA: "
                f"{diagnosis['cce_buy_with_same_day_pa']}"
            )
            print(
                "CCE BUY aligned to latest same-day 20m: "
                f"{diagnosis['cce_buy_aligned_with_latest_same_day_20m']}"
            )
            print(
                "Split status: "
                f"{diagnosis['split']['status']} "
                f"(N={diagnosis['split']['total']})"
            )

            result = {
                "ticker": ticker,
                "alpaca": pa_meta,
                "cce": cce_meta,
                "diagnostic": diagnosis,
                "frozen_candidate_pa_counts": candidates,
            }
            output["results"].append(result)

            print()

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            output["results"].append(
                {
                    "ticker": ticker,
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            print()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"JSON written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
