from __future__ import annotations

"""AI Trader - Price Action Engine V1.0

Phase 1: Candle Close + Outside Bar.

Design principles
-----------------
* Feature engine only: it does not issue BUY/SELL decisions.
* No look-ahead: every feature at t uses data available through t only.
* Daily-close semantics: the current bar is considered confirmed only after
  its close is available.
* Deterministic and independently testable before integration into production.
"""

VERSION = "1.0"

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import argparse
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceActionConfig:
    atr_period: int = 14
    volume_window: int = 20
    breakout_lookback: int = 20
    min_outside_range_atr: float = 0.75
    min_close_strength: float = 0.70
    strong_volume_ratio: float = 1.20


def _f(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        selected = {}
        for field in ("Open", "High", "Low", "Close", "Volume"):
            for col in x.columns:
                if field in [str(part).strip() for part in col]:
                    selected[field] = col
                    break
        if selected:
            x = x[[selected[k] for k in selected]]
            x.columns = list(selected.keys())

    rename = {}
    for c in x.columns:
        key = str(c).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[c] = key.capitalize()
    x = x.rename(columns=rename)
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in x.columns:
            if c == "Volume":
                x[c] = 0.0
            else:
                return pd.DataFrame()

    x = x[["Open", "High", "Low", "Close", "Volume"]].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["High", "Low", "Close"])
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _true_range(data: pd.DataFrame) -> pd.Series:
    prev_close = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def build_features(
    df: pd.DataFrame,
    cfg: Optional[PriceActionConfig] = None,
) -> pd.DataFrame:
    """Build Phase-1 price-action features for every available bar.

    The row at t uses only OHLCV values at t and historical values <= t-1.
    Rolling reference levels are shifted by one bar so the current bar cannot
    redefine the level against which it is tested.
    """
    cfg = cfg or PriceActionConfig()
    if cfg.atr_period < 2 or cfg.volume_window < 2 or cfg.breakout_lookback < 2:
        raise ValueError("periods must be >= 2")

    data = _clean_ohlcv(df)
    if data.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=data.index)

    o = data["Open"]
    h = data["High"]
    l = data["Low"]
    c = data["Close"]
    v = data["Volume"]

    tr = _true_range(data)
    atr = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
    range_ = h - l
    safe_range = range_.where(range_ > 0, np.nan)

    # Candle geometry.
    body = (c - o).abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    close_position = (c - l) / safe_range
    body_pct = body / safe_range
    upper_wick_pct = upper_wick / safe_range
    lower_wick_pct = lower_wick / safe_range

    out["candle_bullish"] = (c > o).astype("int8")
    out["candle_bearish"] = (c < o).astype("int8")
    out["candle_body_pct"] = body_pct
    out["candle_upper_wick_pct"] = upper_wick_pct
    out["candle_lower_wick_pct"] = lower_wick_pct
    out["candle_close_position"] = close_position.clip(0.0, 1.0)
    out["candle_close_strength"] = (close_position * 2.0 - 1.0).clip(-1.0, 1.0)
    out["candle_strong_close"] = (
        close_position >= cfg.min_close_strength
    ).astype("int8")
    out["candle_weak_close"] = (
        close_position <= (1.0 - cfg.min_close_strength)
    ).astype("int8")

    # Previous-bar references. No current-bar contamination.
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    prev_close = c.shift(1)
    prior_breakout_high = h.shift(1).rolling(
        cfg.breakout_lookback, min_periods=cfg.breakout_lookback
    ).max()
    prior_breakout_low = l.shift(1).rolling(
        cfg.breakout_lookback, min_periods=cfg.breakout_lookback
    ).min()

    outside_bar = (h > prev_high) & (l < prev_low)
    outside_range_atr = range_ / atr

    # Direction is deliberately based on close acceptance, not merely candle
    # color. A neutral outside bar is possible and is not directional evidence.
    outside_bullish = outside_bar & (close_position >= cfg.min_close_strength)
    outside_bearish = outside_bar & (
        close_position <= (1.0 - cfg.min_close_strength)
    )
    outside_neutral = outside_bar & ~outside_bullish & ~outside_bearish

    vol_ma = v.shift(1).rolling(
        cfg.volume_window, min_periods=cfg.volume_window
    ).mean()
    volume_ratio = v / vol_ma.replace(0, np.nan)

    out["outside_bar"] = outside_bar.astype("int8")
    out["outside_bar_bullish"] = outside_bullish.astype("int8")
    out["outside_bar_bearish"] = outside_bearish.astype("int8")
    out["outside_bar_neutral"] = outside_neutral.astype("int8")
    out["outside_bar_range_atr"] = outside_range_atr
    out["outside_bar_expanded_range"] = (
        outside_bar & (outside_range_atr >= cfg.min_outside_range_atr)
    ).astype("int8")
    out["outside_bar_close_position"] = close_position.clip(0.0, 1.0).where(
        outside_bar
    )
    out["outside_bar_volume_ratio"] = volume_ratio.where(outside_bar)
    out["outside_bar_strong_volume"] = (
        outside_bar & (volume_ratio >= cfg.strong_volume_ratio)
    ).astype("int8")

    # Close-confirmed acceptance of prior structure. Current bar is compared
    # with levels formed only from bars strictly before it.
    out["close_above_prev_high"] = (c > prev_high).astype("int8")
    out["close_below_prev_low"] = (c < prev_low).astype("int8")
    out["close_above_breakout_high"] = (
        c > prior_breakout_high
    ).astype("int8")
    out["close_below_breakout_low"] = (
        c < prior_breakout_low
    ).astype("int8")

    out["atr"] = atr
    out["atr_pct"] = atr / c
    out["volume_ratio"] = volume_ratio

    # Explicit state labels make downstream logging and tests unambiguous.
    def close_state(row: pd.Series) -> str:
        cp = _f(row["candle_close_position"])
        if not math.isfinite(cp):
            return "UNAVAILABLE"
        if cp >= cfg.min_close_strength:
            return "CLOSE_NEAR_HIGH"
        if cp <= 1.0 - cfg.min_close_strength:
            return "CLOSE_NEAR_LOW"
        return "CLOSE_MID_RANGE"

    def outside_state(row: pd.Series) -> str:
        if int(row["outside_bar"]) != 1:
            return "NONE"
        if int(row["outside_bar_bullish"]) == 1:
            return "BULLISH"
        if int(row["outside_bar_bearish"]) == 1:
            return "BEARISH"
        return "NEUTRAL"

    out["candle_close_state"] = out.apply(close_state, axis=1)
    out["outside_bar_state"] = out.apply(outside_state, axis=1)

    # These are confirmations, not trading signals.
    out["price_action_confirmation"] = (
        (out["candle_strong_close"] == 1)
        | (out["outside_bar_bullish"] == 1)
        | (out["outside_bar_bearish"] == 1)
    ).astype("int8")

    out["price_action_direction"] = np.select(
        [
            out["outside_bar_bullish"] == 1,
            out["outside_bar_bearish"] == 1,
            out["candle_close_strength"] >= cfg.min_close_strength,
            out["candle_close_strength"] <= -cfg.min_close_strength,
        ],
        ["BULLISH", "BEARISH", "BULLISH", "BEARISH"],
        default="NEUTRAL",
    )

    # A current bar is not usable for a close-confirmation decision until its
    # close exists. Historical rows in this batch do have a close, so this is
    # an explicit semantic marker rather than an intraday prediction.
    out["bar_closed"] = c.notna().astype("int8")

    return out


def analyze_latest(
    df: pd.DataFrame,
    cfg: Optional[PriceActionConfig] = None,
) -> Dict[str, Any]:
    """Return the latest confirmed Phase-1 price-action state."""
    cfg = cfg or PriceActionConfig()
    features = build_features(df, cfg)
    if features.empty:
        return {
            "engine": "Price Action Engine",
            "version": VERSION,
            "status": "NO_DATA",
            "features": {},
            "config": asdict(cfg),
        }

    row = features.iloc[-1]
    fields = [
        "candle_bullish", "candle_bearish", "candle_body_pct",
        "candle_upper_wick_pct", "candle_lower_wick_pct",
        "candle_close_position", "candle_close_strength",
        "candle_strong_close", "candle_weak_close", "candle_close_state",
        "outside_bar", "outside_bar_bullish", "outside_bar_bearish",
        "outside_bar_neutral", "outside_bar_range_atr", "outside_bar_expanded_range",
        "outside_bar_close_position", "outside_bar_volume_ratio",
        "outside_bar_strong_volume", "outside_bar_state",
        "close_above_prev_high", "close_below_prev_low",
        "close_above_breakout_high", "close_below_breakout_low",
        "atr", "atr_pct", "volume_ratio", "price_action_confirmation",
        "price_action_direction", "bar_closed",
    ]
    values: Dict[str, Any] = {}
    for key in fields:
        value = row[key]
        if isinstance(value, (np.integer, int)):
            values[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            values[key] = round(float(value), 6) if math.isfinite(float(value)) else None
        else:
            values[key] = str(value)

    return {
        "engine": "Price Action Engine",
        "version": VERSION,
        "status": "ANALYZED",
        "as_of": str(features.index[-1].date()) if hasattr(features.index[-1], "date") else str(features.index[-1]),
        "features": values,
        "config": asdict(cfg),
        "policy": {
            "signal_generation": False,
            "close_confirmed_only": True,
            "lookahead_free": True,
            "note": "Phase 1 produces price-action evidence; downstream engines decide how to use it.",
        },
    }


def _synthetic_data(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = np.linspace(100.0, 120.0, n)
    open_ = close - 0.5
    high = close + 0.5
    low = close - 1.5
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def run_self_test() -> bool:
    cfg = PriceActionConfig()
    df = _synthetic_data()

    # 1. Baseline: ordinary bullish candle, no outside bar.
    f = build_features(df, cfg)
    assert not f.empty
    assert int(f.iloc[-1]["outside_bar"]) == 0
    assert f.iloc[-1]["candle_close_state"] == "CLOSE_NEAR_HIGH"

    # 2. Bullish outside bar with close near high.
    x = df.copy()
    i = -1
    x.iloc[i, x.columns.get_loc("Open")] = 119.0
    x.iloc[i, x.columns.get_loc("High")] = 123.0
    x.iloc[i, x.columns.get_loc("Low")] = 98.0
    x.iloc[i, x.columns.get_loc("Close")] = 122.5
    x.iloc[i, x.columns.get_loc("Volume")] = 1_500_000.0
    f = build_features(x, cfg)
    r = f.iloc[-1]
    assert int(r["outside_bar"]) == 1
    assert int(r["outside_bar_bullish"]) == 1
    assert int(r["outside_bar_bearish"]) == 0
    assert r["outside_bar_state"] == "BULLISH"
    assert int(r["bar_closed"]) == 1

    # 3. Bearish outside bar with close near low.
    x = df.copy()
    x.iloc[i, x.columns.get_loc("Open")] = 121.0
    x.iloc[i, x.columns.get_loc("High")] = 123.0
    x.iloc[i, x.columns.get_loc("Low")] = 97.0
    x.iloc[i, x.columns.get_loc("Close")] = 98.0
    f = build_features(x, cfg)
    r = f.iloc[-1]
    assert int(r["outside_bar"]) == 1
    assert int(r["outside_bar_bearish"]) == 1
    assert r["outside_bar_state"] == "BEARISH"

    # 4. Neutral outside bar must not be directional.
    x = df.copy()
    x.iloc[i, x.columns.get_loc("High")] = 123.0
    x.iloc[i, x.columns.get_loc("Low")] = 97.0
    x.iloc[i, x.columns.get_loc("Close")] = 110.0
    f = build_features(x, cfg)
    r = f.iloc[-1]
    assert int(r["outside_bar"]) == 1
    assert int(r["outside_bar_neutral"]) == 1
    assert r["outside_bar_state"] == "NEUTRAL"

    # 5. Breakout acceptance uses prior bars only.
    x = df.copy()
    previous_high = float(x["High"].iloc[-2])
    x.iloc[i, x.columns.get_loc("Open")] = previous_high
    x.iloc[i, x.columns.get_loc("High")] = previous_high + 2.0
    x.iloc[i, x.columns.get_loc("Low")] = previous_high - 0.5
    x.iloc[i, x.columns.get_loc("Close")] = previous_high + 1.5
    f = build_features(x, cfg)
    assert int(f.iloc[-1]["close_above_prev_high"]) == 1

    # 6. Look-ahead guard: changing current high must not change prior level.
    x1 = df.copy()
    x2 = df.copy()
    x1.iloc[-1, x1.columns.get_loc("High")] = 121.0
    x2.iloc[-1, x2.columns.get_loc("High")] = 999.0
    f1 = build_features(x1, cfg)
    f2 = build_features(x2, cfg)
    # The prior-bar breakout reference is not exposed directly, but whether
    # close clears it must remain tied to prior data, not the current high.
    assert int(f1.iloc[-1]["close_above_prev_high"]) == int(f2.iloc[-1]["close_above_prev_high"])
    assert int(f1.iloc[-1]["close_below_prev_low"]) == int(f2.iloc[-1]["close_below_prev_low"])

    # 7. Missing volume must not crash the engine.
    x = df.drop(columns=["Volume"])
    f = build_features(x, cfg)
    assert not f.empty

    # 8. Empty data fails closed.
    empty = build_features(pd.DataFrame(), cfg)
    assert empty.empty
    latest = analyze_latest(pd.DataFrame(), cfg)
    assert latest["status"] == "NO_DATA"

    # 9. Latest output never emits a trading signal.
    latest = analyze_latest(df, cfg)
    assert latest["policy"]["signal_generation"] is False
    assert latest["policy"]["lookahead_free"] is True

    print("PRICE ACTION ENGINE V1.0 SELF-TEST: PASS (9/9)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader Price Action Engine V1.0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    print(json.dumps({"engine": "Price Action Engine", "version": VERSION}, indent=2))


if __name__ == "__main__":
    main()
