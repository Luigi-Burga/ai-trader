"""
AI Trader - Volume Market Profile (VMP) - Analytical Engine V1.0

Purpose
-------
Standalone analytical module for AI Trader.

IMPORTANT
---------
This module is NOT connected to CCE, Signal Engine, Risk Gate, Executor,
or main.py. It only calculates Volume Market Profile features.

Data source
-----------
Designed for daily OHLCV data such as yfinance output.

Look-ahead policy
-----------------
By default, the profile uses ONLY COMPLETED daily bars.

For a decision made on trading day T before T closes:
    profile_end = last completed bar before T

For historical analysis:
    an explicit end date may be supplied; the supplied OHLCV rows through
    that date are treated as completed observations.

The module never uses rows after profile_end.

Approximation
-------------
Daily OHLCV does not contain the exact intrabar volume-at-price distribution.
Therefore this engine uses a deterministic RANGE-OVERLAP allocation:

For each daily bar:
    [Low, High] + Volume

the bar volume is distributed across price bins in proportion to the
overlap between the bar's price range and each bin.

This is an approximation, NOT tick-level or 1-minute Volume Profile.

Profiles
--------
Default windows:
    VMP20   = 20 completed sessions
    VMP60   = 60 completed sessions
    VMP252  = 252 completed sessions

Value Area:
    70% of total profile volume.

Rows:
    50 bins per profile by default.

Primary outputs
---------------
For each window:
    poc
    vah
    val
    profile_high
    profile_low
    value_area_pct
    value_area_position
    poc_distance_pct
    vah_distance_pct
    val_distance_pct

Across windows:
    poc_migration_20_60_pct
    poc_migration_60_252_pct
    value_area_alignment_20_60
    value_area_alignment_20_60_252

No trading signal is generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (20, 60, 252)
DEFAULT_ROWS = 50
DEFAULT_VALUE_AREA_PCT = 0.70


@dataclass(frozen=True)
class VMPConfig:
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    rows: int = DEFAULT_ROWS
    value_area_pct: float = DEFAULT_VALUE_AREA_PCT
    completed_bars_only: bool = True

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows cannot be empty.")
        if any(int(w) <= 0 for w in self.windows):
            raise ValueError("all windows must be positive.")
        if int(self.rows) < 10:
            raise ValueError("rows must be at least 10.")
        if not 0 < float(self.value_area_pct) < 1:
            raise ValueError("value_area_pct must be between 0 and 1.")


def _normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("OHLCV data is empty.")

    frame = data.copy()

    # Accept common yfinance MultiIndex/single-level output.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            str(col[0]).strip().lower()
            if isinstance(col, tuple)
            else str(col).strip().lower()
            for col in frame.columns
        ]
    else:
        frame.columns = [str(c).strip().lower() for c in frame.columns]

    aliases = {
        "adj close": "adj_close",
    }
    frame = frame.rename(columns=aliases)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = frame[["open", "high", "low", "close", "volume"]].copy()

    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["high", "low", "close", "volume"])

    frame = frame[(frame["high"] >= frame["low"]) & (frame["volume"] >= 0)]

    if frame.empty:
        raise ValueError("No valid OHLCV rows remain after normalization.")

    frame = frame.sort_index()

    # Remove duplicate timestamps while retaining the latest row.
    frame = frame[~frame.index.duplicated(keep="last")]

    return frame


def _completed_daily_data(
    data: pd.DataFrame,
    *,
    end_date=None,
) -> pd.DataFrame:
    """
    Normalize data and optionally truncate it at an explicit completed-bar date.

    `end_date` is the key look-ahead control for historical testing:
    only observations with index <= end_date are eligible.

    The caller must supply completed observations. This module deliberately
    does not guess whether the final row is an unfinished intraday bar.
    """
    frame = _normalize_ohlcv(data)

    if end_date is not None:
        cutoff = pd.Timestamp(end_date)
        if getattr(frame.index, "tz", None) is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(frame.index.tz)
        elif getattr(frame.index, "tz", None) is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)

        frame = frame.loc[frame.index <= cutoff]

    if frame.empty:
        raise ValueError("No completed OHLCV observations remain after end_date.")

    return frame


def _price_bins(low: float, high: float, rows: int) -> tuple[np.ndarray, float]:
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Profile range contains non-finite values.")

    if high <= low:
        # Degenerate range: one effective bin centered on the price.
        pad = max(abs(low) * 1e-6, 1e-6)
        low -= pad
        high += pad

    edges = np.linspace(low, high, rows + 1)
    width = float(edges[1] - edges[0])
    return edges, width


def _allocate_bar_volume(
    volume: float,
    bar_low: float,
    bar_high: float,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Allocate a bar's volume by proportional overlap with price bins.

    This is deterministic and avoids pretending that daily OHLCV contains
    exact volume-at-price information.
    """
    result = np.zeros(len(edges) - 1, dtype=float)

    if volume <= 0:
        return result

    if bar_high <= bar_low:
        idx = np.searchsorted(edges, bar_low, side="right") - 1
        idx = max(0, min(idx, len(result) - 1))
        result[idx] = volume
        return result

    total_range = bar_high - bar_low

    left = np.maximum(edges[:-1], bar_low)
    right = np.minimum(edges[1:], bar_high)
    overlap = np.maximum(0.0, right - left)

    overlap_sum = float(overlap.sum())

    if overlap_sum <= 0:
        idx = np.searchsorted(edges, (bar_low + bar_high) / 2, side="right") - 1
        idx = max(0, min(idx, len(result) - 1))
        result[idx] = volume
        return result

    result = volume * (overlap / total_range)

    # Numerical conservation: force allocated volume to equal input volume.
    allocated = float(result.sum())
    if allocated > 0:
        result *= volume / allocated

    return result


def _value_area(
    profile_volume: np.ndarray,
    prices: np.ndarray,
    value_area_pct: float,
) -> tuple[int, int, int]:
    """
    Return (poc_idx, val_idx, vah_idx).

    Value Area follows the standard expansion concept:
    start at POC, then compare adjacent bins above/below and add the
    larger-volume side until the target percentage is reached.
    """
    if profile_volume.size == 0 or profile_volume.sum() <= 0:
        raise ValueError("Profile contains no volume.")

    poc_idx = int(np.argmax(profile_volume))
    target = float(profile_volume.sum()) * value_area_pct

    included = {poc_idx}
    accumulated = float(profile_volume[poc_idx])

    lower = poc_idx - 1
    upper = poc_idx + 1

    while accumulated < target and (lower >= 0 or upper < len(profile_volume)):
        lower_volume = (
            float(profile_volume[lower]) if lower >= 0 else -1.0
        )
        upper_volume = (
            float(profile_volume[upper]) if upper < len(profile_volume) else -1.0
        )

        if upper_volume > lower_volume:
            chosen = upper
            upper += 1
        elif lower_volume > upper_volume:
            chosen = lower
            lower -= 1
        else:
            # Tie: closest to POC. If symmetric, choose above POC.
            if upper >= 0 and upper < len(profile_volume):
                chosen = upper
                upper += 1
            else:
                chosen = lower
                lower -= 1

        if chosen < 0 or chosen >= len(profile_volume):
            break

        included.add(chosen)
        accumulated += float(profile_volume[chosen])

    val_idx = min(included)
    vah_idx = max(included)

    return poc_idx, val_idx, vah_idx


def calculate_profile(
    data: pd.DataFrame,
    *,
    window: int,
    rows: int = DEFAULT_ROWS,
    value_area_pct: float = DEFAULT_VALUE_AREA_PCT,
    end_date=None,
) -> dict:
    """
    Calculate one rolling Volume Market Profile.

    The final row of `data` is considered the latest completed observation.
    No rows after it are used.
    """
    frame = _completed_daily_data(data, end_date=end_date)

    if len(frame) < window:
        raise ValueError(
            f"Insufficient history for VMP{window}: "
            f"required={window}, available={len(frame)}"
        )

    sample = frame.tail(window)

    profile_low = float(sample["low"].min())
    profile_high = float(sample["high"].max())

    edges, bin_width = _price_bins(profile_low, profile_high, rows)
    profile_volume = np.zeros(rows, dtype=float)

    for row in sample.itertuples(index=False):
        profile_volume += _allocate_bar_volume(
            volume=float(row.volume),
            bar_low=float(row.low),
            bar_high=float(row.high),
            edges=edges,
        )

    centers = (edges[:-1] + edges[1:]) / 2.0

    poc_idx, val_idx, vah_idx = _value_area(
        profile_volume,
        centers,
        value_area_pct,
    )

    poc = float(centers[poc_idx])
    val = float(centers[val_idx])
    vah = float(centers[vah_idx])

    current_price = float(sample["close"].iloc[-1])

    if current_price < val:
        value_position = "BELOW_VALUE"
    elif current_price > vah:
        value_position = "ABOVE_VALUE"
    else:
        value_position = "INSIDE_VALUE"

    def pct_distance(level: float) -> float:
        if level == 0:
            return np.nan
        return (current_price - level) / level * 100.0

    total_volume = float(profile_volume.sum())

    return {
        "window": int(window),
        "bars_used": int(len(sample)),
        "profile_start": sample.index[0],
        "profile_end": sample.index[-1],
        "current_price": current_price,
        "profile_low": profile_low,
        "profile_high": profile_high,
        "poc": poc,
        "vah": vah,
        "val": val,
        "value_area_pct": float(value_area_pct),
        "value_area_volume": total_volume * float(value_area_pct),
        "profile_volume": total_volume,
        "value_position": value_position,
        "poc_distance_pct": pct_distance(poc),
        "vah_distance_pct": pct_distance(vah),
        "val_distance_pct": pct_distance(val),
        "bin_width": bin_width,
    }


def calculate_vmp(
    data: pd.DataFrame,
    *,
    config: Optional[VMPConfig] = None,
    end_date=None,
) -> dict:
    """
    Calculate VMP20/VMP60/VMP252 and cross-window features.

    No trading signal is generated.
    """
    cfg = config or VMPConfig()

    frame = _completed_daily_data(data, end_date=end_date)

    profiles: Dict[str, dict] = {}

    for window in cfg.windows:
        profiles[f"vmp{window}"] = calculate_profile(
            frame,
            window=int(window),
            rows=cfg.rows,
            value_area_pct=cfg.value_area_pct,
        )

    result = {
        "engine": "Volume Market Profile",
        "version": "1.0",
        "data_policy": "COMPLETED_BARS_ONLY",
        "allocation_method": "RANGE_OVERLAP",
        "rows": int(cfg.rows),
        "value_area_pct": float(cfg.value_area_pct),
        "profiles": profiles,
    }

    def get_poc(window: int) -> Optional[float]:
        item = profiles.get(f"vmp{window}")
        return None if item is None else float(item["poc"])

    poc20 = get_poc(20)
    poc60 = get_poc(60)
    poc252 = get_poc(252)

    current_price = float(frame["close"].iloc[-1])

    if poc20 is not None and poc60 is not None and poc60 != 0:
        poc_migration_20_60_pct = (poc20 - poc60) / poc60 * 100.0
    else:
        poc_migration_20_60_pct = None

    if poc60 is not None and poc252 is not None and poc252 != 0:
        poc_migration_60_252_pct = (poc60 - poc252) / poc252 * 100.0
    else:
        poc_migration_60_252_pct = None

    value_alignment = {}
    for window in cfg.windows:
        item = profiles[f"vmp{window}"]
        value_alignment[f"vmp{window}"] = {
            "position": item["value_position"],
            "above_vah": bool(current_price > item["vah"]),
            "below_val": bool(current_price < item["val"]),
        }

    result["cross_window"] = {
        "poc_migration_20_60_pct": poc_migration_20_60_pct,
        "poc_migration_60_252_pct": poc_migration_60_252_pct,
        "value_alignment": value_alignment,
    }

    return result


def flatten_features(vmp_result: dict) -> dict:
    """
    Convert the analytical result into a compact feature dictionary.

    This helper does not connect to CCE.
    """
    features = {}

    for key, profile in vmp_result["profiles"].items():
        features[f"{key}_poc_distance_pct"] = profile["poc_distance_pct"]
        features[f"{key}_vah_distance_pct"] = profile["vah_distance_pct"]
        features[f"{key}_val_distance_pct"] = profile["val_distance_pct"]

        position_map = {
            "BELOW_VALUE": -1.0,
            "INSIDE_VALUE": 0.0,
            "ABOVE_VALUE": 1.0,
        }
        features[f"{key}_value_position"] = position_map[
            profile["value_position"]
        ]

    features.update({
        "vmp_poc_migration_20_60_pct":
            vmp_result["cross_window"]["poc_migration_20_60_pct"],
        "vmp_poc_migration_60_252_pct":
            vmp_result["cross_window"]["poc_migration_60_252_pct"],
    })

    return features


def self_test() -> None:
    """Deterministic mathematical smoke tests without network access."""
    idx = pd.date_range("2025-01-01", periods=300, freq="B")

    close = np.linspace(100, 130, len(idx))
    high = close + 2
    low = close - 2
    open_ = close - 0.5
    volume = np.full(len(idx), 1_000_000.0)

    data = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=idx)

    result = calculate_vmp(data)

    assert "vmp20" in result["profiles"]
    assert "vmp60" in result["profiles"]
    assert "vmp252" in result["profiles"]

    for key, profile in result["profiles"].items():
        assert profile["bars_used"] == int(key.replace("vmp", ""))
        assert profile["val"] <= profile["poc"] <= profile["vah"]
        assert profile["profile_volume"] > 0

    features = flatten_features(result)
    assert "vmp20_poc_distance_pct" in features
    assert "vmp60_poc_distance_pct" in features
    assert "vmp252_poc_distance_pct" in features

    # Look-ahead guard:
    # Calculate the profile as of the penultimate completed bar while the
    # dataframe ALSO contains a future bar. The future bar must be ignored.
    cutoff = data.index[-2]
    as_of_with_future = calculate_vmp(data, end_date=cutoff)
    as_of_without_future = calculate_vmp(data.iloc[:-1], end_date=cutoff)

    for window in DEFAULT_WINDOWS:
        a = as_of_with_future["profiles"][f"vmp{window}"]
        b = as_of_without_future["profiles"][f"vmp{window}"]
        assert a["profile_end"] == b["profile_end"] == cutoff
        assert np.isclose(a["poc"], b["poc"])
        assert np.isclose(a["vah"], b["vah"])
        assert np.isclose(a["val"], b["val"])

    # Explicitly prove that the future row was excluded.
    latest = data.index[-1]
    assert as_of_with_future["profiles"]["vmp20"]["profile_end"] < latest

    print("VMP self-test: PASS")
    print("Look-ahead guard: PASS")


if __name__ == "__main__":
    self_test()
