"""
Characteristic Curve Engine (CCE)
==================================

Statistical historical-behavior engine for AI Trader.

The engine:
1. Builds a normalized price-behavior feature vector.
2. Finds historically similar states using Mahalanobis distance.
3. Measures forward-return distributions, MFE and MAE.
4. Produces a Characteristic Score (0-100), confidence, regime,
   historical pattern statistics and a BUY/WATCH/HOLD/REDUCE/SELL signal.
    Trade-entry/exit decisions are intentionally separated for the Dynamic Entry Engine.

Design goals:
- No fixed buy price is used to determine the signal.
- No future data is used in the feature vector.
- Historical matches are separated by a minimum number of days.
- If the sample is too small, the engine returns INSUFFICIENT_HISTORY.
- Works with yfinance OHLCV DataFrames and MultiIndex columns.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import yfinance as yf


FEATURES = [
    "trend20", "trend50", "trend200", "slope50",
    "roc5", "roc20", "roc60", "rsi_norm",
    "atr_pct", "atr_z", "volatility_z",
    "drawdown20", "drawdown50", "drawdown252",
    "recovery50", "position50", "position252",
    "breakout20", "volume_z", "rs_momentum",
]

HORIZONS = (1, 5, 10, 20, 60)
VERSION = "1.4"


@dataclass
class CurveConfig:
    period: str = "5y"
    interval: str = "1d"
    benchmark: Optional[str] = None

    min_history: int = 320
    min_matches: int = 20
    max_matches: int = 50
    match_quantile: float = 0.20
    min_gap_days: int = 20

    # Match-quality controls (V1.4)
    excellent_similarity: float = 0.70
    good_similarity: float = 0.55
    moderate_similarity: float = 0.40
    max_fallback_ratio: float = 0.50
    weak_trading_confidence: float = 50.0

    covariance_window: int = 756
    covariance_ridge: float = 0.05

    # Score weights
    probability_weight: float = 25.0
    return_weight: float = 20.0
    reward_risk_weight: float = 15.0
    trend_weight: float = 10.0
    momentum_weight: float = 10.0
    volatility_weight: float = 5.0
    relative_strength_weight: float = 5.0
    volume_weight: float = 5.0
    cycle_weight: float = 5.0

    buy_score: float = 70.0
    strong_buy_score: float = 85.0
    reduce_score: float = 40.0
    sell_score: float = 30.0

    buy_probability: float = 0.60
    minimum_reward_risk: float = 1.25
    strong_reward_risk: float = 2.0

    risk_fraction: float = 0.01


def _scalar(value) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance single/multi-column output."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance commonly returns columns such as:
        #   (Price, Close), (Price, High), ...
        # or (Close, TQQQ), (High, TQQQ), ...
        # Find the OHLCV field by looking across ALL levels instead of
        # assuming that it is the last level.
        wanted = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume",
                  "adj close": "Close"}
        selected = {}
        for col in out.columns:
            parts = [str(x).strip() for x in col if str(x).strip() not in {"", "None"}]
            for part in parts:
                key = part.lower()
                if key in wanted and wanted[key] not in selected:
                    selected[wanted[key]] = col
                    break

        if selected:
            out = out[[selected[k] for k in selected]]
            out.columns = list(selected.keys())

    rename = {}
    for c in out.columns:
        lc = str(c).lower()
        if lc in {"open", "high", "low", "close", "volume"}:
            rename[c] = lc.capitalize()
        elif lc == "adj close":
            rename[c] = "Close"
    out = out.rename(columns=rename)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in out.columns:
            if c == "Volume":
                out[c] = 0.0
            else:
                return pd.DataFrame()

    out = out[needed].copy()
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Close", "High", "Low"]).sort_index()
    return out


def download_history(ticker: str, cfg: CurveConfig) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _clean_ohlcv(raw)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _rolling_position(close: pd.Series, window: int) -> pd.Series:
    lo = close.rolling(window).min()
    hi = close.rolling(window).max()
    return (close - lo) / (hi - lo).replace(0, np.nan)


def build_features(df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build the causal feature matrix. Every feature at t uses data <= t."""
    x = _clean_ohlcv(df)
    if x.empty:
        return pd.DataFrame()

    c = x["Close"]
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    atr14 = _atr(x, 14)
    atr_pct = atr14 / c
    returns = c.pct_change()

    f = pd.DataFrame(index=x.index)
    f["trend20"] = c / sma20 - 1
    f["trend50"] = c / sma50 - 1
    f["trend200"] = c / sma200 - 1
    f["slope50"] = sma50.pct_change(20)
    f["roc5"] = c.pct_change(5)
    f["roc20"] = c.pct_change(20)
    f["roc60"] = c.pct_change(60)
    f["rsi_norm"] = (_rsi(c, 14) - 50) / 50

    f["atr_pct"] = atr_pct
    f["atr_z"] = _zscore(atr_pct, 126)
    f["volatility_z"] = _zscore(returns.rolling(20).std(), 126)

    f["drawdown20"] = c / c.rolling(20).max() - 1
    f["drawdown50"] = c / c.rolling(50).max() - 1
    f["drawdown252"] = c / c.rolling(252).max() - 1

    f["recovery50"] = (c - c.rolling(50).min()) / (
        c.rolling(50).max() - c.rolling(50).min()
    ).replace(0, np.nan)

    f["position50"] = _rolling_position(c, 50)
    f["position252"] = _rolling_position(c, 252)
    f["breakout20"] = c / c.shift(1).rolling(20).max() - 1

    volume = x["Volume"].replace(0, np.nan)
    f["volume_z"] = _zscore(volume, 20)

    if benchmark is not None and not benchmark.empty:
        b = _clean_ohlcv(benchmark)["Close"].reindex(f.index).ffill()
        rs = c / b
        f["rs_momentum"] = rs / rs.shift(20) - 1
    else:
        f["rs_momentum"] = 0.0

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def _causal_standardize(features: pd.DataFrame, current_idx: int, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using only rows before the current observation."""
    start = max(0, current_idx - window)
    hist = features.iloc[start:current_idx].dropna()
    if len(hist) < 50:
        hist = features.iloc[:current_idx].dropna()

    mu = hist.mean().values
    sigma = hist.std(ddof=0).replace(0, 1.0).values
    cov = np.cov(hist.values, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.eye(len(mu))
    ridge = 0.05 * np.trace(cov) / max(len(mu), 1)
    ridge = ridge if math.isfinite(ridge) and ridge > 0 else 0.05
    cov = cov + np.eye(len(mu)) * ridge
    return mu, sigma, cov


def _mahalanobis(x: np.ndarray, matrix: np.ndarray, inv_cov: np.ndarray) -> float:
    d = x - matrix
    return float(np.sqrt(max(0.0, d @ inv_cov @ d.T)))


def _select_spaced(indices: List[int], distances: Dict[int, float], max_n: int, gap: int) -> List[int]:
    selected: List[int] = []
    for idx in sorted(indices, key=lambda i: distances[i]):
        if all(abs(idx - s) >= gap for s in selected):
            selected.append(idx)
            if len(selected) >= max_n:
                break
    return selected


def _forward_stats(high: pd.Series, low: pd.Series, close: pd.Series, idx: int, horizons=HORIZONS) -> Dict[str, float]:
    """Measure forward behavior without using the entry day.

    Return uses future Close; MFE uses future High; MAE uses future Low.
    This prevents 1-day MFE/MAE from collapsing to the close-to-close return.
    """
    p0 = float(close.iloc[idx])
    out = {}
    for h in horizons:
        if idx + h >= len(close):
            continue
        future_high = high.iloc[idx + 1: idx + h + 1].astype(float)
        future_low = low.iloc[idx + 1: idx + h + 1].astype(float)
        future_close = close.iloc[idx + 1: idx + h + 1].astype(float)
        if future_close.empty or future_high.empty or future_low.empty:
            continue
        out[f"return_{h}d"] = float(future_close.iloc[-1] / p0 - 1)
        out[f"mfe_{h}d"] = float(future_high.max() / p0 - 1)
        out[f"mae_{h}d"] = float(future_low.min() / p0 - 1)
    return out


def _percentile(values: List[float], q: float) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(np.percentile(vals, q))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_result(
    current_features: pd.Series,
    horizon_stats: Dict[str, Dict[str, float]],
    cfg: CurveConfig,
) -> Tuple[float, Dict[str, float]]:
    h = horizon_stats.get("20d", {})
    p_up = h.get("prob_positive", 0.5)
    median_ret = h.get("median_return", 0.0)
    median_mae = abs(h.get("median_mae", 0.0))
    median_mfe = max(0.0, h.get("median_mfe", 0.0))

    rr = median_mfe / max(median_mae, 0.005)

    probability_score = 25 * _clip01((p_up - 0.40) / 0.40)
    return_score = 20 * _clip01((median_ret + 0.05) / 0.15)
    rr_score = 15 * _clip01((rr - 1.0) / 2.0)

    trend_raw = (
        0.30 * current_features["trend20"] +
        0.40 * current_features["trend50"] +
        0.30 * current_features["trend200"]
    )
    trend_score = 10 * _clip01((trend_raw + 0.10) / 0.20)

    momentum_raw = (
        0.25 * current_features["roc5"] +
        0.45 * current_features["roc20"] +
        0.30 * current_features["rsi_norm"]
    )
    momentum_score = 10 * _clip01((momentum_raw + 0.10) / 0.20)

    vol_z = current_features.get("volatility_z", 0.0)
    volatility_score = 5 * _clip01(1 - abs(vol_z) / 4)

    rs_score = 5 * _clip01((current_features["rs_momentum"] + 0.10) / 0.20)
    volume_score = 5 * _clip01((current_features["volume_z"] + 1) / 3)

    # Cycle proxy: recovery + position + trend.
    cycle_raw = (
        0.40 * current_features["recovery50"] +
        0.30 * current_features["position50"] +
        0.30 * _clip01((current_features["trend50"] + 0.10) / 0.20)
    )
    cycle_score = 5 * _clip01(cycle_raw)

    breakdown = {
        "historical_probability": probability_score,
        "expected_return": return_score,
        "reward_risk": rr_score,
        "trend": trend_score,
        "momentum": momentum_score,
        "volatility": volatility_score,
        "relative_strength": rs_score,
        "volume": volume_score,
        "cycle": cycle_score,
    }
    return float(sum(breakdown.values())), breakdown


def _classify_cycle(f: pd.Series) -> str:
    rsi = f["rsi_norm"] * 50 + 50
    dd = f["drawdown252"]
    pos = f["position252"]
    trend = f["trend50"]
    momentum = f["roc20"]

    if dd <= -0.40 and rsi < 40:
        return "CAPITULATION"
    if trend < -0.05 and momentum < -0.05:
        return "DOWNTREND"
    if pos < 0.15 and momentum >= 0:
        return "BASE"
    if momentum > 0.05 and trend > 0:
        if rsi >= 75:
            return "EXTREME_MOMENTUM"
        return "CONFIRMED_BULL" if pos > 0.65 else "EARLY_BULL"
    if momentum > 0 and pos > 0.25:
        return "RECOVERY"
    if pos > 0.85:
        return "DISTRIBUTION"
    return "CONSOLIDATION"


def _signal(score: float, p20: float, rr: float, cycle: str, cfg: CurveConfig) -> str:
    if cycle == "DOWNTREND" and score < 50:
        return "SELL"
    if cycle == "EXTREME_MOMENTUM" and rr < 1.25:
        return "REDUCE"
    if (
        score >= cfg.strong_buy_score
        and p20 >= 0.65
        and rr >= cfg.strong_reward_risk
    ):
        return "STRONG_BUY"
    if score >= cfg.buy_score and p20 >= cfg.buy_probability and rr >= cfg.minimum_reward_risk:
        return "BUY"
    if score < cfg.sell_score:
        return "SELL"
    if score < cfg.reduce_score:
        return "REDUCE"
    return "WATCH"


def _levels(price: float, atr_pct: float, stats20: Dict[str, float]) -> Dict[str, Optional[float]]:
    mae10 = abs(stats20.get("mae_p10", 0.0))
    mae25 = abs(stats20.get("mae_p25", 0.0))
    mfe50 = max(0.0, stats20.get("mfe_p50", 0.0))
    mfe75 = max(0.0, stats20.get("mfe_p75", 0.0))
    mfe90 = max(0.0, stats20.get("mfe_p90", 0.0))

    risk_pct = max(mae25, 2.0 * atr_pct, 0.01)
    stop = price * (1 - risk_pct)

    return {
        "entry_price": round(price, 4),
        "entry_zone_low": round(price * (1 - max(mae25, atr_pct)), 4),
        "entry_zone_high": round(price, 4),
        "dynamic_stop": round(stop, 4),
        "take_profit_1": round(price * (1 + mfe50), 4),
        "take_profit_2": round(price * (1 + mfe75), 4),
        "take_profit_3": round(price * (1 + mfe90), 4),
    }


def _market_trend_label(f: pd.Series) -> str:
    t20, t50, t200 = f["trend20"], f["trend50"], f["trend200"]
    if t200 > 0 and t50 < 0:
        return "LONG_TERM_BULLISH_SHORT_TERM_BEARISH"
    if t200 < 0 and t50 < 0:
        return "BEARISH"
    if t200 > 0 and t50 > 0 and t20 > 0:
        return "BULLISH"
    if t200 > 0:
        return "LONG_TERM_BULLISH"
    return "MIXED"


def _momentum_label(f: pd.Series) -> str:
    raw = 0.25 * f["roc5"] + 0.45 * f["roc20"] + 0.30 * f["rsi_norm"]
    if raw <= -0.05:
        return "STRONGLY_NEGATIVE"
    if raw < 0:
        return "NEGATIVE"
    if raw >= 0.05:
        return "STRONGLY_POSITIVE"
    return "POSITIVE"


def _relative_strength_label(f: pd.Series) -> str:
    rs = f["rs_momentum"]
    if rs <= -0.05:
        return "WEAK"
    if rs >= 0.05:
        return "STRONG"
    return "NEUTRAL"


def _volume_label(f: pd.Series) -> str:
    z = f["volume_z"]
    if z <= -1.5:
        return "VERY_LOW"
    if z < -0.5:
        return "LOW"
    if z >= 1.5:
        return "VERY_HIGH"
    if z > 0.5:
        return "HIGH"
    return "NORMAL"


def _match_quality(mean_distance: float, matches: int, primary_matches: int, cfg: CurveConfig) -> Dict[str, object]:
    """Classify historical-match quality without changing the statistical score.

    V1.4 distinguishes pattern quality from sample size. Similarity is derived
    from the same exponential distance transform used by the engine. The
    fallback ratio reports how much the sample had to be supplemented after
    the preferred similarity-quantile selection.
    """
    similarity = math.exp(-mean_distance / 5.0) if math.isfinite(mean_distance) else 0.0
    fallback_matches = max(0, matches - primary_matches)
    fallback_ratio = fallback_matches / matches if matches else 1.0

    if similarity >= cfg.excellent_similarity:
        quality = "EXCELLENT"
    elif similarity >= cfg.good_similarity:
        quality = "GOOD"
    elif similarity >= cfg.moderate_similarity:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    return {
        "quality": quality,
        "similarity": round(similarity, 6),
        "primary_matches": int(primary_matches),
        "fallback_matches": int(fallback_matches),
        "fallback_ratio": round(fallback_ratio, 6),
    }


def analyze_dataframe(
    ticker: str,
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    cfg: Optional[CurveConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or CurveConfig()
    data = _clean_ohlcv(df)
    features = build_features(data, benchmark_df)

    valid = features[FEATURES].dropna()
    if len(valid) < cfg.min_history:
        return {
            "ticker": ticker,
            "signal": "INSUFFICIENT_HISTORY",
            "score": 0.0,
            "confidence": 0.0,
            "matches": 0,
            "error": f"Need at least {cfg.min_history} valid feature rows; got {len(valid)}",
        }

    # Latest valid feature row and its location.
    current_date = valid.index[-1]
    current_pos = features.index.get_loc(current_date)
    x_current = valid.iloc[-1].values.astype(float)

    # Causal standardization. The current observation is excluded from fit.
    mu, sigma, cov = _causal_standardize(features[FEATURES], current_pos, cfg.covariance_window)
    hist = features[FEATURES].iloc[:current_pos].copy()
    hist_z = (hist - mu) / sigma
    current_z = (x_current - mu) / sigma

    hist_z = hist_z.replace([np.inf, -np.inf], np.nan).dropna()
    if len(hist_z) < cfg.min_matches:
        return {
            "ticker": ticker,
            "signal": "INSUFFICIENT_HISTORY",
            "score": 0.0,
            "confidence": 0.0,
            "matches": len(hist_z),
            "error": "Insufficient valid historical feature states.",
        }

    inv_cov = np.linalg.pinv(np.cov(hist_z.values, rowvar=False) + np.eye(len(FEATURES)) * cfg.covariance_ridge)
    # hist_z is indexed by dates (Timestamp).  Forward-window calculations
    # require the ORIGINAL integer position in `data`/`features`.  Do not
    # enumerate hist_z: dropna() may have removed rows, which would shift the
    # position and associate a distance with the wrong historical date.
    feature_positions = {date: pos for pos, date in enumerate(features.index)}
    distances = {}
    for date, row in hist_z.iterrows():
        pos = feature_positions.get(date)
        if pos is None:
            continue
        distances[pos] = _mahalanobis(
            current_z, row.to_numpy(dtype=float), inv_cov
        )

    # Reject observations without enough future data.
    candidates = []
    for pos, dist in distances.items():
        if pos + max(HORIZONS) < len(data):
            date = features.index[pos]
            candidates.append((pos, date, dist))

    candidates.sort(key=lambda z: z[2])
    if not candidates:
        return {
            "ticker": ticker,
            "signal": "INSUFFICIENT_HISTORY",
            "score": 0.0,
            "confidence": 0.0,
            "matches": 0,
            "error": "No historical states with complete forward windows.",
        }

    # Primary selection: use the closest historical states inside the configured
    # distance quantile.  With a spacing constraint, however, a dense cluster of
    # similar observations can leave fewer than min_matches independent states.
    # In that case, progressively relax the distance cutoff and add the next
    # closest candidates until the minimum independent sample is reached.
    # This preserves the spacing rule while avoiding a false INSUFFICIENT_HISTORY
    # caused only by an overly narrow similarity quantile.
    quantile_idx = min(
        len(candidates) - 1,
        max(cfg.min_matches - 1, int(len(candidates) * cfg.match_quantile)),
    )
    cutoff = candidates[quantile_idx][2]
    eligible = [p for p in candidates if p[2] <= cutoff]
    dist_map = {p[0]: p[2] for p in eligible}
    selected_positions = _select_spaced(
        [p[0] for p in eligible], dist_map, cfg.max_matches, cfg.min_gap_days
    )
    primary_match_count = len(selected_positions)

    if len(selected_positions) < cfg.min_matches:
        # Supplement with the next-nearest states, maintaining chronological
        # spacing.  The candidates list is already sorted by distance.
        selected_set = set(selected_positions)
        for pos, _date, dist in candidates:
            if pos in selected_set:
                continue
            if all(abs(pos - selected) >= cfg.min_gap_days for selected in selected_positions):
                selected_positions.append(pos)
                selected_set.add(pos)
                dist_map[pos] = dist
                if len(selected_positions) >= cfg.min_matches:
                    break

    # If the fallback still cannot produce the required independent sample,
    # the data genuinely does not contain enough separated historical states.

    high = data["High"].reset_index(drop=True)
    low = data["Low"].reset_index(drop=True)
    close = data["Close"].reset_index(drop=True)
    records = []
    for pos in selected_positions:
        stats = _forward_stats(high, low, close, pos)
        stats["distance"] = dist_map[pos]
        records.append(stats)

    if len(records) < cfg.min_matches:
        return {
            "ticker": ticker,
            "signal": "INSUFFICIENT_HISTORY",
            "score": 0.0,
            "confidence": 0.0,
            "matches": len(records),
            "error": f"Only {len(records)} independent historical matches after spacing.",
        }

    horizon_stats: Dict[str, Dict[str, float]] = {}
    for h in HORIZONS:
        key = f"return_{h}d"
        vals = [r[key] for r in records if key in r]
        mfe = [r[f"mfe_{h}d"] for r in records if f"mfe_{h}d" in r]
        mae = [r[f"mae_{h}d"] for r in records if f"mae_{h}d" in r]
        horizon_stats[f"{h}d"] = {
            "prob_positive": float(np.mean(np.array(vals) > 0)) if vals else 0.5,
            "mean_return": float(np.mean(vals)) if vals else 0.0,
            "median_return": float(np.median(vals)) if vals else 0.0,
            "std_return": float(np.std(vals, ddof=0)) if vals else 0.0,
            "p10_return": _percentile(vals, 10) or 0.0,
            "p90_return": _percentile(vals, 90) or 0.0,
            "median_mfe": float(np.median(mfe)) if mfe else 0.0,
            "median_mae": float(np.median(mae)) if mae else 0.0,
            "mfe_p50": _percentile(mfe, 50) or 0.0,
            "mfe_p75": _percentile(mfe, 75) or 0.0,
            "mfe_p90": _percentile(mfe, 90) or 0.0,
            "mae_p10": _percentile(mae, 10) or 0.0,
            "mae_p25": _percentile(mae, 25) or 0.0,
            "mae_p50": _percentile(mae, 50) or 0.0,
        }

    current = valid.iloc[-1]
    score, breakdown = _score_result(current, horizon_stats, cfg)
    cycle = _classify_cycle(current)
    h20 = horizon_stats["20d"]

    rr = max(0.0, h20["mfe_p50"]) / max(abs(h20["mae_p50"]), 0.005)
    signal = _signal(score, h20["prob_positive"], rr, cycle, cfg)

    # V1.4: keep historical confidence separate from trading confidence.
    # Historical confidence reflects the statistical sample; trading confidence
    # also penalizes reliance on fallback matches and weak pattern similarity.
    mean_distance = float(np.mean([r["distance"] for r in records]))
    match_quality = _match_quality(mean_distance, len(records), primary_match_count, cfg)
    similarity = match_quality["similarity"]
    consistency = h20["prob_positive"]
    sample_factor = min(1.0, len(records) / cfg.max_matches)
    historical_confidence = 100 * (
        0.35 * sample_factor +
        0.25 * similarity +
        0.20 * consistency +
        0.20 * _clip01(h20["prob_positive"] * 1.15)
    )

    fallback_penalty = min(0.25, match_quality["fallback_ratio"] * 0.50)
    trading_confidence = historical_confidence * (1.0 - fallback_penalty)
    if match_quality["quality"] == "WEAK":
        trading_confidence *= 0.80

    # Conservative V1.4 guard: do not allow weakly matched/fallback-heavy
    # patterns to produce an aggressive BUY solely from a high raw score.
    signal = _signal(score, h20["prob_positive"], rr, cycle, cfg)
    if signal in {"BUY", "STRONG_BUY"}:
        if (match_quality["quality"] == "WEAK"
                or match_quality["fallback_ratio"] > cfg.max_fallback_ratio
                or trading_confidence < cfg.weak_trading_confidence):
            signal = "WATCH"

    confidence = trading_confidence

    price = float(data["Close"].iloc[-1])
    atr_pct = float(current["atr_pct"])
    levels = _levels(price, atr_pct, h20)

    h20 = horizon_stats["20d"]
    trend = _market_trend_label(current)
    momentum = _momentum_label(current)
    relative_strength = _relative_strength_label(current)
    volume = _volume_label(current)

    # Descriptive only: this is an historical excursion ratio, not a
    # probability-adjusted trade reward/risk calculation.
    historical_excursion_ratio = rr

    historical_pattern = {
        "probability_positive_20d": round(h20["prob_positive"], 6),
        "mean_return_20d": round(h20["mean_return"], 6),
        "median_return_20d": round(h20["median_return"], 6),
        "median_mfe_20d": round(h20["median_mfe"], 6),
        "median_mae_20d": round(h20["median_mae"], 6),
        "mfe_p50": round(h20["mfe_p50"], 6),
        "mfe_p75": round(h20["mfe_p75"], 6),
        "mfe_p90": round(h20["mfe_p90"], 6),
        "mae_p10": round(h20["mae_p10"], 6),
        "mae_p25": round(h20["mae_p25"], 6),
        "mae_p50": round(h20["mae_p50"], 6),
        "historical_excursion_ratio_20d": round(historical_excursion_ratio, 3),
    }

    market_condition = {
        "trend": trend,
        "momentum": momentum,
        "relative_strength": relative_strength,
        "volume": volume,
        "cycle": cycle,
    }

    decision_context = {
        "signal": signal,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "historical_confidence": round(historical_confidence, 2),
        "trading_confidence": round(trading_confidence, 2),
        "match_quality": match_quality["quality"],
        "fallback_ratio": match_quality["fallback_ratio"],
    }

    return {
        "engine": "Characteristic Curve Engine",
        "version": VERSION,
        "ticker": ticker,
        "as_of": str(data.index[-1].date()),
        "current_price": round(price, 4),
        "price": round(price, 4),  # backward compatibility
        "signal": signal,          # backward compatibility
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "cycle": cycle,
        "matches": len(records),
        "mean_match_distance": round(mean_distance, 4),
        "similarity": round(similarity * 100, 2),
        "match_quality": match_quality["quality"],
        "primary_matches": match_quality["primary_matches"],
        "fallback_matches": match_quality["fallback_matches"],
        "fallback_ratio": match_quality["fallback_ratio"],
        "historical_confidence": round(historical_confidence, 2),
        "trading_confidence": round(trading_confidence, 2),
        "features": {k: round(_scalar(current[k]), 6) for k in FEATURES},
        "score_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "horizons": {
            k: {kk: round(vv, 6) for kk, vv in v.items()}
            for k, v in horizon_stats.items()
        },
        "historical_pattern": historical_pattern,
        "market_condition": market_condition,
        "decision_context": decision_context,
        "historical_probability_20d": round(h20["prob_positive"], 6),
        "historical_return_20d": round(h20["median_return"], 6),
        "historical_mfe_20d": round(h20["median_mfe"], 6),
        "historical_mae_20d": round(h20["median_mae"], 6),
        "historical_excursion_ratio_20d": round(historical_excursion_ratio, 3),
        "reward_risk_20d": round(rr, 3),  # backward compatibility
        "levels": {
            "current_price": round(price, 4),
            "historical_pullback_low": levels["entry_zone_low"],
            "historical_pullback_high": levels["entry_zone_high"],
            "historical_stop_reference": levels["dynamic_stop"],
            "historical_tp_50": levels["take_profit_1"],
            "historical_tp_75": levels["take_profit_2"],
            "historical_tp_90": levels["take_profit_3"],
            "risk_pct": round(abs(h20["mae_p25"]), 6),
        },
        "legacy_levels": levels,
        "config": asdict(cfg),
    }


def analyze(
    ticker: str,
    cfg: Optional[CurveConfig] = None,
    benchmark: Optional[str] = None,
) -> Dict[str, object]:
    cfg = cfg or CurveConfig(benchmark=benchmark)
    benchmark = benchmark or cfg.benchmark

    df = download_history(ticker, cfg)
    if df.empty:
        return {"ticker": ticker, "signal": "NO_DATA", "score": 0.0, "confidence": 0.0}

    bdf = None
    if benchmark:
        bdf = download_history(benchmark, cfg)
        if bdf.empty:
            bdf = None

    return analyze_dataframe(ticker, df, bdf, cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    result = analyze(
        args.ticker,
        CurveConfig(period=args.period, benchmark=args.benchmark),
        benchmark=args.benchmark,
    )
    print(pd.Series(result).to_string())
