"""
Characteristic Curve Engine (CCE)
=================================

Historical-pattern engine for stock/ETF analysis.

Purpose
-------
Find historical market states that resemble the current state of an asset
and use the subsequent historical behavior of those states to estimate:

- Probability of positive return
- Expected return
- Maximum Favorable Excursion (MFE)
- Maximum Adverse Excursion (MAE)
- Reward / Risk
- Trend / Momentum / Volatility / Relative Strength
- Market-cycle classification
- Buy / Watch / Reduce / Sell signal

Important
---------
All features are causal. No future information is used when constructing
the feature vector for a historical observation.

Version
-------
1.1

Changes in 1.1
--------------
- FIXED CRITICAL HISTORICAL POSITION ALIGNMENT BUG.
- Historical matches now preserve their original DataFrame positions
  after NaN removal.
- Forward statistics are therefore calculated against the correct
  historical dates.
- Preserved original scoring and signal logic.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

FEATURES = [
    "trend20",
    "trend50",
    "trend200",
    "slope50",
    "roc5",
    "roc20",
    "roc60",
    "rsi_norm",
    "atr_pct",
    "atr_z",
    "volatility_z",
    "drawdown20",
    "drawdown50",
    "drawdown252",
    "recovery50",
    "position50",
    "position252",
    "breakout20",
    "volume_z",
    "rs_momentum",
]

HORIZONS = (1, 5, 10, 20, 60)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

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

    covariance_window: int = 756
    covariance_ridge: float = 0.05

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


# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def _scalar(value) -> float:
    """
    Safely convert a pandas/numpy scalar into float.
    """
    if isinstance(value, pd.Series):
        if value.empty:
            return float("nan")
        value = value.iloc[0]

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return float("nan")
        value = value.iloc[0, 0]

    try:
        return float(value)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------------------

def _clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV data returned by yfinance.

    Handles:
        - Standard columns
        - MultiIndex columns
        - Single ticker downloads
        - Adj Close + Close simultaneously

    Important:
        If both Close and Adj Close exist, Close is preferred.
    """

    if data is None or data.empty:
        raise ValueError("Empty market data.")

    df = data.copy()

    # =========================================================
    # 1. Normalize MultiIndex columns
    # =========================================================

    if isinstance(df.columns, pd.MultiIndex):

        normalized_columns = []

        for col in df.columns:

            parts = [
                str(part).strip()
                for part in col
                if str(part).strip()
            ]

            # Search for the OHLCV field in any level
            field = None

            for part in parts:

                part_lower = part.lower()

                if part_lower in {
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "volume",
                }:
                    field = part_lower
                    break

            if field is None:
                field = "_".join(parts).lower()

            normalized_columns.append(field)

        df.columns = normalized_columns

    # =========================================================
    # 2. Normalize standard columns
    # =========================================================

    else:

        normalized_columns = []

        for col in df.columns:

            name = str(col).strip().lower()

            if name == "adj close":
                name = "adj close"

            normalized_columns.append(name)

        df.columns = normalized_columns

    # =========================================================
    # 3. Remove duplicated OHLCV columns safely
    # =========================================================
    #
    # yfinance may provide:
    #
    #   close
    #   adj close
    #
    # We use CLOSE and ignore ADJ CLOSE.
    #
    # It may also produce duplicate columns after flattening.
    # Keep the first occurrence.
    # =========================================================

    if df.columns.duplicated().any():

        df = df.loc[
            :,
            ~df.columns.duplicated(
                keep="first"
            ),
        ]

    # =========================================================
    # 4. If Close exists, remove Adj Close
    # =========================================================

    if (
        "close" in df.columns
        and "adj close" in df.columns
    ):

        df = df.drop(
            columns=["adj close"]
        )

    # =========================================================
    # 5. Required OHLCV fields
    # =========================================================

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            f"{missing}. "
            f"Available columns: "
            f"{list(df.columns)}"
        )

    # =========================================================
    # 6. Keep only OHLCV
    # =========================================================

    df = df[
        required
    ].copy()

    # =========================================================
    # 7. Force every column to be 1-dimensional numeric
    # =========================================================

    for column in required:

        series = df[column]

        # Defensive protection against duplicate columns
        if isinstance(
            series,
            pd.DataFrame,
        ):

            series = series.iloc[
                :, 0
            ]

        df[column] = pd.to_numeric(
            series,
            errors="coerce",
        )

    # =========================================================
    # 8. Clean index
    # =========================================================

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # =========================================================
    # 9. Remove rows without essential OHLC data
    # =========================================================

    df = df.dropna(
        subset=[
            "high",
            "low",
            "close",
        ]
    )

    if df.empty:
        raise ValueError(
            "No valid OHLCV rows after cleaning."
        )

    return df

# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def download_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data.
    """

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data is None or data.empty:
        raise ValueError(
            f"No historical data available for ticker '{ticker}'."
        )

    return _clean_ohlcv(data)


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI.
    """

    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.
    """

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def _zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 50,
) -> pd.Series:
    """
    Rolling causal z-score.
    """

    mean = series.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    std = series.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    z = (series - mean) / std.replace(0, np.nan)

    return z


def _rolling_position(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Position of current value inside rolling high/low range.

    Returns:
        0   = rolling low
        1   = rolling high
    """

    rolling_low = series.rolling(
        window=window,
        min_periods=window,
    ).min()

    rolling_high = series.rolling(
        window=window,
        min_periods=window,
    ).max()

    denominator = (
        rolling_high - rolling_low
    ).replace(0, np.nan)

    return (
        (series - rolling_low) / denominator
    )


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(
    data: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build causal characteristic-curve features.

    Every feature at time t uses only information available at or before t.
    """

    df = data.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    features["trend20"] = close / sma20 - 1.0
    features["trend50"] = close / sma50 - 1.0
    features["trend200"] = close / sma200 - 1.0

    features["slope50"] = sma50.pct_change(20)

    # ---------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------

    features["roc5"] = close.pct_change(5)
    features["roc20"] = close.pct_change(20)
    features["roc60"] = close.pct_change(60)

    rsi = _rsi(close)

    features["rsi_norm"] = rsi / 100.0

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    atr = _atr(
        high,
        low,
        close,
    )

    features["atr_pct"] = atr / close

    features["atr_z"] = _zscore(
        features["atr_pct"],
        window=252,
    )

    daily_returns = close.pct_change()

    volatility = daily_returns.rolling(20).std()

    features["volatility_z"] = _zscore(
        volatility,
        window=252,
    )

    # ---------------------------------------------------------
    # Drawdowns
    # ---------------------------------------------------------

    rolling_max20 = close.rolling(20).max()
    rolling_max50 = close.rolling(50).max()
    rolling_max252 = close.rolling(252).max()

    features["drawdown20"] = (
        close / rolling_max20 - 1.0
    )

    features["drawdown50"] = (
        close / rolling_max50 - 1.0
    )

    features["drawdown252"] = (
        close / rolling_max252 - 1.0
    )

    # ---------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------

    features["recovery50"] = (
        close / rolling_max50
    )

    # ---------------------------------------------------------
    # Position inside historical range
    # ---------------------------------------------------------

    features["position50"] = _rolling_position(
        close,
        50,
    )

    features["position252"] = _rolling_position(
        close,
        252,
    )

    # ---------------------------------------------------------
    # Breakout
    # ---------------------------------------------------------

    previous_high20 = close.shift(1).rolling(20).max()

    features["breakout20"] = (
        close / previous_high20 - 1.0
    )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    volume_log = np.log1p(volume)

    features["volume_z"] = _zscore(
        volume_log,
        window=252,
    )

    # ---------------------------------------------------------
    # Relative strength vs benchmark
    # ---------------------------------------------------------

    if benchmark is not None:

        benchmark_close = benchmark.reindex(
            df.index
        ).ffill()

        relative_price = (
            close / benchmark_close
        )

        features["rs_momentum"] = (
            relative_price.pct_change(20)
        )

    else:

        features["rs_momentum"] = 0.0

    return features


# ---------------------------------------------------------------------------
# CAUSAL STANDARDIZATION
# ---------------------------------------------------------------------------

def _causal_standardize(
    features: pd.DataFrame,
    current_idx: int,
    window: int,
):
    """
    Calculate mean/std/covariance using ONLY observations before
    the current observation.

    This prevents look-ahead bias.
    """

    historical = features.iloc[:current_idx].copy()

    if historical.empty:
        raise ValueError(
            "Not enough historical observations for standardization."
        )

    if len(historical) > window:
        historical = historical.iloc[-window:]

    mu = historical.mean()

    sigma = historical.std()

    sigma = sigma.replace(0, np.nan)

    sigma = sigma.fillna(1.0)

    standardized = (
        (historical - mu) / sigma
    )

    standardized = standardized.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(standardized) < 10:
        raise ValueError(
            "Not enough valid observations for covariance estimation."
        )

    covariance = np.cov(
        standardized.to_numpy(dtype=float),
        rowvar=False,
    )

    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    covariance += (
        np.eye(covariance.shape[0])
        * 0.05
    )

    return mu, sigma, covariance


# ---------------------------------------------------------------------------
# DISTANCE
# ---------------------------------------------------------------------------

def _mahalanobis(
    current_vector: np.ndarray,
    historical_vector: np.ndarray,
    inverse_covariance: np.ndarray,
) -> float:
    """
    Mahalanobis distance.
    """

    diff = (
        current_vector - historical_vector
    )

    value = float(
        diff.T
        @ inverse_covariance
        @ diff
    )

    return math.sqrt(
        max(value, 0.0)
    )


# ---------------------------------------------------------------------------
# SPACED MATCH SELECTION
# ---------------------------------------------------------------------------

def _select_spaced(
    candidates: List[Tuple[int, float]],
    min_gap_days: int,
    max_matches: int,
) -> List[int]:
    """
    Select nearest historical matches while preventing clusters
    of almost identical dates.

    candidates:
        [(original_position, distance), ...]
    """

    selected = []

    for position, distance in candidates:

        if all(
            abs(position - selected_position)
            >= min_gap_days
            for selected_position in selected
        ):

            selected.append(position)

        if len(selected) >= max_matches:
            break

    return selected


# ---------------------------------------------------------------------------
# FORWARD STATISTICS
# ---------------------------------------------------------------------------

def _forward_stats(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    position: int,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate forward return / MFE / MAE from an exact historical
    observation position.

    Definitions
    -----------
    Return:
        Closing return at the end of the horizon.

    MFE:
        Maximum Favorable Excursion based on HIGH prices during
        the forward period.

    MAE:
        Maximum Adverse Excursion based on LOW prices during
        the forward period.

    IMPORTANT
    ---------
    'position' refers to the ORIGINAL position inside the complete
    DataFrame.

    The entry day itself is excluded from the forward window.
    """

    results = {}

    if position < 0 or position >= len(close):
        return results

    entry_price = _scalar(
        close.iloc[position]
    )

    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
    ):
        return results

    for horizon in HORIZONS:

        end_position = (
            position + horizon
        )

        # Need a complete forward horizon
        if end_position >= len(close):
            continue

        future_high = high.iloc[
            position + 1:
            end_position + 1
        ]

        future_low = low.iloc[
            position + 1:
            end_position + 1
        ]

        future_close = close.iloc[
            position + 1:
            end_position + 1
        ]

        if (
            future_high.empty
            or future_low.empty
            or future_close.empty
        ):
            continue

        high_values = future_high.to_numpy(
            dtype=float
        )

        low_values = future_low.to_numpy(
            dtype=float
        )

        close_values = future_close.to_numpy(
            dtype=float
        )

        if (
            len(high_values) == 0
            or len(low_values) == 0
            or len(close_values) == 0
        ):
            continue

        if (
            not np.all(np.isfinite(high_values))
            or not np.all(np.isfinite(low_values))
            or not np.all(np.isfinite(close_values))
        ):
            continue

        # -----------------------------------------------------
        # Forward closing return
        # -----------------------------------------------------

        forward_return = (
            close_values[-1]
            / entry_price
            - 1.0
        )

        # -----------------------------------------------------
        # Maximum Favorable Excursion
        #
        # Highest intraday price reached after entry.
        # -----------------------------------------------------

        mfe = (
            np.max(high_values)
            / entry_price
            - 1.0
        )

        # -----------------------------------------------------
        # Maximum Adverse Excursion
        #
        # Lowest intraday price reached after entry.
        # -----------------------------------------------------

        mae = (
            np.min(low_values)
            / entry_price
            - 1.0
        )

        results[horizon] = {
            "return": float(
                forward_return
            ),

            "mfe": float(
                mfe
            ),

            "mae": float(
                mae
            ),
        }

    return results

# ---------------------------------------------------------------------------
# AGGREGATE HORIZON STATISTICS
# ---------------------------------------------------------------------------

def _aggregate_horizon_stats(
    all_stats: List[Dict[int, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate historical forward statistics.
    """

    result = {}

    for horizon in HORIZONS:

        returns = []
        mfes = []
        maes = []

        for stats in all_stats:

            if horizon not in stats:
                continue

            item = stats[horizon]

            returns.append(item["return"])
            mfes.append(item["mfe"])
            maes.append(item["mae"])

        if not returns:
            result[str(horizon)] = {
                "count": 0,
                "prob_positive": float("nan"),
                "mean_return": float("nan"),
                "median_return": float("nan"),
                "mean_mfe": float("nan"),
                "median_mfe": float("nan"),
                "mean_mae": float("nan"),
                "median_mae": float("nan"),
                "mfe_p50": float("nan"),
                "mfe_p75": float("nan"),
                "mfe_p90": float("nan"),
                "mae_p10": float("nan"),
                "mae_p25": float("nan"),
                "mae_p50": float("nan"),
            }

            continue

        returns_array = np.asarray(
            returns,
            dtype=float,
        )

        mfes_array = np.asarray(
            mfes,
            dtype=float,
        )

        maes_array = np.asarray(
            maes,
            dtype=float,
        )

        result[str(horizon)] = {
            "count": int(len(returns_array)),

            "prob_positive": float(
                np.mean(
                    returns_array > 0
                )
            ),

            "mean_return": float(
                np.mean(returns_array)
            ),

            "median_return": float(
                np.median(returns_array)
            ),

            "mean_mfe": float(
                np.mean(mfes_array)
            ),

            "median_mfe": float(
                np.median(mfes_array)
            ),

            "mean_mae": float(
                np.mean(maes_array)
            ),

            "median_mae": float(
                np.median(maes_array)
            ),

            "mfe_p50": float(
                np.percentile(
                    mfes_array,
                    50,
                )
            ),

            "mfe_p75": float(
                np.percentile(
                    mfes_array,
                    75,
                )
            ),

            "mfe_p90": float(
                np.percentile(
                    mfes_array,
                    90,
                )
            ),

            "mae_p10": float(
                np.percentile(
                    maes_array,
                    10,
                )
            ),

            "mae_p25": float(
                np.percentile(
                    maes_array,
                    25,
                )
            ),

            "mae_p50": float(
                np.percentile(
                    maes_array,
                    50,
                )
            ),
        }

    return result


# ---------------------------------------------------------------------------
# SCORE
# ---------------------------------------------------------------------------

def _clip01(value: float) -> float:
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _score_result(
    current_features: pd.Series,
    horizon_stats: Dict[str, Dict[str, float]],
    config: CurveConfig,
):
    """
    Calculate characteristic-curve score.

    Uses 20-day historical behavior as the primary horizon.
    """

    stats20 = horizon_stats.get(
        "20",
        {},
    )

    p_up = _scalar(
        stats20.get(
            "prob_positive",
            np.nan,
        )
    )

    median_return = _scalar(
        stats20.get(
            "median_return",
            np.nan,
        )
    )

    median_mae = _scalar(
        stats20.get(
            "median_mae",
            np.nan,
        )
    )

    median_mfe = _scalar(
        stats20.get(
            "median_mfe",
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # Reward / Risk
    # ---------------------------------------------------------

    rr_denominator = max(
        abs(median_mae),
        0.005,
    )

    rr = median_mfe / rr_denominator

    # ---------------------------------------------------------
    # Probability score
    # ---------------------------------------------------------

    probability_component = (
        config.probability_weight
        * _clip01(
            (
                p_up - 0.40
            ) / 0.40
        )
    )

    # ---------------------------------------------------------
    # Return score
    # ---------------------------------------------------------

    return_component = (
        config.return_weight
        * _clip01(
            (
                median_return + 0.05
            ) / 0.15
        )
    )

    # ---------------------------------------------------------
    # Reward / risk score
    # ---------------------------------------------------------

    reward_risk_component = (
        config.reward_risk_weight
        * _clip01(
            (
                rr - 1.0
            ) / 2.0
        )
    )

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------

    trend20 = _scalar(
        current_features.get(
            "trend20",
            np.nan,
        )
    )

    trend50 = _scalar(
        current_features.get(
            "trend50",
            np.nan,
        )
    )

    trend200 = _scalar(
        current_features.get(
            "trend200",
            np.nan,
        )
    )

    trend_raw = (
        0.40 * _clip01(
            (trend20 + 0.10) / 0.20
        )
        + 0.35 * _clip01(
            (trend50 + 0.15) / 0.30
        )
        + 0.25 * _clip01(
            (trend200 + 0.20) / 0.40
        )
    )

    trend_component = (
        config.trend_weight
        * trend_raw
    )

    # ---------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------

    roc5 = _scalar(
        current_features.get(
            "roc5",
            np.nan,
        )
    )

    roc20 = _scalar(
        current_features.get(
            "roc20",
            np.nan,
        )
    )

    rsi_norm = _scalar(
        current_features.get(
            "rsi_norm",
            np.nan,
        )
    )

    momentum_raw = (
        0.30 * _clip01(
            (roc5 + 0.10) / 0.20
        )
        + 0.40 * _clip01(
            (roc20 + 0.15) / 0.30
        )
        + 0.30 * _clip01(
            (rsi_norm - 0.30) / 0.40
        )
    )

    momentum_component = (
        config.momentum_weight
        * momentum_raw
    )

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    volatility_z = _scalar(
        current_features.get(
            "volatility_z",
            np.nan,
        )
    )

    volatility_raw = _clip01(
        1.0 - (
            volatility_z + 1.0
        ) / 3.0
    )

    volatility_component = (
        config.volatility_weight
        * volatility_raw
    )

    # ---------------------------------------------------------
    # Relative strength
    # ---------------------------------------------------------

    rs_momentum = _scalar(
        current_features.get(
            "rs_momentum",
            np.nan,
        )
    )

    relative_strength_raw = _clip01(
        (
            rs_momentum + 0.10
        ) / 0.20
    )

    relative_strength_component = (
        config.relative_strength_weight
        * relative_strength_raw
    )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    volume_z = _scalar(
        current_features.get(
            "volume_z",
            np.nan,
        )
    )

    volume_raw = _clip01(
        (
            volume_z + 1.0
        ) / 2.0
    )

    volume_component = (
        config.volume_weight
        * volume_raw
    )

    # ---------------------------------------------------------
    # Cycle proxy
    # ---------------------------------------------------------

    recovery50 = _scalar(
        current_features.get(
            "recovery50",
            np.nan,
        )
    )

    position50 = _scalar(
        current_features.get(
            "position50",
            np.nan,
        )
    )

    cycle_raw = (
        0.40 * _clip01(
            recovery50
        )
        + 0.30 * _clip01(
            position50
        )
        + 0.30 * _clip01(
            (
                trend50 + 0.15
            ) / 0.30
        )
    )

    cycle_component = (
        config.cycle_weight
        * cycle_raw
    )

    # ---------------------------------------------------------
    # Total
    # ---------------------------------------------------------

    score = (
        probability_component
        + return_component
        + reward_risk_component
        + trend_component
        + momentum_component
        + volatility_component
        + relative_strength_component
        + volume_component
        + cycle_component
    )

    breakdown = {
        "probability": float(
            probability_component
        ),
        "return": float(
            return_component
        ),
        "reward_risk": float(
            reward_risk_component
        ),
        "trend": float(
            trend_component
        ),
        "momentum": float(
            momentum_component
        ),
        "volatility": float(
            volatility_component
        ),
        "relative_strength": float(
            relative_strength_component
        ),
        "volume": float(
            volume_component
        ),
        "cycle": float(
            cycle_component
        ),
    }

    return (
        float(score),
        breakdown,
        float(rr),
    )


# ---------------------------------------------------------------------------
# CYCLE CLASSIFICATION
# ---------------------------------------------------------------------------

def _classify_cycle(
    current_features: pd.Series,
) -> str:
    """
    Classify current market cycle.
    """

    drawdown252 = _scalar(
        current_features.get(
            "drawdown252",
            np.nan,
        )
    )

    trend50 = _scalar(
        current_features.get(
            "trend50",
            np.nan,
        )
    )

    roc20 = _scalar(
        current_features.get(
            "roc20",
            np.nan,
        )
    )

    rsi_norm = _scalar(
        current_features.get(
            "rsi_norm",
            np.nan,
        )
    )

    momentum = (
        0.5 * roc20
        + 0.5 * trend50
    )

    position252 = _scalar(
        current_features.get(
            "position252",
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # Capitulation
    # ---------------------------------------------------------

    if (
        drawdown252 <= -0.40
        and rsi_norm < 0.40
    ):
        return "CAPITULATION"

    # ---------------------------------------------------------
    # Downtrend
    # ---------------------------------------------------------

    if (
        trend50 < -0.05
        and roc20 < -0.05
    ):
        return "DOWNTREND"

    # ---------------------------------------------------------
    # Base
    # ---------------------------------------------------------

    if (
        position252 < 0.15
        and momentum >= 0
    ):
        return "BASE"

    # ---------------------------------------------------------
    # Extreme momentum
    # ---------------------------------------------------------

    if (
        momentum > 0.05
        and trend50 > 0
        and rsi_norm >= 0.75
    ):
        return "EXTREME_MOMENTUM"

    # ---------------------------------------------------------
    # Confirmed bull
    # ---------------------------------------------------------

    if (
        momentum > 0.05
        and trend50 > 0
        and position252 > 0.65
    ):
        return "CONFIRMED_BULL"

    # ---------------------------------------------------------
    # Early bull
    # ---------------------------------------------------------

    if (
        momentum > 0.05
        and trend50 > 0
    ):
        return "EARLY_BULL"

    # ---------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------

    if (
        momentum > 0
        and position252 > 0.25
    ):
        return "RECOVERY"

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------

    if position252 > 0.85:
        return "DISTRIBUTION"

    return "CONSOLIDATION"


# ---------------------------------------------------------------------------
# SIGNAL
# ---------------------------------------------------------------------------

def _signal(
    score: float,
    p20: float,
    rr: float,
    cycle: str,
    config: CurveConfig,
) -> str:
    """
    Convert CCE metrics into trading signal.
    """

    if (
        cycle == "DOWNTREND"
        and score < 50
    ):
        return "SELL"

    if (
        cycle == "EXTREME_MOMENTUM"
        and rr < config.minimum_reward_risk
    ):
        return "REDUCE"

    if (
        score >= config.strong_buy_score
        and p20 >= 0.65
        and rr >= config.strong_reward_risk
    ):
        return "STRONG_BUY"

    if (
        score >= config.buy_score
        and p20 >= config.buy_probability
        and rr >= config.minimum_reward_risk
    ):
        return "BUY"

    if score < config.sell_score:
        return "SELL"

    if score < config.reduce_score:
        return "REDUCE"

    return "WATCH"


# ---------------------------------------------------------------------------
# LEVELS
# ---------------------------------------------------------------------------

def _levels(
    price: float,
    horizon_stats: Dict[str, Dict[str, float]],
    current_features: pd.Series,
):
    """
    Estimate historical pullback / stop / take-profit levels.
    """

    stats20 = horizon_stats.get(
        "20",
        {},
    )

    mae_p10 = _scalar(
        stats20.get(
            "mae_p10",
            np.nan,
        )
    )

    mae_p25 = _scalar(
        stats20.get(
            "mae_p25",
            np.nan,
        )
    )

    mfe_p50 = _scalar(
        stats20.get(
            "mfe_p50",
            np.nan,
        )
    )

    mfe_p75 = _scalar(
        stats20.get(
            "mfe_p75",
            np.nan,
        )
    )

    mfe_p90 = _scalar(
        stats20.get(
            "mfe_p90",
            np.nan,
        )
    )

    atr_pct = _scalar(
        current_features.get(
            "atr_pct",
            np.nan,
        )
    )

    if not np.isfinite(atr_pct):
        atr_pct = 0.03

    if not np.isfinite(mae_p25):
        mae_p25 = -0.05

    if not np.isfinite(mae_p10):
        mae_p10 = -0.10

    if not np.isfinite(mfe_p50):
        mfe_p50 = 0.05

    if not np.isfinite(mfe_p75):
        mfe_p75 = 0.10

    if not np.isfinite(mfe_p90):
        mfe_p90 = 0.15

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------

    risk_pct = max(
        abs(mae_p25),
        2.0 * atr_pct,
        0.01,
    )

    stop_price = (
        price * (1.0 - risk_pct)
    )

    # ---------------------------------------------------------
    # Historical pullback zone
    #
    # This is NOT automatically a recommended entry.
    # It represents the historical adverse excursion zone.
    # ---------------------------------------------------------

    pullback_pct = max(
        abs(mae_p25),
        atr_pct,
    )

    entry_zone_low = (
        price * (1.0 - pullback_pct)
    )

    entry_zone_high = price

    # ---------------------------------------------------------
    # Take profit levels
    # ---------------------------------------------------------

    take_profit_50 = (
        price * (1.0 + max(mfe_p50, 0.0))
    )

    take_profit_75 = (
        price * (1.0 + max(mfe_p75, 0.0))
    )

    take_profit_90 = (
        price * (1.0 + max(mfe_p90, 0.0))
    )

    return {
        "entry_price": float(price),

        "entry_zone_low": float(
            entry_zone_low
        ),

        "entry_zone_high": float(
            entry_zone_high
        ),

        "stop_price": float(
            stop_price
        ),

        "take_profit_50": float(
            take_profit_50
        ),

        "take_profit_75": float(
            take_profit_75
        ),

        "take_profit_90": float(
            take_profit_90
        ),

        "risk_pct": float(
            risk_pct
        ),

        "historical_mae_p10": float(
            mae_p10
        ),

        "historical_mae_p25": float(
            mae_p25
        ),

        "historical_mfe_p50": float(
            mfe_p50
        ),

        "historical_mfe_p75": float(
            mfe_p75
        ),

        "historical_mfe_p90": float(
            mfe_p90
        ),
    }


# ---------------------------------------------------------------------------
# MAIN DATAFRAME ANALYSIS
# ---------------------------------------------------------------------------

def analyze_dataframe(
    data: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    config: Optional[CurveConfig] = None,
    ticker: str = "",
) -> Dict:
    """
    Analyze a ticker using the Characteristic Curve Engine.
    """

    if config is None:
        config = CurveConfig()

    data = _clean_ohlcv(data)

    if len(data) < config.min_history:
        raise ValueError(
            f"Insufficient history for {ticker}. "
            f"Required={config.min_history}, "
            f"available={len(data)}."
        )

    features = build_features(
        data,
        benchmark=benchmark,
    )

    # ---------------------------------------------------------
    # Valid current observations
    # ---------------------------------------------------------

    valid = features[
        FEATURES
    ].dropna()

    if valid.empty:
        raise ValueError(
            "No valid feature observations."
        )

    current_timestamp = valid.index[-1]

    # ---------------------------------------------------------
    # ORIGINAL POSITION
    #
    # This is important because the feature dataframe contains
    # leading NaNs caused by rolling indicators.
    # ---------------------------------------------------------

    current_pos = features.index.get_loc(
        current_timestamp
    )

    current_features = features.loc[
        current_timestamp,
        FEATURES,
    ].copy()

    current_price = _scalar(
        data["close"].iloc[current_pos]
    )

    # ---------------------------------------------------------
    # Causal standardization
    # ---------------------------------------------------------

    mu, sigma, covariance = (
        _causal_standardize(
            features[FEATURES],
            current_pos,
            config.covariance_window,
        )
    )

    # ---------------------------------------------------------
    # Historical observations
    #
    # CRITICAL FIX:
    #
    # Preserve original DataFrame positions BEFORE dropna().
    #
    # Previously:
    #
    #   hist_z = hist_z.dropna()
    #   enumerate(hist_z)
    #
    # caused the first valid historical observation to become
    # position 0, even though it might actually be position 252+
    # in the original price series.
    #
    # This contaminated all forward return / MFE / MAE calculations.
    # ---------------------------------------------------------

    hist = features[
        FEATURES
    ].iloc[:current_pos].copy()

    # Explicit original positional index
    hist["_original_position"] = np.arange(
        current_pos,
        dtype=int,
    )

    # Standardize
    hist_values = (
        hist[FEATURES] - mu
    ) / sigma

    hist_values = hist_values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Preserve original position through NaN filtering
    hist_values["_original_position"] = (
        hist["_original_position"]
    )

    valid_hist = hist_values.dropna(
        subset=FEATURES
    )

    if valid_hist.empty:
        raise ValueError(
            "No valid historical feature observations."
        )

    # ---------------------------------------------------------
    # Current standardized vector
    # ---------------------------------------------------------

    current_z = (
        (
            current_features
            - mu
        ) / sigma
    )

    current_z = current_z.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if current_z.isna().any():
        raise ValueError(
            "Current feature vector contains NaN values."
        )

    current_z_array = (
        current_z.to_numpy(
            dtype=float
        )
    )

    # ---------------------------------------------------------
    # Covariance matrix
    #
    # Use causal covariance returned by _causal_standardize().
    # ---------------------------------------------------------

    try:

        inv_cov = np.linalg.pinv(
            covariance
        )

    except np.linalg.LinAlgError:

        covariance = (
            covariance
            + np.eye(
                len(FEATURES)
            )
            * config.covariance_ridge
        )

        inv_cov = np.linalg.pinv(
            covariance
        )

    # ---------------------------------------------------------
    # Calculate distances
    #
    # IMPORTANT:
    # 'position' remains the ORIGINAL position in the price
    # dataframe.
    # ---------------------------------------------------------

    distances = {}

    for original_position, row in zip(
        valid_hist[
            "_original_position"
        ].astype(int).to_numpy(),

        valid_hist[
            FEATURES
        ].to_numpy(
            dtype=float
        ),
    ):

        distance = _mahalanobis(
            current_z_array,
            row,
            inv_cov,
        )

        distances[
            int(original_position)
        ] = float(distance)

    if not distances:
        raise ValueError(
            "Unable to calculate historical distances."
        )

    # ---------------------------------------------------------
    # Candidate filtering
    # ---------------------------------------------------------

    max_forward_horizon = max(
        HORIZONS
    )

    candidate_distances = [
        (
            position,
            distance,
        )

        for position, distance
        in distances.items()

        if (
            position
            + max_forward_horizon
            < len(data)
        )
    ]

    if len(candidate_distances) < config.min_matches:
        raise ValueError(
            "Not enough historical candidates "
            "with complete forward horizons. "
            f"Candidates={len(candidate_distances)}, "
            f"required={config.min_matches}."
        )

    candidate_distances.sort(
        key=lambda x: x[1]
    )

    # ---------------------------------------------------------
    # Quantile cutoff
    # ---------------------------------------------------------

    distance_values = np.asarray(
        [
            distance
            for _, distance
            in candidate_distances
        ],
        dtype=float,
    )

    cutoff = float(
        np.quantile(
            distance_values,
            config.match_quantile,
        )
    )

    eligible = [
        (
            position,
            distance,
        )

        for position, distance
        in candidate_distances

        if distance <= cutoff
    ]

    # Guarantee minimum matches
    if len(eligible) < config.min_matches:
        eligible = candidate_distances[
            :config.min_matches
        ]

    # ---------------------------------------------------------
    # Select spaced matches
    # ---------------------------------------------------------

    selected_positions = _select_spaced(
        eligible,
        min_gap_days=config.min_gap_days,
        max_matches=config.max_matches,
    )

    # ---------------------------------------------------------
    # If spacing removed too many matches, progressively relax
    # the requirement while preserving nearest-distance ordering.
    # ---------------------------------------------------------

    if len(selected_positions) < config.min_matches:

        fallback = [
            position
            for position, _ in candidate_distances
        ]

        selected_positions = []

        for position in fallback:

            if all(
                abs(
                    position
                    - selected_position
                ) >= max(
                    5,
                    config.min_gap_days // 2,
                )

                for selected_position
                in selected_positions
            ):

                selected_positions.append(
                    position
                )

            if (
                len(selected_positions)
                >= config.min_matches
            ):
                break

    # Final cap
    selected_positions = selected_positions[
        :config.max_matches
    ]

    # ---------------------------------------------------------
    # Forward statistics
    # ---------------------------------------------------------

    all_forward_stats = []

    selected_distances = []

    for position in selected_positions:

        stats = _forward_stats(
            data["high"],
            data["low"],
            data["close"],
            position,
        )

        if stats:

            all_forward_stats.append(
                stats
            )

            selected_distances.append(
                distances[position]
            )

    if len(all_forward_stats) < config.min_matches:
        raise ValueError(
            "Not enough valid historical matches "
            "after forward-stat calculation. "
            f"Matches={len(all_forward_stats)}, "
            f"required={config.min_matches}."
        )

    matches = len(
        all_forward_stats
    )

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------

    horizon_stats = (
        _aggregate_horizon_stats(
            all_forward_stats
        )
    )

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------

    score, score_breakdown, rr = (
        _score_result(
            current_features,
            horizon_stats,
            config,
        )
    )

    # ---------------------------------------------------------
    # Cycle
    # ---------------------------------------------------------

    cycle = _classify_cycle(
        current_features
    )

    # ---------------------------------------------------------
    # Signal
    # ---------------------------------------------------------

    stats20 = horizon_stats.get(
        "20",
        {},
    )

    p20 = _scalar(
        stats20.get(
            "prob_positive",
            np.nan,
        )
    )

    signal = _signal(
        score=score,
        p20=p20,
        rr=rr,
        cycle=cycle,
        config=config,
    )

    # ---------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------

    mean_distance = float(
        np.mean(
            selected_distances
        )
    )

    similarity = float(
        np.exp(
            -mean_distance / 5.0
        )
    )

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    sample_factor = min(
        1.0,
        matches
        / config.max_matches,
    )

    consistency = (
        p20
        if np.isfinite(p20)
        else 0.0
    )

    confidence = 100.0 * (
        0.35 * sample_factor
        + 0.25 * similarity
        + 0.20 * consistency
        + 0.20 * np.clip(
            consistency * 1.15,
            0.0,
            1.0,
        )
    )

    # ---------------------------------------------------------
    # Levels
    # ---------------------------------------------------------

    levels = _levels(
        price=current_price,
        horizon_stats=horizon_stats,
        current_features=current_features,
    )

    # ---------------------------------------------------------
    # Match details
    #
    # Useful for validation/debugging.
    # ---------------------------------------------------------

    match_details = []

    for position in selected_positions:

        if position not in distances:
            continue

        timestamp = data.index[
            position
        ]

        match_details.append(
            {
                "position": int(position),

                "date": str(
                    timestamp.date()
                ),

                "distance": float(
                    distances[position]
                ),

                "price": float(
                    _scalar(
                        data["close"].iloc[
                            position
                        ]
                    )
                ),
            }
        )

    match_details.sort(
        key=lambda x: x["distance"]
    )

    # ---------------------------------------------------------
    # Feature output
    # ---------------------------------------------------------

    feature_output = {}

    for feature in FEATURES:

        value = _scalar(
            current_features[
                feature
            ]
        )

        feature_output[
            feature
        ] = (
            None
            if not np.isfinite(value)
            else float(value)
        )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    result = {
        "engine": "Characteristic Curve Engine",

        "version": "1.1",

        "ticker": ticker,

        "as_of": str(
            current_timestamp.date()
        ),

        "price": float(
            current_price
        ),

        "signal": signal,

        "score": float(
            score
        ),

        "confidence": float(
            confidence
        ),

        "cycle": cycle,

        "matches": int(
            matches
        ),

        "mean_match_distance": float(
            mean_distance
        ),

        "similarity": float(
            similarity * 100.0
        ),

        "features": feature_output,

        "score_breakdown": score_breakdown,

        "horizons": horizon_stats,

        "reward_risk_20d": float(
            rr
        ),

        "levels": levels,

        "matches_detail": match_details,

        "config": asdict(
            config
        ),
    }

    return result


# ---------------------------------------------------------------------------
# PUBLIC ANALYZE FUNCTION
# ---------------------------------------------------------------------------

def analyze(
    ticker: str,
    benchmark: Optional[str] = None,
    period: str = "5y",
    interval: str = "1d",
) -> Dict:
    """
    Public analysis entry point.
    """

    config = CurveConfig(
        period=period,
        interval=interval,
        benchmark=benchmark,
    )

    data = download_history(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    benchmark_close = None

    if benchmark:

        benchmark_data = download_history(
            ticker=benchmark,
            period=period,
            interval=interval,
        )

        benchmark_close = (
            benchmark_data["close"]
        )

    return analyze_dataframe(
        data=data,
        benchmark=benchmark_close,
        config=config,
        ticker=ticker,
    )


# ---------------------------------------------------------------------------
# JSON SERIALIZATION
# ---------------------------------------------------------------------------

def _json_safe(value):
    """
    Convert numpy/pandas values into JSON-safe objects.
    """

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(v)
            for v in value
        ]

    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Characteristic Curve Engine"
        )
    )

    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. TQQQ",
    )

    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark ticker, e.g. QQQ",
    )

    parser.add_argument(
        "--period",
        default="5y",
        help="Yahoo Finance period, e.g. 2y, 5y, 10y",
    )

    parser.add_argument(
        "--interval",
        default="1d",
        help="Yahoo Finance interval",
    )

    args = parser.parse_args()

    try:

        result = analyze(
            ticker=args.ticker,
            benchmark=args.benchmark,
            period=args.period,
            interval=args.interval,
        )

        print()
        print(
            "=================================================="
        )

        print(
            "Characteristic Curve Engine"
        )

        print(
            "=================================================="
        )

        print(
            f"Ticker:       {result['ticker']}"
        )

        print(
            f"As of:        {result['as_of']}"
        )

        print(
            f"Price:        {result['price']:.3f}"
        )

        print(
            f"Signal:       {result['signal']}"
        )

        print(
            f"Score:        {result['score']:.2f}"
        )

        print(
            f"Confidence:   {result['confidence']:.2f}"
        )

        print(
            f"Cycle:        {result['cycle']}"
        )

        print(
            f"Matches:      {result['matches']}"
        )

        print(
            "Mean distance:"
            f" {result['mean_match_distance']:.4f}"
        )

        print(
            f"Similarity:   {result['similarity']:.2f}"
        )

        print()

        print(
            "Features:"
        )

        print(
            json.dumps(
                result["features"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print()

        print(
            "Score breakdown:"
        )

        print(
            json.dumps(
                result["score_breakdown"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print()

        print(
            "Horizons:"
        )

        print(
            json.dumps(
                result["horizons"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print()

        print(
            f"Reward/Risk 20d: "
            f"{result['reward_risk_20d']:.3f}"
        )

        print()

        print(
            "Levels:"
        )

        print(
            json.dumps(
                result["levels"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print()

        print(
            "Top historical matches:"
        )

        for match in result[
            "matches_detail"
        ][:10]:

            print(
                f"  {match['date']} | "
                f"price={match['price']:.3f} | "
                f"distance={match['distance']:.4f} | "
                f"position={match['position']}"
            )

        print()

        print(
            "Config:"
        )

        print(
            json.dumps(
                result["config"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "=================================================="
        )

    except Exception as exc:

        print(
            f"Characteristic Curve Engine error: {exc}"
        )

        raise


if __name__ == "__main__":
    main()