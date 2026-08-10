"""
SOXL Cycle Analyzer
===================

Quantitative cycle/regime analyzer for Direxion Daily Semiconductor Bull 3X
Shares (SOXL).

Designed to integrate with the AI Trader project.

Outputs:
- Technical indicators: SMA20/50/200, RSI14, MACD, ATR14
- Volume Z-score
- Drawdown from 20/50/252-day highs
- Recovery from 20/50/252-day lows
- SOXL/SOXX relative strength
- Cross-asset confirmation: NVDA, MU, AMD, QQQ
- Cycle classification
- SOXL score (0-100)
- BUY / HOLD / REDUCE / SELL signal
- Dynamic entry, stop and take-profit levels

IMPORTANT:
SOXL targets 3x the DAILY performance of its benchmark. This analyzer
therefore treats SOXL as a tactical instrument and uses SOXX/SOX proxies
for regime confirmation.

Dependencies:
    pip install yfinance pandas numpy

Usage:
    python soxl_cycle_analyzer.py
    python soxl_cycle_analyzer.py --period 2y
    python soxl_cycle_analyzer.py --json
    python soxl_cycle_analyzer.py --ticker SOXL
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKER = "SOXL"
DEFAULT_BENCHMARK = "SOXX"
CONFIRMATION_TICKERS = {
    "NVDA": "NVDA",
    "MU": "MU",
    "AMD": "AMD",
    "QQQ": "QQQ",
}


@dataclass
class AnalyzerConfig:
    ticker: str = DEFAULT_TICKER
    benchmark: str = DEFAULT_BENCHMARK
    period: str = "2y"
    interval: str = "1d"

    sma_fast: int = 20
    sma_medium: int = 50
    sma_slow: int = 200

    rsi_period: int = 14
    atr_period: int = 14

    volume_window: int = 20
    drawdown_short: int = 20
    drawdown_medium: int = 50
    drawdown_long: int = 252

    # Cycle / score thresholds
    capitulation_drawdown: float = -0.30
    severe_drawdown: float = -0.40
    extreme_drawdown: float = -0.50

    extreme_rsi: float = 75.0
    bullish_rsi: float = 50.0
    volume_confirmation_z: float = 1.0

    breakout_lookback: int = 20

    # Position management
    initial_risk_fraction: float = 0.01
    stop_atr_multiple: float = 2.5
    tp1_pct: float = 0.20
    tp2_pct: float = 0.35
    tp3_rsi: float = 75.0


def download_history(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted daily OHLCV data and normalize the yfinance format."""
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No market data returned by yfinance.")

    # MultiIndex can appear for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        frames = {}
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(-1):
                continue
            frames[ticker] = raw.xs(ticker, axis=1, level=-1)
        return frames

    return {tickers[0]: raw}


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing OHLCV columns: {missing}")

    df = df[required].dropna()
    return df


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # If there has been no loss, RSI should be 100.
    result = result.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    histogram = line - signal
    return line, signal, histogram


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def add_indicators(df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    df = _clean_frame(df)

    df["sma20"] = sma(df["close"], cfg.sma_fast)
    df["sma50"] = sma(df["close"], cfg.sma_medium)
    df["sma200"] = sma(df["close"], cfg.sma_slow)

    df["rsi14"] = rsi(df["close"], cfg.rsi_period)

    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])

    df["atr14"] = atr(df, cfg.atr_period)
    df["atr_pct"] = df["atr14"] / df["close"]

    df["volume_z"] = zscore(df["volume"], cfg.volume_window)

    for window in (
        cfg.drawdown_short,
        cfg.drawdown_medium,
        cfg.drawdown_long,
    ):
        rolling_high = df["close"].rolling(window).max()
        rolling_low = df["close"].rolling(window).min()

        df[f"high_{window}"] = rolling_high
        df[f"low_{window}"] = rolling_low

        df[f"drawdown_{window}"] = df["close"] / rolling_high - 1.0
        df[f"recovery_{window}"] = df["close"] / rolling_low - 1.0

    df["sma20_distance"] = df["close"] / df["sma20"] - 1.0
    df["sma50_distance"] = df["close"] / df["sma50"] - 1.0

    df["breakout_high"] = (
        df["close"].rolling(cfg.breakout_lookback).max().shift(1)
    )
    df["breakout"] = df["close"] > df["breakout_high"]

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    return df


def latest_row(df: pd.DataFrame) -> pd.Series:
    valid = df.dropna(subset=["close"])
    if valid.empty:
        raise RuntimeError("No valid rows available.")
    return valid.iloc[-1]


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def relative_strength(
    soxl_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            soxl_df["close"].rename("soxl"),
            benchmark_df["close"].rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    combined["ratio"] = combined["soxl"] / combined["benchmark"]
    combined["ratio_sma20"] = combined["ratio"].rolling(20).mean()
    combined["ratio_sma50"] = combined["ratio"].rolling(50).mean()
    combined["ratio_momentum_20d"] = combined["ratio"].pct_change(20)

    return combined


def confirmation_state(
    frames: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, object]]:
    """Evaluate whether each confirmation asset is bullish."""
    result = {}

    for name, df in frames.items():
        row = latest_row(df)

        above20 = row["close"] > row["sma20"] if pd.notna(row["sma20"]) else False
        above50 = row["close"] > row["sma50"] if pd.notna(row["sma50"]) else False
        above200 = (
            row["close"] > row["sma200"] if pd.notna(row["sma200"]) else False
        )
        macd_positive = (
            row["macd_hist"] > 0 if pd.notna(row["macd_hist"]) else False
        )
        rsi_bullish = (
            row["rsi14"] >= 50 if pd.notna(row["rsi14"]) else False
        )

        score = sum(
            [
                above20,
                above50,
                above200,
                macd_positive,
                rsi_bullish,
            ]
        )

        result[name] = {
            "close": safe_float(row["close"]),
            "rsi14": safe_float(row["rsi14"]),
            "above_sma20": above20,
            "above_sma50": above50,
            "above_sma200": above200,
            "macd_positive": macd_positive,
            "rsi_bullish": rsi_bullish,
            "bullish_points": score,
        }

    return result


def calculate_score(
    soxl: pd.DataFrame,
    soxx: pd.DataFrame,
    confirmations: Dict[str, Dict[str, object]],
    rs: pd.DataFrame,
    cfg: AnalyzerConfig,
) -> tuple[int, Dict[str, float]]:
    """Calculate a 0-100 tactical SOXL score."""
    s = latest_row(soxl)
    b = latest_row(soxx)
    r = latest_row(rs)

    points: Dict[str, float] = {}

    # 20 points: SOXX trend
    soxx_points = 0.0
    if b["close"] > b["sma20"]:
        soxx_points += 5
    if b["close"] > b["sma50"]:
        soxx_points += 7
    if b["sma50"] > b["sma200"]:
        soxx_points += 5
    if b["macd_hist"] > 0:
        soxx_points += 3
    points["SOXX trend"] = soxx_points

    # 15 points: SOX proxy momentum via SOXX
    momentum_points = 0.0
    if b["return_5d"] > 0:
        momentum_points += 4
    if b["return_20d"] > 0:
        momentum_points += 5
    if b["rsi14"] >= 50:
        momentum_points += 3
    if b["rsi14"] >= 60:
        momentum_points += 3
    points["SOXX momentum"] = momentum_points

    # Confirmation assets
    weights = {"NVDA": 12, "MU": 12, "AMD": 8, "QQQ": 8}
    for ticker, weight in weights.items():
        state = confirmations[ticker]
        bullish = float(state["bullish_points"]) / 5.0
        points[ticker] = round(weight * bullish, 2)

    # 7 points: SOXL RSI
    rsi_points = 0.0
    if s["rsi14"] >= 50:
        rsi_points += 3
    if s["rsi14"] >= 60:
        rsi_points += 2
    if s["rsi14"] >= 70:
        rsi_points += 2
    points["RSI"] = rsi_points

    # 6 points: MACD
    macd_points = 0.0
    if s["macd_hist"] > 0:
        macd_points += 4
    if s["macd"] > s["macd_signal"]:
        macd_points += 2
    points["MACD"] = macd_points

    # 7 points: Volume
    volume_points = 0.0
    if s["volume_z"] >= 0:
        volume_points += 2
    if s["volume_z"] >= cfg.volume_confirmation_z:
        volume_points += 3
    if s["volume_z"] >= 2:
        volume_points += 2
    points["Volume"] = volume_points

    # 5 points: volatility / ATR
    # Moderate volatility is rewarded; extreme volatility is penalized.
    atr_pct = float(s["atr_pct"])
    volatility_points = 5.0
    if atr_pct > 0.12:
        volatility_points = 2.0
    elif atr_pct > 0.18:
        volatility_points = 0.0
    points["ATR / volatility"] = volatility_points

    # Relative strength bonus / penalty is applied by modifying the
    # SOXX trend component rather than creating an extra category.
    ratio_bullish = (
        r["ratio"] > r["ratio_sma20"]
        and r["ratio_sma20"] > r["ratio_sma50"]
    )
    if ratio_bullish:
        points["SOXX trend"] = min(20.0, points["SOXX trend"] + 2.0)
    elif r["ratio"] < r["ratio_sma20"]:
        points["SOXX trend"] = max(0.0, points["SOXX trend"] - 2.0)

    score = int(round(sum(points.values())))
    return max(0, min(100, score)), points


def classify_cycle(
    soxl: pd.DataFrame,
    soxx: pd.DataFrame,
    score: int,
    cfg: AnalyzerConfig,
) -> str:
    s = latest_row(soxl)
    b = latest_row(soxx)

    dd = float(s[f"drawdown_{cfg.drawdown_long}"])
    recovery = float(s[f"recovery_{cfg.drawdown_long}"])

    if dd <= cfg.extreme_drawdown and score < 45:
        return "CAPITULATION"

    if (
        dd <= cfg.severe_drawdown
        and score >= 45
        and s["close"] > s["sma20"]
    ):
        return "EARLY_REVERSAL"

    if (
        s["close"] > s["sma20"]
        and b["close"] > b["sma50"]
        and b["sma50"] > b["sma200"]
        and score >= 65
    ):
        if score >= 80 or s["rsi14"] >= cfg.extreme_rsi:
            return "EXTREME_MOMENTUM"
        return "CONFIRMED_BULL"

    if (
        s["close"] > s["sma20"]
        and s["macd_hist"] > 0
        and score >= 55
    ):
        return "EARLY_BULL"

    if (
        s["close"] > s["sma20"]
        and s["return_20d"] > 0
        and score >= 45
    ):
        return "RECOVERY"

    if s["close"] < s["sma20"] and s["close"] < s["sma50"]:
        return "DOWNTREND"

    if recovery < 0.05:
        return "BASE"

    return "CONSOLIDATION"


def generate_signal(
    cycle: str,
    score: int,
    soxl: pd.DataFrame,
    soxx: pd.DataFrame,
    confirmations: Dict[str, Dict[str, object]],
) -> str:
    s = latest_row(soxl)
    b = latest_row(soxx)

    # Hard risk-off condition.
    if (
        s["close"] < s["sma50"]
        and b["close"] < b["sma50"]
        and score < 45
    ):
        return "SELL"

    if cycle == "CAPITULATION":
        return "WATCH"

    if cycle in {"EARLY_REVERSAL", "RECOVERY", "EARLY_BULL"}:
        return "BUY" if score >= 60 else "WATCH"

    if cycle == "CONFIRMED_BULL":
        return "STRONG_BUY" if score >= 70 else "BUY"

    if cycle == "EXTREME_MOMENTUM":
        return "REDUCE"

    if cycle == "DOWNTREND":
        return "SELL"

    return "HOLD"


def calculate_levels(
    soxl: pd.DataFrame,
    signal: str,
    cfg: AnalyzerConfig,
) -> Dict[str, Optional[float]]:
    s = latest_row(soxl)

    price = float(s["close"])
    atr_value = float(s["atr14"])

    stop = price - cfg.stop_atr_multiple * atr_value

    # Dynamic targets are deliberately calculated from current price,
    # not fixed SOXL price levels.
    tp1 = price * (1 + cfg.tp1_pct)
    tp2 = price * (1 + cfg.tp2_pct)
    tp3 = price + 3.0 * atr_value

    return {
        "entry_price": safe_float(price),
        "dynamic_stop": safe_float(stop),
        "take_profit_1": safe_float(tp1),
        "take_profit_2": safe_float(tp2),
        "take_profit_3_atr": safe_float(tp3),
    }


def analyze(cfg: AnalyzerConfig) -> Dict[str, object]:
    tickers = [cfg.ticker, cfg.benchmark] + list(CONFIRMATION_TICKERS.values())
    raw = download_history(tickers, cfg.period, cfg.interval)

    frames: Dict[str, pd.DataFrame] = {
        ticker: add_indicators(raw[ticker], cfg)
        for ticker in tickers
        if ticker in raw
    }

    missing = [t for t in tickers if t not in frames]
    if missing:
        raise RuntimeError(f"Missing market data for: {missing}")

    soxl = frames[cfg.ticker]
    soxx = frames[cfg.benchmark]

    rs = relative_strength(soxl, soxx)

    confirmations = confirmation_state(
        {ticker: frames[ticker] for ticker in CONFIRMATION_TICKERS}
    )

    score, score_breakdown = calculate_score(
        soxl, soxx, confirmations, rs, cfg
    )

    cycle = classify_cycle(soxl, soxx, score, cfg)
    signal = generate_signal(
        cycle, score, soxl, soxx, confirmations
    )

    s = latest_row(soxl)
    b = latest_row(soxx)
    r = latest_row(rs)

    levels = calculate_levels(soxl, signal, cfg)

    result = {
        "analyzer": "SOXL Cycle Analyzer",
        "as_of": str(soxl.index[-1].date()),
        "ticker": cfg.ticker,
        "benchmark": cfg.benchmark,
        "price": safe_float(s["close"]),
        "score": score,
        "cycle": cycle,
        "signal": signal,
        "indicators": {
            "sma20": safe_float(s["sma20"]),
            "sma50": safe_float(s["sma50"]),
            "sma200": safe_float(s["sma200"]),
            "rsi14": safe_float(s["rsi14"]),
            "macd": safe_float(s["macd"]),
            "macd_signal": safe_float(s["macd_signal"]),
            "macd_hist": safe_float(s["macd_hist"]),
            "atr14": safe_float(s["atr14"]),
            "atr_pct": safe_float(s["atr_pct"]),
            "volume_z": safe_float(s["volume_z"]),
            "sma20_distance": safe_float(s["sma20_distance"]),
            "sma50_distance": safe_float(s["sma50_distance"]),
            "return_1d": safe_float(s["return_1d"]),
            "return_5d": safe_float(s["return_5d"]),
            "return_20d": safe_float(s["return_20d"]),
        },
        "drawdowns": {
            "20d": safe_float(s["drawdown_20"]),
            "50d": safe_float(s["drawdown_50"]),
            "252d": safe_float(s["drawdown_252"]),
        },
        "recoveries": {
            "20d": safe_float(s["recovery_20"]),
            "50d": safe_float(s["recovery_50"]),
            "252d": safe_float(s["recovery_252"]),
        },
        "relative_strength": {
            "ratio_soxl_soxx": safe_float(r["ratio"]),
            "ratio_sma20": safe_float(r["ratio_sma20"]),
            "ratio_sma50": safe_float(r["ratio_sma50"]),
            "ratio_momentum_20d": safe_float(r["ratio_momentum_20d"]),
        },
        "soxx": {
            "price": safe_float(b["close"]),
            "sma20": safe_float(b["sma20"]),
            "sma50": safe_float(b["sma50"]),
            "sma200": safe_float(b["sma200"]),
            "rsi14": safe_float(b["rsi14"]),
            "macd_hist": safe_float(b["macd_hist"]),
            "return_20d": safe_float(b["return_20d"]),
        },
        "confirmations": confirmations,
        "score_breakdown": score_breakdown,
        "levels": levels,
        "risk_notes": [
            "SOXL targets 3x the DAILY benchmark return; long-term results can differ materially from 3x.",
            "A large SOXL drawdown alone is not a buy signal.",
            "Use SOXX and confirmation assets to validate the semiconductor regime.",
            "Dynamic stops/targets should be recalculated daily.",
        ],
    }

    return result


def print_report(result: Dict[str, object]) -> None:
    print("\n" + "=" * 72)
    print("SOXL CYCLE ANALYZER")
    print("=" * 72)

    print(f"As of       : {result['as_of']}")
    print(f"Price       : ${result['price']:.2f}")
    print(f"Score       : {result['score']}/100")
    print(f"Cycle       : {result['cycle']}")
    print(f"Signal      : {result['signal']}")

    ind = result["indicators"]
    print("\nTECHNICAL")
    print(f"SMA20       : ${ind['sma20']:.2f}")
    print(f"SMA50       : ${ind['sma50']:.2f}")
    print(f"SMA200      : ${ind['sma200']:.2f}")
    print(f"RSI14       : {ind['rsi14']:.2f}")
    print(f"MACD Hist   : {ind['macd_hist']:.4f}")
    print(f"ATR14       : ${ind['atr14']:.2f}")
    print(f"ATR %       : {ind['atr_pct'] * 100:.2f}%")
    print(f"Volume Z    : {ind['volume_z']:.2f}")

    dd = result["drawdowns"]
    print("\nDRAWDOWN")
    print(f"20d         : {dd['20d'] * 100:.2f}%")
    print(f"50d         : {dd['50d'] * 100:.2f}%")
    print(f"252d        : {dd['252d'] * 100:.2f}%")

    rs = result["relative_strength"]
    print("\nRELATIVE STRENGTH")
    print(f"SOXL/SOXX   : {rs['ratio_soxl_soxx']:.6f}")
    print(f"RS SMA20    : {rs['ratio_sma20']:.6f}")
    print(f"RS SMA50    : {rs['ratio_sma50']:.6f}")
    print(f"RS 20d mom  : {rs['ratio_momentum_20d'] * 100:.2f}%")

    print("\nCONFIRMATIONS")
    for ticker, state in result["confirmations"].items():
        print(
            f"{ticker:5s}: ${state['close']:.2f} | "
            f"RSI {state['rsi14']:.1f} | "
            f"bullish {state['bullish_points']}/5"
        )

    print("\nSCORE BREAKDOWN")
    for key, value in result["score_breakdown"].items():
        print(f"{key:20s}: {value:5.1f}")

    levels = result["levels"]
    print("\nRISK / TRADE LEVELS")
    print(f"Entry       : ${levels['entry_price']:.2f}")
    print(f"Stop        : ${levels['dynamic_stop']:.2f}")
    print(f"TP1         : ${levels['take_profit_1']:.2f}")
    print(f"TP2         : ${levels['take_profit_2']:.2f}")
    print(f"TP3 (ATR)   : ${levels['take_profit_3_atr']:.2f}")

    print("\n" + "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative SOXL cycle/regime analyzer."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the report.",
    )

    args = parser.parse_args()

    cfg = AnalyzerConfig(
        ticker=args.ticker,
        benchmark=args.benchmark,
        period=args.period,
        interval=args.interval,
    )

    result = analyze(cfg)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
