"""
Watchlist scanner.

Normal stocks/ETFs:
    Uses the existing RSI/MACD/SMA scoring logic.

SOXL:
    Uses the specialized SOXL Cycle Analyzer through signal_engine.py.
"""

import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from app.strategies.signal_engine import generate_signal


def _normalize_ohlcv(df):
    """Normalize yfinance MultiIndex output to standard OHLCV columns."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    price_names = {"open", "high", "low", "close", "adj close", "volume"}
    flattened = []
    for col in df.columns:
        candidates = [str(x).lower() for x in col]
        match = next((x for x in candidates if x in price_names), None)
        flattened.append(match if match else candidates[-1])

    out = df.copy()
    out.columns = flattened

    # Keep exactly one column per OHLCV field.
    selected = {}
    for name in ["open", "high", "low", "close", "volume"]:
        matches = [c for c in out.columns if c == name]
        if matches:
            selected[name] = out[matches[0]]

    if len(selected) == 5:
        return pd.DataFrame(selected, index=out.index)

    # Restore conventional capitalization expected by existing code.
    return out


def analyze_buy_opportunity(df):
    """
    Existing watchlist analysis for non-SOXL instruments.

    Kept compatible with the previous return structure.
    """

    close_prices = pd.Series(
        df["Close"].values.flatten()
    )

    # RSI
    rsi_indicator = RSIIndicator(close_prices)
    rsi = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close_prices)
    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()

    # SMA
    sma_20 = SMAIndicator(
        close_prices,
        window=20
    ).sma_indicator()

    sma_50 = SMAIndicator(
        close_prices,
        window=50
    ).sma_indicator()

    current_price = float(close_prices.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    current_macd = float(macd.iloc[-1])
    current_signal = float(macd_signal.iloc[-1])
    current_sma_20 = float(sma_20.iloc[-1])
    current_sma_50 = float(sma_50.iloc[-1])

    signal = "HOLD"
    confidence = "LOW"
    reasons = []
    score = 0

    if current_rsi < 30:
        score += 1
        reasons.append("RSI Oversold")

    if current_macd > current_signal:
        score += 1
        reasons.append("MACD Bullish")

    if current_price > current_sma_20:
        score += 1
        reasons.append("Above SMA20")

    if current_price > current_sma_50:
        score += 1
        reasons.append("Above SMA50")

    if score >= 3:
        signal = "BUY"
        confidence = "HIGH"
    elif score == 2:
        signal = "WATCH"
        confidence = "MEDIUM"

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "rsi": round(current_rsi, 2),
        "macd": round(current_macd, 4),
        "macd_signal": round(current_signal, 4),
        "sma_20": round(current_sma_20, 2),
        "sma_50": round(current_sma_50, 2),
        "current_price": round(current_price, 2),
    }


def analyze_soxl():
    """
    Run the complete SOXL cycle analysis.

    Returns the complete quantitative result so it can later be consumed
    by Telegram alerts, portfolio logic, or main.py.
    """
    from app.ai.soxl_cycle_analyzer import (
        AnalyzerConfig,
        analyze,
    )

    config = AnalyzerConfig(
        ticker="SOXL",
        benchmark="SOXX",
        period="2y",
        interval="1d",
    )

    return analyze(config)


def _print_soxl_result(result, buy_target=None):
    """Human-readable SOXL console output."""

    price = result["price"]
    score = result["score"]
    cycle = result["cycle"]
    signal = result["signal"]

    indicators = result["indicators"]
    drawdowns = result["drawdowns"]
    levels = result["levels"]

    print("\n" + "=" * 68)
    print("SOXL CYCLE ANALYSIS")
    print("=" * 68)

    print(f"SOXL       => {signal}")
    print(f"Price      : ${price:.2f}")
    print(f"Score      : {score}/100")
    print(f"Cycle      : {cycle}")

    print("\nTECHNICAL")
    print(f"SMA20      : ${indicators['sma20']:.2f}")
    print(f"SMA50      : ${indicators['sma50']:.2f}")
    print(f"SMA200     : ${indicators['sma200']:.2f}")
    print(f"RSI14      : {indicators['rsi14']:.2f}")
    print(f"MACD Hist  : {indicators['macd_hist']:.4f}")
    print(f"ATR14      : ${indicators['atr14']:.2f}")

    print("\nDRAWDOWN")
    print(f"20D        : {drawdowns['20d'] * 100:.2f}%")
    print(f"50D        : {drawdowns['50d'] * 100:.2f}%")
    print(f"252D       : {drawdowns['252d'] * 100:.2f}%")

    print("\nTRADE LEVELS")
    print(f"Entry      : ${levels['entry_price']:.2f}")
    print(f"Stop       : ${levels['dynamic_stop']:.2f}")
    print(f"TP1        : ${levels['take_profit_1']:.2f}")
    print(f"TP2        : ${levels['take_profit_2']:.2f}")
    print(f"TP3        : ${levels['take_profit_3_atr']:.2f}")

    if buy_target is not None:
        print(f"Watchlist target: ${buy_target:.2f}")

    print("=" * 68)


def scan_buy_opportunity(stock):
    """
    Scan one watchlist item.

    SOXL follows the specialized cycle analyzer.
    All other symbols retain the previous scanner behavior.
    """

    ticker = str(stock["ticker"]).upper()
    buy_target = float(stock.get("buy_target", 0))

    # ============================================================
    # SPECIALIZED SOXL ENGINE
    # ============================================================
    if ticker == "SOXL":
        try:
            result = analyze_soxl()
        except Exception as exc:
            print(f"{ticker} => SOXL analyzer error: {exc}")
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "error": str(exc),
            }

        _print_soxl_result(
            result,
            buy_target=buy_target if buy_target else None
        )

        # Optional legacy target-price alert.
        current_price = result["price"]

        if buy_target and current_price <= buy_target:
            print(
                f"BUY ALERT: {ticker} reached "
                f"watchlist target price"
            )

        # Keep complete result available to main/alerts.
        result["ticker"] = ticker
        result["watchlist_buy_target"] = buy_target

        return result

    # ============================================================
    # EXISTING ENGINE FOR ALL OTHER WATCHLIST SYMBOLS
    # ============================================================

    df = yf.download(
        ticker,
        period="6mo",
        progress=False,
        auto_adjust=True
    )
    df = _normalize_ohlcv(df)
    # Existing analyzer expects capitalized OHLCV names.
    df.columns = [str(c).title() for c in df.columns]

    if df.empty:
        print(f"{ticker} => No market data")
        return {
            "ticker": ticker,
            "signal": "NO_DATA",
        }

    result = analyze_buy_opportunity(df)

    # Make the central signal_engine the final authority.
    central_signal = generate_signal(
        df,
        ticker=ticker
    )

    result["signal_engine_signal"] = central_signal

    # Keep the existing scanner signal for compatibility.
    # For normal symbols we don't change the established behavior.
    current_price = result["current_price"]

    print(
        f"{ticker} => "
        f"{result['signal']} | "
        f"Engine={central_signal} | "
        f"Price={current_price:.2f} | "
        f"Target={buy_target:.2f}"
    )

    if current_price <= buy_target:
        print(
            f"BUY ALERT: "
            f"{ticker} reached "
            f"target price"
        )

    result["ticker"] = ticker
    result["watchlist_buy_target"] = buy_target

    return result
