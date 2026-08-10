"""
Central signal engine.

For normal instruments, preserves the existing RSI + MACD logic.
For SOXL, delegates to the specialized quantitative cycle analyzer.
"""

import ta


def _generate_standard_signal(df):
    """Existing signal logic for non-SOXL instruments."""
    if df.empty:
        return "NO_DATA"

    if len(df) < 30:
        return "INSUFFICIENT_DATA"

    data = df.copy()

    close_prices = data["Close"]

    rsi_indicator = ta.momentum.RSIIndicator(
        close=close_prices,
        window=14
    )

    data["rsi"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(close_prices)

    data["macd"] = macd_indicator.macd()
    data["macd_signal"] = macd_indicator.macd_signal()

    rsi = data["rsi"].iloc[-1]
    macd = data["macd"].iloc[-1]
    macd_signal = data["macd_signal"].iloc[-1]

    if buy_signal := (
        rsi < 30 and
        macd > macd_signal
    ):
        return "BUY"

    if sell_signal := (
        rsi > 70 and
        macd < macd_signal
    ):
        return "SELL"

    return "HOLD"


def generate_signal(df, ticker=None, soxl_result=None):
    """
    Generate a signal.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV data already downloaded by the scanner.
    ticker : str, optional
        Symbol being analyzed.
    soxl_result : dict, optional
        Result from soxl_cycle_analyzer.analyze(). Supplying it avoids
        downloading SOXL/sector data twice.

    Returns
    -------
    str
        NO_DATA / INSUFFICIENT_DATA / BUY / STRONG_BUY / WATCH /
        HOLD / REDUCE / SELL
    """

    if ticker and ticker.upper() == "SOXL":
        if soxl_result is None:
            from app.ai.soxl_cycle_analyzer import (
                AnalyzerConfig,
                analyze,
            )

            soxl_result = analyze(
                AnalyzerConfig(
                    ticker="SOXL",
                    benchmark="SOXX",
                    period="2y",
                    interval="1d",
                )
            )

        return soxl_result["signal"]

    return _generate_standard_signal(df)
