import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import (
    MACD , 
    SMAIndicator
)

def analyze_buy_opportunity(df):

   # ====================================
   # CLOSE PRICES
   # ====================================

    close_prices = pd.Series(
    df["Close"].values.flatten()
)


   # ====================================
   # RSI
   # ====================================

    rsi_indicator = RSIIndicator(close_prices)

    rsi = rsi_indicator.rsi()

    # ====================================
    # MACD
    # ====================================

    macd_indicator = MACD(close_prices)

    macd = macd_indicator.macd()

    macd_signal = macd_indicator.macd_signal()

    # ====================================
    # SMA
    # ==================================== 

    sma_20_indicator = SMAIndicator(
        close_prices,
        window=20
    )

    sma_20 = sma_20_indicator.sma_indicator()

    sma_50_indicator = SMAIndicator(
        close_prices,
        window=50
    )

    sma_50 = sma_50_indicator.sma_indicator()

    # ====================================
    # CURRENT VALUES
    # ====================================

    current_price = float(close_prices.iloc[-1])

    current_rsi = float(rsi.iloc[-1])

    current_macd = float(macd.iloc[-1])

    current_signal = float(macd_signal.iloc[-1])

    current_sma_20 = float(sma_20.iloc[-1])

    current_sma_50 = float(sma_50.iloc[-1])

    # ====================================
    # BASIC SIGNAL ENGINE
    # ====================================

    signal = "HOLD"

    confidence = "LOW"

    reasons = []

    score = 0

    # RSI oversold
    if current_rsi < 30:

        score += 1

        reasons.append("RSI Oversold")

    # MACD bullish
    if current_macd > current_signal:

        score += 1

        reasons.append("MACD Bullish")

    # Above SMA20
    if current_price > current_sma_20:

        score += 1

        reasons.append(
            "Above SMA20"
        )

    # Above SMA50
    if current_price > current_sma_50:

        score += 1

        reasons.append(
            "Above SMA50"
        )     

    # ====================================
    # FINAL SIGNAL
    # ====================================
    # Scoring
    if score >= 3:

        signal = "BUY"
        confidence = "HIGH"

    elif score == 2:

        signal = "WATCH"
        confidence = "MEDIUM"

    # ====================================
    # RETURN DATA
    # ====================================    

    return {

        "signal": signal,

        "confidence": confidence,

        "score": score,

        "reasons": reasons,

        "rsi": round(current_rsi, 2),

        "macd": round(current_macd, 4),

        "macd_signal": round(current_signal, 4),

        "sma_20": round(current_sma_20, 2),

        "sma_50": round(current_sma_50,2),

        "current_price": round(current_price,2)

    }

import yfinance as yf


def scan_buy_opportunity(stock):

    ticker = stock["ticker"]

    buy_target = float(
        stock["buy_target"]
    )

    df = yf.download(
        ticker,
        period="6mo",
        progress=False,
        auto_adjust=True
    )

    if df.empty:

        print(
            f"{ticker} => "
            f"No market data"
        )

        return

    result = analyze_buy_opportunity(df)

    current_price = result["current_price"]

    print(
        f"{ticker} => "
        f"{result['signal']} | "
        f"Price={current_price:.2f} | "
        f"Target={buy_target:.2f}"
    )

    #
    # Ideal buy price alert
    #
    if current_price <= buy_target:

        print(
            f"BUY ALERT: "
            f"{ticker} reached "
            f"target price"
        )