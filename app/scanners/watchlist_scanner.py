from ta.momentum import RSIIndicator
from ta.trend import MACD

def analyze_buy_opportunity(df):

    close_prices = df["Close"]

    rsi = RSIIndicator(close_prices).rsi()

    macd_indicator = MACD(close_prices)

    macd = macd_indicator.macd()

    macd_signal = macd_indicator.macd_signal()

    current_rsi = float(rsi.iloc[-1])

    current_macd = float(macd.iloc[-1])

    current_signal = float(macd_signal.iloc[-1])

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

    # Scoring
    if score == 2:

        signal = "BUY"
        confidence = "HIGH"

    elif score == 1:

        signal = "WATCH"
        confidence = "MEDIUM"

    return {

        "signal": signal,

        "confidence": confidence,

        "rsi": round(current_rsi, 2),

        "macd": round(current_macd, 4),

        "macd_signal": round(current_signal, 4),

        "reasons": reasons
    }