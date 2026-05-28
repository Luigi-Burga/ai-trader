def predict_price_direction(data):

    rsi = data["rsi"]

    macd = data["macd"]

    macd_signal = data["macd_signal"]

    current_price = data["current_price"]

    sma_20 = data["sma_20"]

    sma_50 = data["sma_50"]

    score = 0

    reasons = []

    # ====================================
    # RSI ANALYSIS
    # ====================================

    if rsi < 30:

        score += 2

        reasons.append(
            "RSI oversold"
        )

    elif rsi > 70:

        score -= 2

        reasons.append(
            "RSI overbought"
        )

    # ====================================
    # MACD ANALYSIS
    # ====================================

    if macd > macd_signal:

        score += 2

        reasons.append(
            "MACD bullish crossover"
        )

    else:

        score -= 2

        reasons.append(
            "MACD bearish crossover"
        )

    # ====================================
    # TREND ANALYSIS
    # ====================================

    if current_price > sma_20:

        score += 1

        reasons.append(
            "Above SMA20"
        )

    if current_price > sma_50:

        score += 1

        reasons.append(
            "Above SMA50"
        )

    # ====================================
    # AI DECISION
    # ====================================

    if score >= 4:

        prediction = "STRONG_BULLISH"

    elif score >= 2:

        prediction = "BULLISH"

    elif score <= -4:

        prediction = "STRONG_BEARISH"

    elif score <= -2:

        prediction = "BEARISH"

    else:

        prediction = "NEUTRAL"

    return {

        "prediction": prediction,

        "score": score,

        "reasons": reasons
    }