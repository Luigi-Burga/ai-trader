import ta

def generate_sell(df):

    if df.empty:
        return "NO_DATA"

    if len(df) < 30:
        return "INSUFFICIENT_DATA"

    close_prices = df["Close"]

    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(
        close=close_prices,
        window=14
    )

    df["rsi"] = rsi_indicator.rsi()

    # MACD
    macd_indicator = ta.trend.MACD(close_prices)

    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()

    # Latest scalar values
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]

    # Signals
    buy_signal = (
        rsi < 30 and
        macd > macd_signal
    )

    sell_signal = (
        rsi > 70 and
        macd < macd_signal
    )

    if buy_signal:
        return "BUY"

    if sell_signal:
        return "SELL"

    return "HOLD"