import yfinance as yf
import pandas as pd

def get_stock_data(ticker):

    df = yf.download(
        tickers=ticker,
        period="60d",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

def get_current_price(df):

    return float(df["Close"].iloc[-1])