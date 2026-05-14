import asyncio
from datetime import datetime

from app.data.market_data import get_stock_data
from app.strategies.signal_engine import generate_signal
from app.strategies.signal_sell import generate_sell
from app.alerts.telegram_alert import send_alert


WATCHLIST = [
    "NVDA",
    "PLTR",
    "PLTU",
    "SMH",
    "VOO"
]

SALELIST = [
    "BITX","48.26","11%"
    "FNGU","33.83","5%"
    "PLTR","129.88","10%"
    "PLTU","33.42","10%"
]

async def monitor_market():

    while True:

        print("\n===================================")
        print(f"Market Scan: {datetime.now()}")
        print("===================================\n")

        for ticker in WATCHLIST:

            try:

                df = get_stock_data(ticker)
                print(f"{ticker} Data:\n{df}\n")

                signal = generate_signal(df)

                print(f"{ticker} => {signal}")
               # await send_alert(f"{ticker} => {signal}")


            except Exception as e:

                print(f"{ticker} ERROR => {e}")

        await asyncio.sleep(60)



if __name__ == "__main__":
    asyncio.run(monitor_market())