from datetime import datetime
import pytz

def market_is_open():

    est = pytz.timezone("US/Eastern")

    now = datetime.now(est)

    market_open = now.replace(
        hour=9,  #  9
        minute=30, # 30
        second=0
    )

    market_close = now.replace(
        hour=16,   # 16
        minute=00,  #00 
        second=00
    )


    return market_open <= now <= market_close