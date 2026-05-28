from datetime import datetime
import pytz

def market_is_open():

    est = pytz.timezone("US/Eastern")

    now = datetime.now(est)

    market_open = now.replace(
        hour=9,  #  9
        minute=15, # 30
        second=0
    )

    market_close = now.replace(
        hour=15,   # 16
        minute=50,  #00 
        second=0
    )

    return market_open <= now <= market_close