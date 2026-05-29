from datetime import datetime
import pytz

def market_is_open():

    est = pytz.timezone("US/Eastern")

    now = datetime.now(est)

    market_open = now.replace(
        hour=0,  #  9
        minute=0, # 30
        second=0
    )

    market_close = now.replace(
        hour=23,   # 16
        minute=59,  #00 
        second=59
    )

    return market_open <= now <= market_close