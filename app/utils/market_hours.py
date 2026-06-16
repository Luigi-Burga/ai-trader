from datetime import datetime
import pandas_market_calendars as mcal
import pytz


def is_market_open():

    ny_tz = pytz.timezone("America/New_York")

    now = datetime.now(ny_tz)

    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=now.date(),
        end_date=now.date()
    )

    #
    # Holiday or weekend
    #
    if schedule.empty:
        return False

    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]

    return market_open <= now <= market_close