"""
Portfolio Monitor V2.0
----------------------

Performance-focused version of Portfolio Monitor.

Design goal:
    Replace the direct yfinance intraday request with Market Data V2,
    while preserving the existing investment and alert logic.

Preserved behavior:
    - calculate_position_status()
    - current_price from the last Close
    - highest_today from intraday High.max()
    - evaluate_multi_level_alerts()
    - evaluate_target_alert()
    - evaluate_trailing_stop()
    - position["highest_price"] update
    - Telegram messages and send_telegram() contract
    - no changes to investment/alert thresholds or decisions

Data acquisition:
    Market Data V2 handles:
    - persistent intraday cache
    - intraday TTL
    - Yahoo fallback on cache miss
    - ticker normalization

The monitor remains synchronous and keeps the same public function:
    monitor_position(position)
"""

import pandas as pd

from app.data.market_data import get_history

from app.portfolio.target_alert import (
    evaluate_target_alert
)

from app.portfolio.multi_level_alerts import (
    evaluate_multi_level_alerts
)

from app.portfolio.trailing_stop import (
    evaluate_trailing_stop
)

from app.alerts.telegram_alert import (
    send_telegram
)


VERSION = "2.0"


def calculate_position_status(
    current_price,
    buy_price,
    target_profit
):
    """
    Preserve the original portfolio status calculation exactly.
    """

    profit_percent = (
        (current_price - buy_price)
        / buy_price
    ) * 100

    target_price = buy_price * (
        1 + target_profit / 100
    )

    return {
        "profit_percent": round(
            profit_percent,
            2
        ),

        "target_price": round(
            target_price,
            2
        ),

        "target_hit":
            current_price >= target_price
    }


def _normalize_intraday_ohlcv(df):
    """Normalize Market Data V2/yfinance output to 1-D OHLCV columns.

    yfinance may return a MultiIndex even for a single ticker. In that case
    df["Close"] can itself be a DataFrame, which makes iloc[-1] a Series and
    breaks float(). Keep the monitor contract identical by reducing the frame
    to canonical OHLCV columns first.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy(deep=True)

    if isinstance(x.columns, pd.MultiIndex):
        selected = {}
        for col in x.columns:
            parts = [str(part).strip() for part in col]
            for part in parts:
                key = part.lower()
                if key in {"open", "high", "low", "close", "volume"}:
                    canonical = key.capitalize()
                    if canonical not in selected:
                        selected[canonical] = col
                    break

        if selected:
            x = x[[selected[name] for name in selected]]
            x.columns = list(selected.keys())

    rename = {}
    for col in x.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.capitalize()
    x = x.rename(columns=rename)
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    required = ("Open", "High", "Low", "Close", "Volume")
    if any(col not in x.columns for col in required):
        return pd.DataFrame()

    x = x[list(required)].copy()
    for col in required:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.dropna(subset=["High", "Low", "Close"])
    return x


def calculate_entry_risk_sizing(
    position,
    account_equity=None,
    max_risk_pct=1.0,
):
    """Return risk-based sizing when an entry stop is available.

    Existing portfolio positions remain untouched when account_equity or
    stop_price is not configured.
    """
    equity = (
        account_equity
        if account_equity is not None
        else position.get("account_equity")
    )
    entry = position.get("buy_price", position.get("entry_price"))
    stop = position.get("stop_price", position.get("stop"))

    if equity is None or entry is None or stop is None:
        return {
            "status": "NOT_CONFIGURED",
            "reason": "account_equity_and_stop_required",
        }

    return calculate_position_size(
        account_equity=equity,
        entry_price=entry,
        stop_price=stop,
        max_risk_pct=max_risk_pct,
        max_position_pct=position.get("max_position_pct"),
        max_shares=position.get("max_shares"),
    )


def monitor_position(position):

    symbol = position["ticker"]

    buy_price = float(
        position["buy_price"]
    )

    target_profit = float(
        position["target_profit"]
    )

    # ------------------------------------------------------------------
    # MARKET DATA V2
    #
    # The original monitor used:
    #     direct Yahoo history request
    #         period="1d",
    #         interval="1m"
    #     )
    #
    # Only the data-acquisition layer is changed here.
    # The resulting dataframe is consumed exactly as before.
    # ------------------------------------------------------------------
    df = get_history(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=False,
    )

    df = _normalize_intraday_ohlcv(df)

    if df.empty:

        print(
            f"{symbol} -> "
            f"No market data"
        )

        return

    current_price = float(
        df["Close"].iloc[-1]
    )

    #
    # IMPORTANT:
    # Use intraday HIGH
    #
    highest_today = float(
        df["High"].max()
    )

    status = calculate_position_status(
        current_price,
        buy_price,
        target_profit
    )

    risk_sizing = calculate_entry_risk_sizing(position)
    if risk_sizing.get("status") == "OK":
        print(
            f"{symbol} | RiskSizing | "
            f"shares={risk_sizing['shares']} | "
            f"position=${risk_sizing['position_value']:.2f} | "
            f"risk=${risk_sizing['actual_risk']:.2f} | "
            f"risk%={risk_sizing['actual_risk_pct']:.3f}%"
        )

    print(
        f"{symbol} | "
        f"Close={current_price:.2f} | "
        f"High={highest_today:.2f} | "
        f"Profit={status['profit_percent']}%"
    )

    #
    # MULTI LEVEL ALERTS
    #
    alerts = evaluate_multi_level_alerts(
        position,
        highest_today
    )

    for alert in alerts:

        message = (
            f"🚨 {symbol}\n"
            f"Profit Alert: {alert['level']}%\n"
            f"Price: {highest_today:.2f}"
        )

        print(message)

        send_telegram(message)

    #
    # TARGET ALERT
    #
    target = evaluate_target_alert(
        position,
        highest_today
    )

    if target["triggered"]:

        message = (
            f"🎯 TARGET REACHED\n\n"
            f"Ticker: {symbol}\n"
            f"Price Actual: {current_price:.2f}\n"
            f"Price Alto: {highest_today:.2f}\n"
            f"Target: {target['target_price']:.2f}"
        )

        print(message)

        send_telegram(message)

    #
    # TRAILING STOP
    #
    trailing = evaluate_trailing_stop(
        position,
        highest_today
    )

    #
    # Save highest price reached
    #
    position["highest_price"] = max(
        position["highest_price"],
        highest_today
    )

    if trailing["status"] == "SELL":

        message = (
            f"🔴 TRAILING STOP SELL\n\n"
            f"Ticker: {symbol}\n"
            f"Price: {highest_today:.2f}\n"
            f"Stop: {trailing['trailing_price']:.2f}"
        )

        print(message)

        send_telegram(message)
