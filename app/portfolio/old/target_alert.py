def evaluate_target_alert(position, current_price):

    buy_price = position["buy_price"]

    target_profit = position["target_profit"]

    alert_percent = position.get(
    "target_alert_percent",
    75
)

    # ====================================
    # TARGET PRICE
    # ====================================

    target_price = buy_price * (
        1 + target_profit / 100
    )

    # ====================================
    # ALERT PRICE
    # ====================================

    alert_price = buy_price + (
        (target_price - buy_price)
        * (alert_percent / 100)
    )

    # ====================================
    # ALERT CONDITION
    # ====================================

    triggered = current_price >= alert_price

    return {

        "triggered": triggered,

        "alert_price": round(alert_price, 2),

        "target_price": round(target_price, 2)
    }