ALERT_LEVELS = [65, 75, 90, 100]

def evaluate_multi_level_alerts(
    position,
    current_price
):

    buy_price = position["buy_price"]

    target_profit = position["target_profit"]

    alerts_sent = position["alerts_sent"]

    # ====================================
    # TARGET PRICE
    # ====================================

    target_price = buy_price * (
        1 + target_profit / 100
    )

    triggered_alerts = []

    # ====================================
    # EVALUATE EACH LEVEL
    # ====================================

    for level in ALERT_LEVELS:

        # Skip already sent alerts
        if level in alerts_sent:

            continue

        # Calculate alert price
        alert_price = buy_price + (
            (target_price - buy_price)
            * (level / 100)
        )

        # Trigger condition
        if current_price >= alert_price:

            triggered_alerts.append({

                "level": level,

                "alert_price": round(alert_price, 2),

                "target_price": round(target_price, 2)
            })

            # Save sent alert
            alerts_sent.append(level)

    return triggered_alerts