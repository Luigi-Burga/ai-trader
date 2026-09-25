def evaluate_entry_price(
    current_price,
    buy_target
):

    # ====================================
    # DISTANCE TO TARGET
    # ====================================

    difference_percent = (
        (current_price - buy_target)
        / buy_target
    ) * 100

    # ====================================
    # SIGNAL
    # ====================================

    if current_price <= buy_target:

        signal = "BUY_NOW"

    elif difference_percent <= 3:

        signal = "NEAR_BUY_ZONE"

    else:

        signal = "WAIT"

    return {

        "signal": signal,

        "difference_percent": round(
            difference_percent,
            2
        )
    }