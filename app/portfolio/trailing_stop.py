def evaluate_trailing_stop(position, current_price):

    buy_price = position["buy_price"]

    target_profit = position["target_profit"]

    trailing_stop = position["trailing_stop"]

    highest_price = position["highest_price"]

    # ====================================
    # TARGET PRICE
    # ====================================

    target_price = buy_price * (
        1 + target_profit / 100
    )

    # ====================================
    # TARGET NOT REACHED
    # ====================================

    if current_price < target_price:

        return {

            "status": "HOLD",

            "highest_price": highest_price,

            "trailing_price": None
        }

    # ====================================
    # UPDATE HIGHEST PRICE
    # ====================================

    if current_price > highest_price:

        highest_price = current_price

    # ====================================
    # CALCULATE TRAILING STOP
    # ====================================

    trailing_price = highest_price * (
        1 - trailing_stop / 100
    )

    # ====================================
    # SELL SIGNAL
    # ====================================

    if current_price <= trailing_price:

        return {

            "status": "SELL",

            "highest_price": highest_price,

            "trailing_price": trailing_price
        }

    # ====================================
    # CONTINUE HOLDING
    # ====================================

    return {

        "status": "HOLD",

        "highest_price": highest_price,

        "trailing_price": trailing_price
    }