def calculate_position_status(current_price, buy_price, target_profit):

    profit_percent = (
        (current_price - buy_price) / buy_price
    ) * 100

    target_price = buy_price * (
        1 + target_profit / 100
    )

    target_hit = current_price >= target_price

    return {
        "current_price": round(current_price, 2),
        "buy_price": round(buy_price, 2),
        "profit_percent": round(profit_percent, 2),
        "target_price": round(target_price, 2),
        "target_hit": target_hit
    }