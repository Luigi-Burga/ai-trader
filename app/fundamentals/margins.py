def margin_score(info):

    operating_margin = info.get("operatingMargins")

    if operating_margin is None:
        return 0

    margin = operating_margin * 100

    if margin > 25:
        return 15
    elif margin > 15:
        return 10
    elif margin > 5:
        return 5

    return 0