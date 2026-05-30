def debt_score(info):

    debt_to_equity = info.get("debtToEquity")

    if debt_to_equity is None:
        return 0

    if debt_to_equity < 30:
        return 15
    elif debt_to_equity < 100:
        return 10
    elif debt_to_equity < 200:
        return 5

    return 0