def revenue_score(info):

    revenue_growth = info.get("revenueGrowth")

    if revenue_growth is None:
        return 0

    growth = revenue_growth * 100

    if growth > 30:
        return 25
    elif growth > 20:
        return 20
    elif growth > 10:
        return 15
    elif growth > 0:
        return 10

    return 0