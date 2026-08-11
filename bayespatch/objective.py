REGION_COSTS = {
    "cls": 0.05,
    "center": 0.25,
    "q1": 0.25,
    "q2": 0.25,
    "q3": 0.25,
    "q4": 0.25,
    "random": 0.30,
    "upper": 0.50,
    "lower": 0.50,
    "left": 0.50,
    "right": 0.50,
}


def patch_cost(region, alpha):
    return REGION_COSTS[str(region).lower()] * float(alpha)


def sparse_objective(mean_delta, region, alpha, lam=0.03):
    return float(mean_delta) - float(lam) * patch_cost(region, alpha)