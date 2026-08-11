import random


REGIONS = [
    "cls",
    "center",
    "upper",
    "lower",
    "left",
    "right",
    "q1",
    "q2",
    "q3",
    "q4",
    "random",
]


def region_token_indices(region, grid_size=14, random_seed=0):
    region = str(region).lower()

    if region == "cls":
        return [0]

    mid = grid_size // 2

    if region == "center":
        lo = grid_size // 4
        hi = grid_size - lo
        coords = [(r, c) for r in range(lo, hi) for c in range(lo, hi)]
    elif region == "upper":
        coords = [(r, c) for r in range(mid) for c in range(grid_size)]
    elif region == "lower":
        coords = [(r, c) for r in range(mid, grid_size) for c in range(grid_size)]
    elif region == "left":
        coords = [(r, c) for r in range(grid_size) for c in range(mid)]
    elif region == "right":
        coords = [(r, c) for r in range(grid_size) for c in range(mid, grid_size)]
    elif region == "q1":
        coords = [(r, c) for r in range(mid) for c in range(mid)]
    elif region == "q2":
        coords = [(r, c) for r in range(mid) for c in range(mid, grid_size)]
    elif region == "q3":
        coords = [(r, c) for r in range(mid, grid_size) for c in range(mid)]
    elif region == "q4":
        coords = [
            (r, c)
            for r in range(mid, grid_size)
            for c in range(mid, grid_size)
        ]
    elif region == "random":
        rng = random.Random(random_seed)
        patch_indices = list(range(1, 1 + grid_size * grid_size))
        return sorted(rng.sample(patch_indices, 49))
    else:
        raise ValueError(f"Unknown region: {region}")

    return [1 + r * grid_size + c for r, c in coords]
