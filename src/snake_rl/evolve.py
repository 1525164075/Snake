import random
from typing import Optional


def mutate_config(base: dict, bounds: dict, seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    mutated = {}
    for key, value in base.items():
        low, high = bounds[key]
        noise = rng.uniform(-0.1, 0.1) * value
        new_val = value + noise
        new_val = max(low, min(high, new_val))
        mutated[key] = new_val
    return mutated
