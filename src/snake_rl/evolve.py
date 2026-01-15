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


def crossover(parent_a: dict, parent_b: dict, seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    child = {}
    for key in parent_a:
        child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
    return child


def run_ga(
    base: dict,
    bounds: dict,
    evaluate_fn,
    population_size: int,
    generations: int,
    elite_k: int,
    seed: Optional[int] = None,
) -> dict:
    rng = random.Random(seed)
    population = [mutate_config(base, bounds, seed=rng.randint(0, 1_000_000)) for _ in range(population_size)]
    history = []
    best_params = None
    best_fitness = float("-inf")

    for gen in range(generations):
        scored = []
        for params in population:
            fitness = evaluate_fn(params)
            scored.append((fitness, params))
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = dict(params)
        scored.sort(key=lambda x: x[0], reverse=True)
        history.append({"generation": gen, "best_fitness": scored[0][0]})

        elites = [p for _, p in scored[:elite_k]]
        next_pop = elites[:]
        while len(next_pop) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover(parent_a, parent_b, seed=rng.randint(0, 1_000_000))
            child = mutate_config(child, bounds, seed=rng.randint(0, 1_000_000))
            next_pop.append(child)
        population = next_pop

    return {"best_params": best_params, "history": history}
