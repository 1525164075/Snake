import os
import random
from typing import Optional

from snake_rl.logging_utils import append_csv, init_csv, save_line_plot


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
    log_dir: Optional[str] = None,
) -> dict:
    rng = random.Random(seed)
    population = [mutate_config(base, bounds, seed=rng.randint(0, 1_000_000)) for _ in range(population_size)]
    history = []
    best_params = None
    best_fitness = float("-inf")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        init_csv(os.path.join(log_dir, "ga_metrics.csv"), ["generation", "best_fitness", "avg_fitness"])

    for gen in range(generations):
        scored = []
        for params in population:
            fitness = evaluate_fn(params)
            scored.append((fitness, params))
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = dict(params)
        scored.sort(key=lambda x: x[0], reverse=True)
        avg_fitness = sum(f for f, _ in scored) / len(scored)
        history.append({"generation": gen, "best_fitness": scored[0][0], "avg_fitness": avg_fitness})
        if log_dir:
            append_csv(
                os.path.join(log_dir, "ga_metrics.csv"),
                ["generation", "best_fitness", "avg_fitness"],
                {"generation": gen, "best_fitness": scored[0][0], "avg_fitness": avg_fitness},
            )

        elites = [p for _, p in scored[:elite_k]]
        next_pop = elites[:]
        while len(next_pop) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover(parent_a, parent_b, seed=rng.randint(0, 1_000_000))
            child = mutate_config(child, bounds, seed=rng.randint(0, 1_000_000))
            next_pop.append(child)
        population = next_pop

    if log_dir:
        gens = [h["generation"] for h in history]
        bests = [h["best_fitness"] for h in history]
        avgs = [h["avg_fitness"] for h in history]
        save_line_plot(os.path.join(log_dir, "ga_best.png"), gens, bests, "GA Best Fitness", "Generation", "Fitness")
        save_line_plot(os.path.join(log_dir, "ga_avg.png"), gens, avgs, "GA Avg Fitness", "Generation", "Fitness")

    return {"best_params": best_params, "history": history}
