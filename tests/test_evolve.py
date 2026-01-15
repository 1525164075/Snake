from snake_rl.evolve import mutate_config


def test_mutate_config_respects_bounds():
    base = {"lr": 0.001, "epsilon_decay": 0.99, "reward_food": 1.0}
    bounds = {
        "lr": (1e-5, 1e-2),
        "epsilon_decay": (0.9, 0.999),
        "reward_food": (0.5, 2.0),
    }
    mutated = mutate_config(base, bounds, seed=0)
    assert bounds["lr"][0] <= mutated["lr"] <= bounds["lr"][1]
    assert bounds["epsilon_decay"][0] <= mutated["epsilon_decay"] <= bounds["epsilon_decay"][1]
    assert bounds["reward_food"][0] <= mutated["reward_food"] <= bounds["reward_food"][1]
