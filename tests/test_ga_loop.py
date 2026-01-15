from snake_rl.evolve import run_ga


def test_run_ga_returns_history_and_best():
    base = {"lr": 0.001, "epsilon_decay": 0.95}
    bounds = {"lr": (1e-5, 1e-2), "epsilon_decay": (0.9, 0.99)}
    calls = {"n": 0}

    def evaluate_fn(params):
        calls["n"] += 1
        return params["lr"] + params["epsilon_decay"]

    result = run_ga(base, bounds, evaluate_fn, population_size=4, generations=3, elite_k=2, seed=0)
    assert len(result["history"]) == 3
    assert calls["n"] == 12
    assert "best_params" in result
