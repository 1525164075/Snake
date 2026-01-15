import os

from snake_rl.evolve import run_ga


def test_ga_logging_outputs(tmp_path):
    base = {"lr": 0.001, "epsilon_decay": 0.95}
    bounds = {"lr": (1e-5, 1e-2), "epsilon_decay": (0.9, 0.99)}

    def evaluate_fn(params):
        return params["lr"] + params["epsilon_decay"]

    run_ga(base, bounds, evaluate_fn, population_size=4, generations=2, elite_k=2, seed=0, log_dir=str(tmp_path))
    assert os.path.exists(os.path.join(tmp_path, "ga_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "ga_best.png"))
    assert os.path.exists(os.path.join(tmp_path, "ga_avg.png"))
