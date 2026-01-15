import os

from snake_rl.trainer import train_dqn


def test_train_dqn_writes_logs(tmp_path):
    cfg = {
        "env": {"grid_size": 5, "max_steps": 10},
        "train": {
            "episodes": 2,
            "batch_size": 1,
            "min_replay_size": 1,
            "replay_capacity": 20,
            "gamma": 0.9,
            "lr": 0.001,
            "epsilon_start": 0.5,
            "epsilon_end": 0.1,
            "epsilon_decay": 1.0,
            "target_update": 5,
            "eval_every": 0,
        },
        "reward": {"food": 1.0, "step": -0.01, "death": -1.0},
        "eval": {"episodes": 2},
    }
    result = train_dqn(cfg, device="cpu", seed=0, output_dir=str(tmp_path))
    run_dir = result["run_dir"]
    assert os.path.exists(os.path.join(run_dir, "train_metrics.csv"))
    for name in ["reward.png", "score.png", "loss.png", "epsilon.png"]:
        assert os.path.exists(os.path.join(run_dir, name))
