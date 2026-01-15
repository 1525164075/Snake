import os

from snake_rl.logging_utils import log_eval_metrics


def test_eval_logging_outputs(tmp_path):
    metrics = {"episodes": 2, "avg_reward": 1.0, "avg_steps": 5.0, "avg_score": 1.0}
    log_eval_metrics(str(tmp_path), metrics)
    assert os.path.exists(os.path.join(tmp_path, "eval_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "eval_reward.png"))
    assert os.path.exists(os.path.join(tmp_path, "eval_score.png"))
