import os

import torch

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.networks import QNetwork
from snake_rl.trainer import train_dqn, evaluate_policy


def test_train_dqn_returns_history(tmp_path):
    cfg = {
        "env": {"grid_size": 5, "max_steps": 10},
        "train": {
            "episodes": 2,
            "batch_size": 2,
            "min_replay_size": 2,
            "replay_capacity": 50,
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
    assert len(result["history"]) == 2
    assert os.path.exists(result["model_path"])


def test_evaluate_policy_returns_metrics():
    env = SnakeEnv(grid_size=5, max_steps=5, reward_cfg={"food": 1.0, "step": -0.01, "death": -1.0}, seed=0)
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0)
    metrics = evaluate_policy(env, agent, episodes=2, epsilon=0.0)
    assert metrics["episodes"] == 2
    assert metrics["avg_steps"] > 0


def test_qnetwork_uses_conv_and_outputs_actions():
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    assert any("Conv2d" in type(module).__name__ for module in net.modules())
    out = net(torch.zeros(1, 3, 5, 5))
    assert out.shape == (1, 4)
