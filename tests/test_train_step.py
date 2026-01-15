import numpy as np
import torch

from snake_rl.agent import DQNAgent
from snake_rl.networks import QNetwork
from snake_rl.train import train_step


def test_train_step_returns_loss():
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = {
        "state": [np.zeros((3, 5, 5), dtype=np.float32) for _ in range(4)],
        "action": [0, 1, 2, 3],
        "reward": [1.0, 0.0, 0.0, -1.0],
        "next_state": [np.zeros((3, 5, 5), dtype=np.float32) for _ in range(4)],
        "done": [False, False, True, True],
    }
    loss = train_step(agent, optimizer, batch, gamma=0.99)
    assert isinstance(loss, float)
    assert loss >= 0.0
