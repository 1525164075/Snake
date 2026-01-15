import numpy as np
import torch
import torch.nn as nn

from snake_rl.agent import DQNAgent
from snake_rl.train import train_step


class ConstantNet(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch = x.shape[0]
        return torch.full((batch, 4), self.value, device=x.device) + self.dummy


def test_train_step_uses_target_network():
    q_net = ConstantNet(0.0)
    target_net = ConstantNet(1.0)
    agent = DQNAgent(q_net, num_actions=4, epsilon=0.0, target_network=target_net, double_dqn=True)
    optimizer = torch.optim.SGD(agent.q_network.parameters(), lr=0.0)
    batch = {
        "state": [np.zeros((3, 5, 5), dtype=np.float32)],
        "action": [0],
        "reward": [0.0],
        "next_state": [np.zeros((3, 5, 5), dtype=np.float32)],
        "done": [False],
    }
    loss = train_step(agent, optimizer, batch, gamma=1.0)
    assert abs(loss - 1.0) < 1e-6
