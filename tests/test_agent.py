import numpy as np
import torch

from snake_rl.agent import DQNAgent
from snake_rl.networks import QNetwork


def test_agent_selects_greedy_action():
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0)
    state = np.zeros((3, 5, 5), dtype=np.float32)
    action = agent.select_action(state)
    assert 0 <= action < 4
