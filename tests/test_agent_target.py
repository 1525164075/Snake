import torch

from snake_rl.agent import DQNAgent
from snake_rl.networks import QNetwork


def test_update_target_copies_weights():
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0)
    for p in agent.q_network.parameters():
        p.data.fill_(1.0)
    agent.update_target()
    for p in agent.target_network.parameters():
        assert torch.allclose(p.data, torch.ones_like(p.data))
