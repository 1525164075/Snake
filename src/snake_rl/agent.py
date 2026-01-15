import copy
import numpy as np
import torch


class DQNAgent:
    def __init__(
        self,
        q_network,
        num_actions: int,
        epsilon: float = 1.0,
        device: str = "cpu",
        target_network=None,
        double_dqn: bool = True,
    ):
        self.q_network = q_network.to(device)
        self.target_network = target_network.to(device) if target_network is not None else copy.deepcopy(self.q_network)
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.device = device
        self.double_dqn = double_dqn

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_network(state_t)[0]
        return int(torch.argmax(q_vals).item())

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
