import numpy as np
import torch


class DQNAgent:
    def __init__(self, q_network, num_actions: int, epsilon: float = 1.0, device: str = "cpu"):
        self.q_network = q_network.to(device)
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_network(state_t)[0]
        return int(torch.argmax(q_vals).item())
