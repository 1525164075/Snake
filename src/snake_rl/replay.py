import random
from collections import deque
from typing import Optional


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

    def __len__(self):
        return len(self.buffer)
