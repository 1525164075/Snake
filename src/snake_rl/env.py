import random
from typing import Optional

import numpy as np


class SnakeEnv:
    def __init__(self, grid_size: int = 10, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        self.snake = []
        self.food = None

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self._spawn_food()
        return self._encode_state()

    def _spawn_food(self):
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        self.food = self.rng.choice(empty)

    def _encode_state(self):
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        head_r, head_c = self.snake[0]
        state[0, head_r, head_c] = 1.0
        for r, c in self.snake[1:]:
            state[1, r, c] = 1.0
        food_r, food_c = self.food
        state[2, food_r, food_c] = 1.0
        return state

    def step(self, action: int):
        drdc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = drdc[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._encode_state(), -1.0, True, {"ate": False}

        if new_head in self.snake:
            return self._encode_state(), -1.0, True, {"ate": False}

        self.snake.insert(0, new_head)
        ate = new_head == self.food
        if ate:
            reward = 1.0
            self._spawn_food()
        else:
            self.snake.pop()
            reward = -0.01

        return self._encode_state(), reward, False, {"ate": ate}
