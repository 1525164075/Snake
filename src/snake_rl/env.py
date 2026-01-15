import random
from typing import Optional

import numpy as np


class SnakeEnv:
    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 200,
        reward_cfg: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reward_cfg = reward_cfg or {"food": 1.0, "step": -0.01, "death": -1.0}
        self.snake = []
        self.food = None
        self.steps = 0
        self.score = 0
        self._pygame = None
        self._screen = None

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self._spawn_food()
        self.steps = 0
        self.score = 0
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
        self.steps += 1

        ate = False
        done = False
        reward = self.reward_cfg["step"]

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = self.reward_cfg["death"]
            done = True
        elif new_head in self.snake:
            reward = self.reward_cfg["death"]
            done = True
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                ate = True
                self.score += 1
                reward = self.reward_cfg["food"]
                self._spawn_food()
            else:
                self.snake.pop()

        if self.steps >= self.max_steps:
            done = True

        return self._encode_state(), reward, done, {"ate": ate, "score": self.score}

    def render(self, mode: str = "rgb_array", scale: int = 20):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for r, c in self.snake[1:]:
            grid[r, c] = (0, 200, 0)
        head_r, head_c = self.snake[0]
        grid[head_r, head_c] = (0, 80, 255)
        food_r, food_c = self.food
        grid[food_r, food_c] = (255, 60, 60)
        frame = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)

        if mode == "rgb_array":
            return frame
        if mode == "human":
            try:
                import pygame
            except ImportError as exc:
                raise ImportError("pygame is required for human rendering") from exc
            if self._pygame is None:
                self._pygame = pygame
                pygame.init()
                self._screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            surface = self._pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            self._screen.blit(surface, (0, 0))
            self._pygame.display.flip()
            return None
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
