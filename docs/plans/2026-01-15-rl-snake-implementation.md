# RL Snake Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-evolving Snake RL system (DQN baseline + GA evolution), with headless training and renderable demo mode.

**Architecture:** Python package under `src/snake_rl/` with modules for environment, agent, training, evolution, and utilities; CLI scripts in `scripts/` using YAML config from `config/`.

**Tech Stack:** Python 3.10+, PyTorch (MPS optional), pygame, numpy, pyyaml, matplotlib, pytest.

---

### Task 1: Project scaffold + config loader

**Files:**
- Create: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `config/default.yaml`
- Create: `src/snake_rl/__init__.py`
- Create: `src/snake_rl/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from snake_rl.config import load_config

def test_load_config_defaults():
    cfg = load_config("config/default.yaml")
    assert cfg["env"]["grid_size"] == 10
    assert cfg["train"]["batch_size"] > 0
    assert "reward" in cfg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_config_defaults -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl'`

**Step 3: Write minimal implementation**

```python
# tests/conftest.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

```python
# src/snake_rl/config.py
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

```yaml
# config/default.yaml
env:
  grid_size: 10
  max_steps: 200
train:
  batch_size: 64
  gamma: 0.99
  lr: 0.0005
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
reward:
  food: 1.0
  step: -0.01
  death: -1.0
```

```python
# src/snake_rl/__init__.py
__all__ = ["config"]
```

```txt
# requirements.txt
numpy
pygame
pyyaml
torch
matplotlib
pytest
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_load_config_defaults -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt tests/conftest.py tests/test_config.py config/default.yaml src/snake_rl/__init__.py src/snake_rl/config.py
git commit -m "feat: add config loader and defaults"
```

---

### Task 2: Snake environment reset + state encoding

**Files:**
- Create: `src/snake_rl/env.py`
- Test: `tests/test_env_reset.py`

**Step 1: Write the failing test**

```python
# tests/test_env_reset.py
from snake_rl.env import SnakeEnv

def test_reset_initial_state():
    env = SnakeEnv(grid_size=10)
    state = env.reset()
    assert env.grid_size == 10
    assert len(env.snake) == 3
    assert env.food not in env.snake
    assert state.shape == (3, 10, 10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_reset.py::test_reset_initial_state -v`  
Expected: FAIL with `ModuleNotFoundError` or `AttributeError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/env.py
import random
import numpy as np

class SnakeEnv:
    def __init__(self, grid_size: int = 10, seed: int | None = None):
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
        empty = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in self.snake]
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_reset.py::test_reset_initial_state -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/env.py tests/test_env_reset.py
git commit -m "feat: add snake env reset and state encoding"
```

---

### Task 3: Snake environment step + rewards + terminal

**Files:**
- Modify: `src/snake_rl/env.py`
- Test: `tests/test_env_step.py`

**Step 1: Write the failing test**

```python
# tests/test_env_step.py
from snake_rl.env import SnakeEnv

def test_step_moves_and_rewards_food():
    env = SnakeEnv(grid_size=5, seed=0)
    env.reset()
    # place food directly in front of head
    head_r, head_c = env.snake[0]
    env.food = (head_r, head_c + 1)
    state, reward, done, info = env.step(1)  # right
    assert reward > 0
    assert done is False
    assert len(env.snake) == 4
    assert info["ate"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_step.py::test_step_moves_and_rewards_food -v`  
Expected: FAIL with `AttributeError: 'SnakeEnv' object has no attribute 'step'`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/env.py (additions)
    def step(self, action: int):
        # 0=up, 1=right, 2=down, 3=left
        drdc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = drdc[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        reward = 0.0
        done = False
        ate = False

        # wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._encode_state(), -1.0, True, {"ate": False}

        # body collision
        if new_head in self.snake:
            return self._encode_state(), -1.0, True, {"ate": False}

        self.snake.insert(0, new_head)
        if new_head == self.food:
            ate = True
            reward = 1.0
            self._spawn_food()
        else:
            self.snake.pop()
            reward = -0.01

        return self._encode_state(), reward, done, {"ate": ate}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_step.py::test_step_moves_and_rewards_food -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/env.py tests/test_env_step.py
git commit -m "feat: add env step, rewards, and termination"
```

---

### Task 4: Replay buffer

**Files:**
- Create: `src/snake_rl/replay.py`
- Test: `tests/test_replay.py`

**Step 1: Write the failing test**

```python
# tests/test_replay.py
import numpy as np
from snake_rl.replay import ReplayBuffer

def test_replay_push_and_sample():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        state = np.zeros((3, 5, 5), dtype=np.float32)
        buf.push(state, i % 4, 1.0, state, False)
    batch = buf.sample(3)
    assert len(batch["state"]) == 3
    assert len(buf) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay.py::test_replay_push_and_sample -v`  
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/replay.py
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: int | None = None):
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay.py::test_replay_push_and_sample -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/replay.py tests/test_replay.py
git commit -m "feat: add replay buffer"
```

---

### Task 5: Q-network + epsilon-greedy action

**Files:**
- Create: `src/snake_rl/networks.py`
- Create: `src/snake_rl/agent.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agent.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py::test_agent_selects_greedy_action -v`  
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/networks.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.model(x)
```

```python
# src/snake_rl/agent.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py::test_agent_selects_greedy_action -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/networks.py src/snake_rl/agent.py tests/test_agent.py
git commit -m "feat: add q-network and epsilon-greedy agent"
```

---

### Task 6: Single training step update

**Files:**
- Create: `src/snake_rl/train.py`
- Modify: `src/snake_rl/agent.py`
- Test: `tests/test_train_step.py`

**Step 1: Write the failing test**

```python
# tests/test_train_step.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_step.py::test_train_step_returns_loss -v`  
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/train.py
import torch
import torch.nn.functional as F

def train_step(agent, optimizer, batch, gamma: float):
    states = torch.tensor(batch["state"], dtype=torch.float32, device=agent.device)
    actions = torch.tensor(batch["action"], dtype=torch.int64, device=agent.device).unsqueeze(1)
    rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=agent.device).unsqueeze(1)
    next_states = torch.tensor(batch["next_state"], dtype=torch.float32, device=agent.device)
    dones = torch.tensor(batch["done"], dtype=torch.float32, device=agent.device).unsqueeze(1)

    q_values = agent.q_network(states).gather(1, actions)
    with torch.no_grad():
        next_q = agent.q_network(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * (1 - dones) * next_q

    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_step.py::test_train_step_returns_loss -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/train.py src/snake_rl/agent.py tests/test_train_step.py
git commit -m "feat: add single train step update"
```

---

### Task 7: GA evolve hyperparameters

**Files:**
- Create: `src/snake_rl/evolve.py`
- Test: `tests/test_evolve.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve.py
from snake_rl.evolve import mutate_config

def test_mutate_config_respects_bounds():
    base = {"lr": 0.001, "epsilon_decay": 0.99, "reward_food": 1.0}
    bounds = {"lr": (1e-5, 1e-2), "epsilon_decay": (0.9, 0.999), "reward_food": (0.5, 2.0)}
    mutated = mutate_config(base, bounds, seed=0)
    assert bounds["lr"][0] <= mutated["lr"] <= bounds["lr"][1]
    assert bounds["epsilon_decay"][0] <= mutated["epsilon_decay"] <= bounds["epsilon_decay"][1]
    assert bounds["reward_food"][0] <= mutated["reward_food"] <= bounds["reward_food"][1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve.py::test_mutate_config_respects_bounds -v`  
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/evolve.py
import random

def mutate_config(base: dict, bounds: dict, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    mutated = {}
    for key, value in base.items():
        low, high = bounds[key]
        noise = rng.uniform(-0.1, 0.1) * value
        new_val = value + noise
        new_val = max(low, min(high, new_val))
        mutated[key] = new_val
    return mutated
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolve.py::test_mutate_config_respects_bounds -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/evolve.py tests/test_evolve.py
git commit -m "feat: add GA mutation helper"
```

---

### Task 8: Training + demo CLI (skeleton)

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/demo.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from snake_rl.cli import build_train_parser

def test_train_parser_defaults():
    parser = build_train_parser()
    args = parser.parse_args([])
    assert args.config == "config/default.yaml"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_train_parser_defaults -v`  
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/cli.py
import argparse

def build_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser
```

```python
# scripts/train.py
from snake_rl.cli import build_train_parser

def main():
    parser = build_train_parser()
    parser.parse_args()

if __name__ == "__main__":
    main()
```

```python
# scripts/demo.py
def main():
    pass

if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_train_parser_defaults -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/cli.py scripts/train.py scripts/demo.py tests/test_cli.py
git commit -m "feat: add CLI parser and script skeletons"
```

---

**Plan complete and saved to `docs/plans/2026-01-15-rl-snake-implementation.md`.**
