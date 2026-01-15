# Training, Rendering, Evaluation, and GA Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full DQN training, evaluation, rendering, and a GA main loop to evolve hyperparameters and reward weights.

**Architecture:** Extend the Snake environment with scoring and rendering, add Double DQN support in the agent, implement trainer utilities (train/eval), and implement a GA loop that evaluates configurations via a pluggable evaluator. CLI scripts wire these together.

**Tech Stack:** Python 3.9+, PyTorch, pygame, numpy, pyyaml, pytest.

---

### Task 1: Env scoring, max steps, and RGB rendering

**Files:**
- Modify: `src/snake_rl/env.py`
- Test: `tests/test_env_scoring.py`
- Test: `tests/test_env_render.py`

**Step 1: Write the failing tests**

```python
# tests/test_env_scoring.py
from snake_rl.env import SnakeEnv


def test_score_increments_on_food():
    env = SnakeEnv(grid_size=5, max_steps=10, seed=0)
    env.reset()
    head_r, head_c = env.snake[0]
    env.food = (head_r, head_c + 1)
    _, _, done, info = env.step(1)
    assert done is False
    assert env.score == 1
    assert info["ate"] is True


def test_max_steps_terminates():
    env = SnakeEnv(grid_size=5, max_steps=1, seed=0)
    env.reset()
    _, _, done, _ = env.step(1)
    assert done is True
```

```python
# tests/test_env_render.py
import numpy as np
from snake_rl.env import SnakeEnv


def test_render_rgb_array_shape():
    env = SnakeEnv(grid_size=5, max_steps=10, seed=0)
    env.reset()
    frame = env.render(mode="rgb_array", scale=4)
    assert frame.shape == (20, 20, 3)
    assert frame.dtype == np.uint8
```

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/test_env_scoring.py::test_score_increments_on_food -v`
- `pytest tests/test_env_render.py::test_render_rgb_array_shape -v`

Expected: FAIL (missing attributes or render method)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/env.py (additions/changes)
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
```

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_env_scoring.py::test_score_increments_on_food -v`
- `pytest tests/test_env_render.py::test_render_rgb_array_shape -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/env.py tests/test_env_scoring.py tests/test_env_render.py
git commit -m "feat: add env scoring, max steps, and render"
```

---

### Task 2: Target network and Double DQN in train step

**Files:**
- Modify: `src/snake_rl/agent.py`
- Modify: `src/snake_rl/train.py`
- Test: `tests/test_agent_target.py`
- Test: `tests/test_train_step_double.py`

**Step 1: Write the failing tests**

```python
# tests/test_agent_target.py
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
```

```python
# tests/test_train_step_double.py
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
        return torch.full((batch, 4), self.value, device=x.device)


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
```

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/test_agent_target.py::test_update_target_copies_weights -v`
- `pytest tests/test_train_step_double.py::test_train_step_uses_target_network -v`

Expected: FAIL (missing target network / double DQN logic)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/agent.py (additions/changes)
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

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

```python
# src/snake_rl/train.py (modify train_step)
import numpy as np
import torch
import torch.nn.functional as F


def train_step(agent, optimizer, batch, gamma: float):
    states = torch.tensor(np.array(batch["state"]), dtype=torch.float32, device=agent.device)
    actions = torch.tensor(batch["action"], dtype=torch.int64, device=agent.device).unsqueeze(1)
    rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=agent.device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch["next_state"]), dtype=torch.float32, device=agent.device)
    dones = torch.tensor(batch["done"], dtype=torch.float32, device=agent.device).unsqueeze(1)

    q_values = agent.q_network(states).gather(1, actions)
    with torch.no_grad():
        if getattr(agent, "double_dqn", False) and getattr(agent, "target_network", None) is not None:
            next_actions = agent.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = agent.target_network(next_states).gather(1, next_actions)
        elif getattr(agent, "target_network", None) is not None:
            next_q = agent.target_network(next_states).max(1, keepdim=True)[0]
        else:
            next_q = agent.q_network(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * (1 - dones) * next_q

    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_agent_target.py::test_update_target_copies_weights -v`
- `pytest tests/test_train_step_double.py::test_train_step_uses_target_network -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/agent.py src/snake_rl/train.py tests/test_agent_target.py tests/test_train_step_double.py
git commit -m "feat: add target network and double dqn"
```

---

### Task 3: Training and evaluation loops + config updates

**Files:**
- Create: `src/snake_rl/trainer.py`
- Modify: `config/default.yaml`
- Test: `tests/test_trainer.py`

**Step 1: Write the failing test**

```python
# tests/test_trainer.py
import os
from snake_rl.trainer import train_dqn, evaluate_policy
from snake_rl.env import SnakeEnv
from snake_rl.agent import DQNAgent
from snake_rl.networks import QNetwork


def test_train_dqn_returns_history(tmp_path):
    cfg = {
        "env": {"grid_size": 5, "max_steps": 10},
        "train": {
            "episodes": 2,
            "batch_size": 2,
            "min_replay_size": 2,
            "replay_capacity": 50,
            "gamma": 0.9,
            "lr": 0.001,
            "epsilon_start": 0.5,
            "epsilon_end": 0.1,
            "epsilon_decay": 1.0,
            "target_update": 5,
            "eval_every": 0,
        },
        "reward": {"food": 1.0, "step": -0.01, "death": -1.0},
        "eval": {"episodes": 2},
    }
    result = train_dqn(cfg, device="cpu", seed=0, output_dir=str(tmp_path))
    assert len(result["history"]) == 2
    assert os.path.exists(result["model_path"])


def test_evaluate_policy_returns_metrics():
    env = SnakeEnv(grid_size=5, max_steps=5, reward_cfg={"food": 1.0, "step": -0.01, "death": -1.0}, seed=0)
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0)
    metrics = evaluate_policy(env, agent, episodes=2, epsilon=0.0)
    assert metrics["episodes"] == 2
    assert metrics["avg_steps"] > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py::test_train_dqn_returns_history -v`  
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/trainer.py
import os
import time
import torch

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.networks import QNetwork
from snake_rl.replay import ReplayBuffer
from snake_rl.train import train_step


def build_env(cfg, seed=None):
    return SnakeEnv(
        grid_size=cfg["env"]["grid_size"],
        max_steps=cfg["env"]["max_steps"],
        reward_cfg=cfg["reward"],
        seed=seed,
    )


def evaluate_policy(env, agent, episodes: int, epsilon: float = 0.0):
    total_reward = 0.0
    total_steps = 0
    total_score = 0
    prev_eps = agent.epsilon
    agent.epsilon = epsilon
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
        total_reward += ep_reward
        total_steps += ep_steps
        total_score += env.score
    agent.epsilon = prev_eps
    return {
        "episodes": episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
        "avg_score": total_score / episodes,
    }


def train_dqn(cfg, device: str = "cpu", seed: int | None = None, output_dir: str = "runs"):
    env = build_env(cfg, seed=seed)
    input_shape = (3, env.grid_size, env.grid_size)
    agent = DQNAgent(QNetwork(input_shape=input_shape, num_actions=4), num_actions=4, epsilon=cfg["train"]["epsilon_start"], device=device)
    optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=cfg["train"]["lr"])
    replay = ReplayBuffer(capacity=cfg["train"]["replay_capacity"], seed=seed)

    history = []
    epsilon = cfg["train"]["epsilon_start"]
    total_steps = 0

    for _ in range(cfg["train"]["episodes"]):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        agent.epsilon = epsilon
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if len(replay) >= cfg["train"]["min_replay_size"]:
                batch = replay.sample(cfg["train"]["batch_size"])
                train_step(agent, optimizer, batch, gamma=cfg["train"]["gamma"])
            if total_steps % cfg["train"]["target_update"] == 0:
                agent.update_target()

        history.append({"reward": ep_reward, "steps": ep_steps, "score": env.score})
        epsilon = max(cfg["train"]["epsilon_end"], epsilon * cfg["train"]["epsilon_decay"])

    os.makedirs(output_dir, exist_ok=True)
    run_name = str(int(time.time()))
    model_path = os.path.join(output_dir, f"{run_name}_model.pt")
    torch.save(agent.q_network.state_dict(), model_path)
    return {"history": history, "model_path": model_path}
```

```yaml
# config/default.yaml (extend)
env:
  grid_size: 10
  max_steps: 200
train:
  episodes: 200
  batch_size: 64
  min_replay_size: 64
  replay_capacity: 10000
  gamma: 0.99
  lr: 0.0005
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  target_update: 100
  eval_every: 0
reward:
  food: 1.0
  step: -0.01
  death: -1.0
eval:
  episodes: 50
ga:
  population_size: 6
  generations: 4
  elite_k: 2
  train_episodes: 20
  bounds:
    lr: [0.0001, 0.01]
    epsilon_decay: [0.95, 0.999]
    reward_food: [0.5, 2.0]
    reward_step: [-0.05, -0.001]
    reward_death: [-2.0, -0.5]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py::test_train_dqn_returns_history -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/trainer.py config/default.yaml tests/test_trainer.py
git commit -m "feat: add training and evaluation loops"
```

---

### Task 4: GA main loop with evaluator callback

**Files:**
- Modify: `src/snake_rl/evolve.py`
- Test: `tests/test_ga_loop.py`

**Step 1: Write the failing test**

```python
# tests/test_ga_loop.py
from snake_rl.evolve import run_ga


def test_run_ga_returns_history_and_best():
    base = {"lr": 0.001, "epsilon_decay": 0.95}
    bounds = {"lr": (1e-5, 1e-2), "epsilon_decay": (0.9, 0.99)}
    calls = {"n": 0}

    def evaluate_fn(params):
        calls["n"] += 1
        return params["lr"] + params["epsilon_decay"]

    result = run_ga(base, bounds, evaluate_fn, population_size=4, generations=3, elite_k=2, seed=0)
    assert len(result["history"]) == 3
    assert calls["n"] == 12
    assert "best_params" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ga_loop.py::test_run_ga_returns_history_and_best -v`  
Expected: FAIL (missing run_ga)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/evolve.py (additions)
import random
from typing import Optional


def crossover(parent_a: dict, parent_b: dict, seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    child = {}
    for key in parent_a:
        child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
    return child


def run_ga(base: dict, bounds: dict, evaluate_fn, population_size: int, generations: int, elite_k: int, seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    population = [mutate_config(base, bounds, seed=rng.randint(0, 1_000_000)) for _ in range(population_size)]
    history = []
    best_params = None
    best_fitness = float("-inf")

    for gen in range(generations):
        scored = []
        for params in population:
            fitness = evaluate_fn(params)
            scored.append((fitness, params))
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = dict(params)
        scored.sort(key=lambda x: x[0], reverse=True)
        history.append({"generation": gen, "best_fitness": scored[0][0]})

        elites = [p for _, p in scored[:elite_k]]
        next_pop = elites[:]
        while len(next_pop) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover(parent_a, parent_b, seed=rng.randint(0, 1_000_000))
            child = mutate_config(child, bounds, seed=rng.randint(0, 1_000_000))
            next_pop.append(child)
        population = next_pop

    return {"best_params": best_params, "history": history}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ga_loop.py::test_run_ga_returns_history_and_best -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/evolve.py tests/test_ga_loop.py
git commit -m "feat: add GA main loop"
```

---

### Task 5: CLI wiring for train/eval/ga and demo render

**Files:**
- Modify: `src/snake_rl/cli.py`
- Modify: `scripts/train.py`
- Modify: `scripts/demo.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py (extend)
from snake_rl.cli import build_train_parser


def test_train_parser_defaults():
    parser = build_train_parser()
    args = parser.parse_args([])
    assert args.config == "config/default.yaml"
    assert args.mode == "train"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_train_parser_defaults -v`  
Expected: FAIL (missing mode arg)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/cli.py (update)
import argparse


def build_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "ga"], default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--checkpoint", default="")
    return parser
```

```python
# scripts/train.py (update)
from snake_rl.cli import build_train_parser
from snake_rl.config import load_config
from snake_rl.trainer import train_dqn, evaluate_policy, build_env
from snake_rl.networks import QNetwork
from snake_rl.agent import DQNAgent
from snake_rl.evolve import run_ga


def main():
    parser = build_train_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train_dqn(cfg, device=args.device, seed=args.seed, output_dir=args.output)
        return

    if args.mode == "eval":
        env = build_env(cfg, seed=args.seed)
        net = QNetwork(input_shape=(3, env.grid_size, env.grid_size), num_actions=4)
        agent = DQNAgent(net, num_actions=4, epsilon=0.0, device=args.device)
        if args.checkpoint:
            import torch

            agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
            agent.update_target()
        metrics = evaluate_policy(env, agent, episodes=cfg["eval"]["episodes"], epsilon=0.0)
        print(metrics)
        return

    if args.mode == "ga":
        base = {
            "lr": cfg["train"]["lr"],
            "epsilon_decay": cfg["train"]["epsilon_decay"],
            "reward_food": cfg["reward"]["food"],
            "reward_step": cfg["reward"]["step"],
            "reward_death": cfg["reward"]["death"],
        }
        bounds = {k: tuple(v) for k, v in cfg["ga"]["bounds"].items()}

        def evaluate_fn(params):
            local_cfg = dict(cfg)
            local_cfg["train"] = dict(cfg["train"])
            local_cfg["reward"] = dict(cfg["reward"])
            local_cfg["train"]["lr"] = params["lr"]
            local_cfg["train"]["epsilon_decay"] = params["epsilon_decay"]
            local_cfg["train"]["episodes"] = cfg["ga"]["train_episodes"]
            local_cfg["reward"]["food"] = params["reward_food"]
            local_cfg["reward"]["step"] = params["reward_step"]
            local_cfg["reward"]["death"] = params["reward_death"]
            result = train_dqn(local_cfg, device=args.device, seed=args.seed, output_dir=args.output)
            scores = [h["score"] for h in result["history"]]
            return sum(scores) / len(scores)

        result = run_ga(
            base,
            bounds,
            evaluate_fn,
            population_size=cfg["ga"]["population_size"],
            generations=cfg["ga"]["generations"],
            elite_k=cfg["ga"]["elite_k"],
            seed=args.seed,
        )
        print(result)


if __name__ == "__main__":
    main()
```

```python
# scripts/demo.py (update)
import argparse
import time
import torch

from snake_rl.config import load_config
from snake_rl.env import SnakeEnv
from snake_rl.networks import QNetwork
from snake_rl.agent import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = SnakeEnv(
        grid_size=cfg["env"]["grid_size"],
        max_steps=cfg["env"]["max_steps"],
        reward_cfg=cfg["reward"],
        seed=0,
    )
    net = QNetwork(input_shape=(3, env.grid_size, env.grid_size), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0, device=args.device)
    agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    agent.update_target()

    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        env.render(mode="human", scale=args.scale)
        time.sleep(0.03)
    env.close()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_train_parser_defaults -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/cli.py scripts/train.py scripts/demo.py tests/test_cli.py
git commit -m "feat: wire training, eval, ga, and demo CLI"
```

---

**Plan complete and saved to `docs/plans/2026-01-15-rl-snake-training-ga.md`.**
