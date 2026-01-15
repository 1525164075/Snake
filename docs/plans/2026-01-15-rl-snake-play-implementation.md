# RL Snake 实时播放实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增 `play` 模式，加载 checkpoint 并实时窗口播放贪吃蛇策略。

**Architecture:** 在 CLI 中增加 `play` 参数分支；新增 `snake_rl.play` 模块负责准备模型与播放循环；`scripts/train.py` 调用播放入口。播放循环支持 `fps`、`episodes`、`scale` 和 `max_steps` 覆盖。

**Tech Stack:** Python 3.9+, PyTorch, NumPy, (可选) pygame.

---

### Task 1: CLI 参数支持 `play`

**Files:**
- Modify: `src/snake_rl/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
def test_play_mode_args():
    parser = build_train_parser()
    args = parser.parse_args(["--mode", "play", "--checkpoint", "model.pt"])
    assert args.mode == "play"
    assert args.checkpoint == "model.pt"
    assert args.fps == 10
    assert args.episodes == 3
    assert args.scale == 20
    assert args.max_steps is None
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py::test_play_mode_args -v`  
Expected: FAIL (unknown args or missing defaults)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/cli.py
parser.add_argument("--mode", choices=["train", "eval", "ga", "play"], default="train")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--scale", type=int, default=20)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py::test_play_mode_args -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/cli.py tests/test_cli.py
git commit -m "feat: add play mode cli args"
```

---

### Task 2: 播放准备与循环逻辑

**Files:**
- Create: `src/snake_rl/play.py`
- Test: `tests/test_play.py`

**Step 1: Write the failing test**

```python
# tests/test_play.py
from pathlib import Path

import pytest
import torch

from snake_rl.networks import QNetwork
from snake_rl.play import prepare_play, play_episodes


def _cfg():
    return {
        "env": {"grid_size": 5, "max_steps": 20},
        "reward": {"food": 1.0, "step": -0.01, "death": -1.0},
    }


def test_prepare_play_loads_checkpoint(tmp_path: Path):
    cfg = _cfg()
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    ckpt = tmp_path / "model.pt"
    torch.save(net.state_dict(), ckpt)

    env, agent = prepare_play(cfg, checkpoint=str(ckpt), device="cpu", seed=0, max_steps=123)
    assert env.max_steps == 123
    assert agent.epsilon == 0.0


def test_prepare_play_missing_checkpoint(tmp_path: Path):
    cfg = _cfg()
    with pytest.raises(FileNotFoundError):
        prepare_play(cfg, checkpoint=str(tmp_path / "missing.pt"), device="cpu")


def test_play_episodes_rgb_array_runs(tmp_path: Path):
    cfg = _cfg()
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    ckpt = tmp_path / "model.pt"
    torch.save(net.state_dict(), ckpt)
    env, agent = prepare_play(cfg, checkpoint=str(ckpt), device="cpu", seed=0)
    ok = play_episodes(env, agent, episodes=1, fps=0, render_mode="rgb_array", scale=5)
    assert ok is True
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_play.py -v`  
Expected: FAIL (module/function not found)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/play.py
from pathlib import Path
import time

import torch

from snake_rl.agent import DQNAgent
from snake_rl.networks import QNetwork
from snake_rl.trainer import build_env


def prepare_play(cfg, checkpoint: str, device: str = "cpu", seed=None, max_steps=None):
    if not checkpoint:
        raise ValueError("checkpoint is required for play mode")
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    env = build_env(cfg, seed=seed)
    if max_steps is not None:
        env.max_steps = max_steps
    net = QNetwork(input_shape=(3, env.grid_size, env.grid_size), num_actions=4)
    agent = DQNAgent(net, num_actions=4, epsilon=0.0, device=device)
    agent.q_network.load_state_dict(torch.load(path, map_location=device))
    agent.update_target()
    return env, agent


def _pump_events():
    import pygame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def play_episodes(env, agent, episodes=1, fps=10, render_mode="human", scale=20) -> bool:
    delay = 0.0 if fps <= 0 else 1.0 / fps
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if render_mode == "human":
                try:
                    if not _pump_events():
                        env.close()
                        return False
                except ImportError as exc:
                    raise ImportError("pygame is required for play mode") from exc
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)
            env.render(mode=render_mode, scale=scale)
            if delay > 0:
                time.sleep(delay)
    env.close()
    return True
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_play.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/play.py tests/test_play.py
git commit -m "feat: add play helpers for realtime demo"
```

---

### Task 3: CLI 播放入口

**Files:**
- Modify: `scripts/train.py`

**Step 1: Write the failing test**

Skip automated test（播放依赖窗口与 pygame）；使用下方手动验证。

**Step 2: Implement play branch**

```python
# scripts/train.py
from snake_rl.play import prepare_play, play_episodes
...
    if args.mode == "play":
        try:
            env, agent = prepare_play(
                cfg,
                checkpoint=args.checkpoint,
                device=args.device,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(exc)
            return
        play_episodes(env, agent, episodes=args.episodes, fps=args.fps, render_mode="human", scale=args.scale)
        return
```

**Step 3: Manual verification**

Run: `PYTHONPATH=src python3 scripts/train.py --mode play --checkpoint runs/train_xxx/model.pt --fps 10 --episodes 3`
Expected: 弹出窗口并实时播放，关闭窗口后退出。

**Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add play mode to cli"
```

---

**Plan complete and saved to `docs/plans/2026-01-15-rl-snake-play-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration  
**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
