# RL Snake Logging and Plotting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为训练/评估/GA 过程增加 CSV 日志与 PNG 曲线输出。

**Architecture:** 新增日志工具模块（CSV + Matplotlib Agg），训练过程记录每回合指标并绘图，GA 每代记录最佳/平均适应度并绘图，CLI 负责创建运行目录并输出结果路径。

**Tech Stack:** Python 3.9+, Matplotlib (Agg), NumPy, CSV, PyTorch.

---

### Task 1: 日志工具（CSV + 折线图）

**Files:**
- Create: `src/snake_rl/logging_utils.py`
- Test: `tests/test_logging_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_utils.py
from pathlib import Path

from snake_rl.logging_utils import init_csv, append_csv, save_line_plot


def test_csv_and_plot_created(tmp_path: Path):
    csv_path = tmp_path / "metrics.csv"
    init_csv(str(csv_path), ["episode", "reward"])
    append_csv(str(csv_path), ["episode", "reward"], {"episode": 1, "reward": 0.5})
    content = csv_path.read_text(encoding="utf-8")
    assert "episode,reward" in content
    assert "1,0.5" in content

    plot_path = tmp_path / "reward.png"
    save_line_plot(str(plot_path), [1, 2], [0.1, 0.2], "Reward", "Episode", "Reward")
    assert plot_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging_utils.py::test_csv_and_plot_created -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl.logging_utils'`

**Step 3: Write minimal implementation**

```python
# src/snake_rl/logging_utils.py
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_run_dir(output_dir: str, prefix: str) -> str:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{prefix}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_csv(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv(path: str, fieldnames: list[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def save_line_plot(path: str, x, y, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging_utils.py::test_csv_and_plot_created -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/logging_utils.py tests/test_logging_utils.py
git commit -m "feat: add logging utilities for csv and plots"
```

---

### Task 2: 训练日志与曲线输出

**Files:**
- Modify: `src/snake_rl/trainer.py`
- Test: `tests/test_train_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_train_logging.py
import os

from snake_rl.trainer import train_dqn


def test_train_dqn_writes_logs(tmp_path):
    cfg = {
        "env": {"grid_size": 5, "max_steps": 10},
        "train": {
            "episodes": 2,
            "batch_size": 1,
            "min_replay_size": 1,
            "replay_capacity": 20,
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
    run_dir = result["run_dir"]
    assert os.path.exists(os.path.join(run_dir, "train_metrics.csv"))
    for name in ["reward.png", "score.png", "loss.png", "epsilon.png"]:
        assert os.path.exists(os.path.join(run_dir, name))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_logging.py::test_train_dqn_writes_logs -v`  
Expected: FAIL (missing run_dir/log files)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/trainer.py (additions/changes)
from snake_rl.logging_utils import init_csv, append_csv, save_line_plot, make_run_dir


def train_dqn(...):
    ...
    run_dir = make_run_dir(output_dir, "train")
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    fields = ["episode", "reward", "score", "steps", "loss", "epsilon"]
    init_csv(metrics_path, fields)

    episode_idx = 0
    for ...:
        ...
        loss_sum = 0.0
        loss_count = 0
        while not done:
            ...
            if len(replay) >= cfg["train"]["min_replay_size"]:
                batch = replay.sample(cfg["train"]["batch_size"])
                loss_val = train_step(...)
                loss_sum += loss_val
                loss_count += 1
        avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
        append_csv(metrics_path, fields, {
            "episode": episode_idx,
            "reward": ep_reward,
            "score": env.score,
            "steps": ep_steps,
            "loss": avg_loss,
            "epsilon": epsilon,
        })
        episode_idx += 1

    save_line_plot(os.path.join(run_dir, "reward.png"), ...)
    save_line_plot(os.path.join(run_dir, "score.png"), ...)
    save_line_plot(os.path.join(run_dir, "loss.png"), ...)
    save_line_plot(os.path.join(run_dir, "epsilon.png"), ...)
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(...)
    return {"history": history, "model_path": model_path, "run_dir": run_dir}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_logging.py::test_train_dqn_writes_logs -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/trainer.py tests/test_train_logging.py
git commit -m "feat: add training csv logs and plots"
```

---

### Task 3: 评估日志与曲线输出

**Files:**
- Modify: `src/snake_rl/logging_utils.py`
- Modify: `scripts/train.py`
- Test: `tests/test_eval_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_eval_logging.py
import os

from snake_rl.logging_utils import log_eval_metrics


def test_eval_logging_outputs(tmp_path):
    metrics = {"episodes": 2, "avg_reward": 1.0, "avg_steps": 5.0, "avg_score": 1.0}
    log_eval_metrics(str(tmp_path), metrics)
    assert os.path.exists(os.path.join(tmp_path, "eval_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "eval_reward.png"))
    assert os.path.exists(os.path.join(tmp_path, "eval_score.png"))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_logging.py::test_eval_logging_outputs -v`  
Expected: FAIL (missing function)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/logging_utils.py (additions)
def log_eval_metrics(run_dir: str, metrics: dict) -> None:
    fields = ["episodes", "avg_reward", "avg_steps", "avg_score"]
    path = os.path.join(run_dir, "eval_metrics.csv")
    init_csv(path, fields)
    append_csv(path, fields, metrics)
    save_line_plot(os.path.join(run_dir, "eval_reward.png"), [1], [metrics["avg_reward"]], "Eval Reward", "Eval", "Reward")
    save_line_plot(os.path.join(run_dir, "eval_score.png"), [1], [metrics["avg_score"]], "Eval Score", "Eval", "Score")
```

```python
# scripts/train.py (eval branch)
from snake_rl.logging_utils import make_run_dir, log_eval_metrics
...
run_dir = make_run_dir(args.output, "eval")
log_eval_metrics(run_dir, metrics)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_logging.py::test_eval_logging_outputs -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/logging_utils.py scripts/train.py tests/test_eval_logging.py
git commit -m "feat: add eval csv logs and plots"
```

---

### Task 4: GA 日志与曲线输出

**Files:**
- Modify: `src/snake_rl/evolve.py`
- Modify: `scripts/train.py`
- Test: `tests/test_ga_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_ga_logging.py
import os

from snake_rl.evolve import run_ga


def test_ga_logging_outputs(tmp_path):
    base = {"lr": 0.001, "epsilon_decay": 0.95}
    bounds = {"lr": (1e-5, 1e-2), "epsilon_decay": (0.9, 0.99)}

    def evaluate_fn(params):
        return params["lr"] + params["epsilon_decay"]

    run_ga(base, bounds, evaluate_fn, population_size=4, generations=2, elite_k=2, seed=0, log_dir=str(tmp_path))
    assert os.path.exists(os.path.join(tmp_path, "ga_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "ga_best.png"))
    assert os.path.exists(os.path.join(tmp_path, "ga_avg.png"))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ga_logging.py::test_ga_logging_outputs -v`  
Expected: FAIL (missing logging)

**Step 3: Write minimal implementation**

```python
# src/snake_rl/evolve.py (additions)
from snake_rl.logging_utils import init_csv, append_csv, save_line_plot


def run_ga(..., log_dir: Optional[str] = None) -> dict:
    ...
    for gen in range(generations):
        ...
        avg_fitness = sum(f for f, _ in scored) / len(scored)
        history.append({"generation": gen, "best_fitness": scored[0][0], "avg_fitness": avg_fitness})
        if log_dir:
            if gen == 0:
                init_csv(os.path.join(log_dir, "ga_metrics.csv"), ["generation", "best_fitness", "avg_fitness"])
            append_csv(..., {"generation": gen, "best_fitness": scored[0][0], "avg_fitness": avg_fitness})
    if log_dir:
        gens = [h["generation"] for h in history]
        bests = [h["best_fitness"] for h in history]
        avgs = [h["avg_fitness"] for h in history]
        save_line_plot(os.path.join(log_dir, "ga_best.png"), gens, bests, "GA Best Fitness", "Generation", "Fitness")
        save_line_plot(os.path.join(log_dir, "ga_avg.png"), gens, avgs, "GA Avg Fitness", "Generation", "Fitness")
```

```python
# scripts/train.py (ga branch)
from snake_rl.logging_utils import make_run_dir
...
run_dir = make_run_dir(args.output, "ga")
result = run_ga(..., log_dir=run_dir)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ga_logging.py::test_ga_logging_outputs -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/snake_rl/evolve.py scripts/train.py tests/test_ga_logging.py
git commit -m "feat: add ga csv logs and plots"
```

---

### Task 5: 训练输出路径回显

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/test_train_logging.py` (reuse)

**Step 1: Write the failing test**

Reuse `tests/test_train_logging.py` from Task 2.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_logging.py::test_train_dqn_writes_logs -v`  
Expected: FAIL if train script does not expose run directory information

**Step 3: Write minimal implementation**

```python
# scripts/train.py (train branch)
result = train_dqn(...)
print({"run_dir": result["run_dir"], "model_path": result["model_path"]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_logging.py::test_train_dqn_writes_logs -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "chore: print train run output paths"
```

---

**Plan complete and saved to `docs/plans/2026-01-15-rl-snake-logging-plots.md`.**
