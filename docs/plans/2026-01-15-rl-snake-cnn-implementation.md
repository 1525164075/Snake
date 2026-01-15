# RL Snake CNN 网络与 2000 回合训练实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `QNetwork` 升级为轻量 CNN，并使用 50x50 配置训练 2000 回合，以提升策略质量。

**Architecture:** 替换 `QNetwork` 为卷积网络，使用假输入动态计算全连接输入维度；新增网络结构测试确保输出维度与 `Conv2d` 结构正确；准备 50x50 的 2000 回合配置用于训练与播放。

**Tech Stack:** Python 3.9+, PyTorch, NumPy.

---

### Task 1: CNN 结构与输出维度测试

**Files:**
- Modify: `tests/test_trainer.py`

**Step 1: Write the failing test**

```python
# tests/test_trainer.py
import torch
from snake_rl.networks import QNetwork


def test_qnetwork_uses_conv_and_outputs_actions():
    net = QNetwork(input_shape=(3, 5, 5), num_actions=4)
    assert any("Conv2d" in type(m).__name__ for m in net.modules())
    out = net(torch.zeros(1, 3, 5, 5))
    assert out.shape == (1, 4)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trainer.py::test_qnetwork_uses_conv_and_outputs_actions -v`  
Expected: FAIL (no Conv2d / shape mismatch)

**Step 3: Commit the test**

```bash
git add tests/test_trainer.py
git commit -m "test: add cnn structure check for qnetwork"
```

---

### Task 2: QNetwork 替换为 CNN

**Files:**
- Modify: `src/snake_rl/networks.py`

**Step 1: Implement minimal CNN**

```python
# src/snake_rl/networks.py
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
```

**Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trainer.py::test_qnetwork_uses_conv_and_outputs_actions -v`  
Expected: PASS

**Step 3: Commit**

```bash
git add src/snake_rl/networks.py
git commit -m "feat: replace qnetwork with cnn"
```

---

### Task 3: 50x50 训练配置（2000 回合）

**Files:**
- Create: `config/grid50_cnn.yaml`

**Step 1: Create config**

```yaml
env:
  grid_size: 50
  max_steps: 200
train:
  episodes: 2000
  batch_size: 64
  min_replay_size: 64
  replay_capacity: 10000
  gamma: 0.99
  lr: 0.0005
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay: 0.995
  target_update: 200
  eval_every: 0
reward:
  food: 1.0
  step: -0.01
  death: -1.0
eval:
  episodes: 5
ga:
  population_size: 4
  generations: 2
  elite_k: 2
  train_episodes: 10
  bounds:
    lr: [0.0001, 0.01]
    epsilon_decay: [0.9, 0.99]
    reward_food: [0.5, 2.0]
    reward_step: [-0.05, -0.001]
    reward_death: [-2.0, -0.5]
```

**Step 2: Commit**

```bash
git add config/grid50_cnn.yaml
git commit -m "feat: add 50x50 cnn training config"
```

---

### Task 4: 训练与播放验证（手动）

**Step 1: Train**

Run:
```bash
PYTHONPATH=src python3 scripts/train.py --config config/grid50_cnn.yaml --mode train --output runs
```
Expected: 输出 `run_dir` 与 `model_path`

**Step 2: Play**

Run:
```bash
PYTHONPATH=src python3 scripts/train.py --config config/grid50_cnn.yaml --mode play --checkpoint runs/train_xxx/model.pt --fps 6 --episodes 5 --scale 10
```
Expected: 弹出窗口并播放，表现优于线性模型

---

**Plan complete and saved to `docs/plans/2026-01-15-rl-snake-cnn-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration  
**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
