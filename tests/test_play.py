from pathlib import Path

import pytest
import torch

from snake_rl.networks import QNetwork
import snake_rl.play as play
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


def test_play_episodes_human_renders_before_pump(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.render_called = 0

        def reset(self):
            return "state"

        def step(self, action):
            return "state", 0.0, True, {}

        def render(self, mode="human", scale=20):
            self.render_called += 1

        def close(self):
            return None

    class DummyAgent:
        def select_action(self, state):
            return 0

    env = DummyEnv()
    agent = DummyAgent()

    def fake_pump():
        assert env.render_called > 0
        return True

    monkeypatch.setattr(play, "_pump_events", fake_pump)
    ok = play_episodes(env, agent, episodes=1, fps=0, render_mode="human", scale=1)
    assert ok is True
