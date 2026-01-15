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
