import os
import time
from typing import Optional

import torch

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.networks import QNetwork
from snake_rl.replay import ReplayBuffer
from snake_rl.train import train_step


def build_env(cfg, seed: Optional[int] = None):
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


def train_dqn(cfg, device: str = "cpu", seed: Optional[int] = None, output_dir: str = "runs"):
    env = build_env(cfg, seed=seed)
    input_shape = (3, env.grid_size, env.grid_size)
    agent = DQNAgent(
        QNetwork(input_shape=input_shape, num_actions=4),
        num_actions=4,
        epsilon=cfg["train"]["epsilon_start"],
        device=device,
    )
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
