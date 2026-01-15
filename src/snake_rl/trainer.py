import os
from typing import Optional

import torch

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.logging_utils import append_csv, init_csv, make_run_dir, save_line_plot
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
    episode_idx = 0

    run_dir = make_run_dir(output_dir, "train")
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    fields = ["episode", "reward", "score", "steps", "loss", "epsilon"]
    init_csv(metrics_path, fields)
    episodes = []
    rewards = []
    scores = []
    losses = []
    epsilons = []

    for _ in range(cfg["train"]["episodes"]):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        loss_sum = 0.0
        loss_count = 0
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
                loss_val = train_step(agent, optimizer, batch, gamma=cfg["train"]["gamma"])
                loss_sum += loss_val
                loss_count += 1
            if total_steps % cfg["train"]["target_update"] == 0:
                agent.update_target()

        avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
        history.append(
            {"reward": ep_reward, "steps": ep_steps, "score": env.score, "loss": avg_loss, "epsilon": epsilon}
        )
        append_csv(
            metrics_path,
            fields,
            {
                "episode": episode_idx,
                "reward": ep_reward,
                "score": env.score,
                "steps": ep_steps,
                "loss": avg_loss,
                "epsilon": epsilon,
            },
        )
        episodes.append(episode_idx)
        rewards.append(ep_reward)
        scores.append(env.score)
        losses.append(avg_loss)
        epsilons.append(epsilon)
        episode_idx += 1
        epsilon = max(cfg["train"]["epsilon_end"], epsilon * cfg["train"]["epsilon_decay"])

    save_line_plot(os.path.join(run_dir, "reward.png"), episodes, rewards, "Reward", "Episode", "Reward")
    save_line_plot(os.path.join(run_dir, "score.png"), episodes, scores, "Score", "Episode", "Score")
    save_line_plot(os.path.join(run_dir, "loss.png"), episodes, losses, "Loss", "Episode", "Loss")
    save_line_plot(os.path.join(run_dir, "epsilon.png"), episodes, epsilons, "Epsilon", "Episode", "Epsilon")

    model_path = os.path.join(run_dir, "model.pt")
    torch.save(agent.q_network.state_dict(), model_path)
    return {"history": history, "model_path": model_path, "run_dir": run_dir}
