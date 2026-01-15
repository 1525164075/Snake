import argparse
import time

import torch

from snake_rl.agent import DQNAgent
from snake_rl.config import load_config
from snake_rl.env import SnakeEnv
from snake_rl.networks import QNetwork


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
