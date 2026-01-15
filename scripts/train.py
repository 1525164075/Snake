from snake_rl.cli import build_train_parser
from snake_rl.config import load_config
from snake_rl.trainer import train_dqn, evaluate_policy, build_env
from snake_rl.networks import QNetwork
from snake_rl.agent import DQNAgent
from snake_rl.evolve import run_ga
from snake_rl.logging_utils import log_eval_metrics, make_run_dir


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
        run_dir = make_run_dir(args.output, "eval")
        log_eval_metrics(run_dir, metrics)
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
