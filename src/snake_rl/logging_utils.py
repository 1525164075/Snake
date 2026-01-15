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


def log_eval_metrics(run_dir: str, metrics: dict) -> None:
    fields = ["episodes", "avg_reward", "avg_steps", "avg_score"]
    path = os.path.join(run_dir, "eval_metrics.csv")
    init_csv(path, fields)
    append_csv(path, fields, metrics)
    save_line_plot(
        os.path.join(run_dir, "eval_reward.png"),
        [1],
        [metrics["avg_reward"]],
        "Eval Reward",
        "Eval",
        "Reward",
    )
    save_line_plot(
        os.path.join(run_dir, "eval_score.png"),
        [1],
        [metrics["avg_score"]],
        "Eval Score",
        "Eval",
        "Score",
    )
