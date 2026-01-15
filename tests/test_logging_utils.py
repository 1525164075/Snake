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
