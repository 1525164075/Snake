from snake_rl.config import load_config


def test_load_config_defaults():
    cfg = load_config("config/default.yaml")
    assert cfg["env"]["grid_size"] == 10
    assert cfg["train"]["batch_size"] > 0
    assert "reward" in cfg
