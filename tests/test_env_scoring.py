from snake_rl.env import SnakeEnv


def test_score_increments_on_food():
    env = SnakeEnv(grid_size=5, max_steps=10, seed=0)
    env.reset()
    head_r, head_c = env.snake[0]
    env.food = (head_r, head_c + 1)
    _, _, done, info = env.step(1)
    assert done is False
    assert env.score == 1
    assert info["ate"] is True


def test_max_steps_terminates():
    env = SnakeEnv(grid_size=5, max_steps=1, seed=0)
    env.reset()
    _, _, done, _ = env.step(1)
    assert done is True
