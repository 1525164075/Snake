from snake_rl.env import SnakeEnv


def test_reset_initial_state():
    env = SnakeEnv(grid_size=10)
    state = env.reset()
    assert env.grid_size == 10
    assert len(env.snake) == 3
    assert env.food not in env.snake
    assert state.shape == (3, 10, 10)
