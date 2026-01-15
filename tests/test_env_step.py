from snake_rl.env import SnakeEnv


def test_step_moves_and_rewards_food():
    env = SnakeEnv(grid_size=5, seed=0)
    env.reset()
    head_r, head_c = env.snake[0]
    env.food = (head_r, head_c + 1)
    state, reward, done, info = env.step(1)
    assert reward > 0
    assert done is False
    assert len(env.snake) == 4
    assert info["ate"] is True
