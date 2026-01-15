import numpy as np

from snake_rl.env import SnakeEnv


def test_render_rgb_array_shape():
    env = SnakeEnv(grid_size=5, max_steps=10, seed=0)
    env.reset()
    frame = env.render(mode="rgb_array", scale=4)
    assert frame.shape == (20, 20, 3)
    assert frame.dtype == np.uint8
