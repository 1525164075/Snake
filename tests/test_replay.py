import numpy as np
from snake_rl.replay import ReplayBuffer


def test_replay_push_and_sample():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(5):
        state = np.zeros((3, 5, 5), dtype=np.float32)
        buf.push(state, i % 4, 1.0, state, False)
    batch = buf.sample(3)
    assert len(batch["state"]) == 3
    assert len(buf) == 5
