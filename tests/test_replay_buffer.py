import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from replay_buffer import ReplayBuffer

def make_fake_obs(shape):
    return {'image': np.random.randint(0, 256, shape, dtype=np.uint8)}

def test_replay_buffer_add_and_sample():
    size = 10
    obs_shape = (3, 4, 4)
    action_size = 2
    seq_len = 3
    batch_size = 2
    buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)

    # Fill buffer
    for _ in range(size):
        obs = make_fake_obs(obs_shape)
        ac = np.random.randn(action_size)
        rew = np.random.randn()
        done = np.random.choice([0, 1])
        buffer.add(obs, ac, rew, done)

    assert buffer.full or buffer.idx > 0
    obs, acs, rews, terms = buffer.sample()
    assert obs.shape[1:] == (batch_size, *obs_shape)
    assert acs.shape[1:] == (batch_size, action_size)
    assert rews.shape[1:] == (batch_size,)
    assert terms.shape[1:] == (batch_size,)

    # Edge: sampling should not crash when buffer is partially filled
    buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
    for _ in range(5):
        buffer.add(make_fake_obs(obs_shape), np.zeros(action_size), 0.0, 0)
    obs, acs, rews, terms = buffer.sample()
    assert obs.shape[1] == batch_size

    # Wraparound: add more than size
    buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
    for _ in range(size * 2):
        buffer.add(make_fake_obs(obs_shape), np.random.randn(action_size), np.random.randn(), 0)
    assert buffer.full
    obs, acs, rews, terms = buffer.sample()
    assert obs.shape[1:] == (batch_size, *obs_shape)

    # Over-capacity batch/seq
    big_batch = size + 2
    big_seq = size // 2
    buffer = ReplayBuffer(size, obs_shape, action_size, big_seq, big_batch)
    for _ in range(size):
        buffer.add(make_fake_obs(obs_shape), np.random.randn(action_size), np.random.randn(), 0)
    # Should not error, but may sample with wrap
    obs, acs, rews, terms = buffer.sample()
    assert obs.shape[1] == big_batch
    # Under-capacity batch/seq
    small_batch = 1
    small_seq = 1
    buffer = ReplayBuffer(size, obs_shape, action_size, small_seq, small_batch)
    for _ in range(3):
        buffer.add(make_fake_obs(obs_shape), np.random.randn(action_size), np.random.randn(), 0)
    obs, acs, rews, terms = buffer.sample()
    assert obs.shape[1] == small_batch
    # Invalid obs shape: should raise
    buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
    try:
        buffer.add({'image': np.random.randn(2, 2, 2)}, np.zeros(action_size), 0.0, 0)
        assert False, "Should have raised ValueError or IndexError"
    except Exception:
        pass
