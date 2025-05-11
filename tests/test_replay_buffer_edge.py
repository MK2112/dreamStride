import pytest
import numpy as np
from replay_buffer import ReplayBuffer

def test_replay_buffer_empty_sample():
    # Sampling from empty buffer should raise or handle gracefully
    buffer = ReplayBuffer(5, (3, 4, 4), 2, 3, 2)
    with pytest.raises(Exception):
        buffer.sample()

def test_replay_buffer_malformed_add():
    buffer = ReplayBuffer(5, (3, 4, 4), 2, 3, 2)
    # None observation
    with pytest.raises(Exception):
        buffer.add(None, np.zeros(2), 0.0, 0)
    # Malformed action
    with pytest.raises(Exception):
        buffer.add({'image': np.zeros((3,4,4), dtype=np.uint8)}, None, 0.0, 0)
    # Malformed reward
    with pytest.raises(Exception):
        buffer.add({'image': np.zeros((3,4,4), dtype=np.uint8)}, np.zeros(2), None, 0)
    # Malformed terminal
    with pytest.raises(Exception):
        buffer.add({'image': np.zeros((3,4,4), dtype=np.uint8)}, np.zeros(2), 0.0, None)
