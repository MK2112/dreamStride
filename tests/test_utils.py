import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils import get_parameters, FreezeParameters, Logger, compute_return
import tempfile
import os
from torch.nn import Linear

def test_get_parameters():
    m1 = Linear(2,2)
    m2 = Linear(2,2)
    params = get_parameters([m1, m2])
    assert all(hasattr(p, 'data') for p in params)

def test_freeze_parameters():
    m = Linear(2,2)
    orig = [p.requires_grad for p in m.parameters()]
    # Normal freeze/unfreeze
    with FreezeParameters([m]):
        assert not any(p.requires_grad for p in m.parameters())
    assert [p.requires_grad for p in m.parameters()] == orig
    # Nested freeze
    with FreezeParameters([m]):
        with FreezeParameters([m]):
            assert not any(p.requires_grad for p in m.parameters())
        assert not any(p.requires_grad for p in m.parameters())
    assert [p.requires_grad for p in m.parameters()] == orig
    # Exception handling: ensure restoration
    try:
        with FreezeParameters([m]):
            raise RuntimeError("test error")
    except RuntimeError:
        pass
    assert [p.requires_grad for p in m.parameters()] == orig

def test_logger_and_pickle():
    import threading
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir)
        logger.log_scalar(1.0, 'foo', 0)
        logger.log_scalars({'bar': 2.0}, 1)
        logger.flush()
        # Check pickle file exists
        pickle_path = os.path.join(tmpdir, "scalar_data.pkl")
        assert os.path.exists(pickle_path)
        # Test log dir creation
        new_dir = os.path.join(tmpdir, "subdir")
        os.makedirs(new_dir, exist_ok=True)
        logger2 = Logger(new_dir)
        logger2.log_scalar(2.0, 'baz', 2)
        logger2.flush()
        # Test concurrent writes
        def log_thread():
            for i in range(5):
                logger.log_scalar(i, 'thread', i)
        threads = [threading.Thread(target=log_thread) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.flush()
        # Test corrupted pickle file handling
        with open(pickle_path, 'wb') as f:
            f.write(b'not a pickle')
        # Should not raise, just overwrite
        logger.dump_scalars_to_pickle({'a': 1}, 99)

def test_compute_return():
    # Normal case
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    discounts = torch.tensor([1.0, 1.0, 1.0])
    td_lam = 0.9
    last_value = torch.tensor(0.0)
    returns = compute_return(rewards, values, discounts, td_lam, last_value)
    assert returns.shape == rewards.shape
    # Degenerate: all zeros
    rewards = torch.zeros(3)
    values = torch.zeros(3)
    discounts = torch.zeros(3)
    last_value = torch.tensor(0.0)
    out = compute_return(rewards, values, discounts, td_lam, last_value)
    assert torch.allclose(out, torch.zeros_like(out))
    # Degenerate: empty tensors
    rewards = torch.tensor([])
    values = torch.tensor([])
    discounts = torch.tensor([])
    last_value = torch.tensor(0.0)
    out = compute_return(rewards, values, discounts, td_lam, last_value)
    assert out.shape == rewards.shape
    # Extreme td_lam
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    discounts = torch.tensor([1.0, 1.0, 1.0])
    last_value = torch.tensor(1.0)
    out1 = compute_return(rewards, values, discounts, 0.0, last_value)
    out2 = compute_return(rewards, values, discounts, 1.0, last_value)
    assert out1.shape == out2.shape == rewards.shape
