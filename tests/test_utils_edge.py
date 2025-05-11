import torch
from utils import FreezeParameters, compute_return


def test_freeze_parameters_empty():
    # Should not raise error
    with FreezeParameters([]):
        pass

def test_compute_return_negative_discount():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    discounts = torch.tensor([-1.0, -1.0, -1.0])
    last_value = torch.tensor(1.0)
    td_lam = 0.9
    # Should not crash, but output may be nonsensical
    out = compute_return(rewards, values, discounts, td_lam, last_value)
    assert out.shape == rewards.shape

def test_compute_return_td_lam_out_of_bounds():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    discounts = torch.tensor([1.0, 1.0, 1.0])
    last_value = torch.tensor(1.0)
    # td_lam < 0
    out1 = compute_return(rewards, values, discounts, -0.1, last_value)
    # td_lam > 1
    out2 = compute_return(rewards, values, discounts, 1.1, last_value)
    assert out1.shape == out2.shape == rewards.shape
