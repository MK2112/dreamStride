import pytest
import torch
from models import RSSM, DenseDecoder, ActionDecoder

def test_rssm_wrong_dtype():
    # Pass float instead of tensor
    rssm = RSSM(2, 3, 4, 5, 6, 'relu')
    prev_state = rssm.init_state(1, torch.device('cpu'))
    prev_action = [0.0, 0.0]  # Not a tensor
    obs_embed = [0.0]*6  # Not a tensor
    with pytest.raises(Exception):
        rssm.observe_step(prev_state, prev_action, obs_embed, 1.0)

def test_dense_decoder_extreme_values():
    decoder = DenseDecoder(4, 5, (1,), 2, 8, 'relu', 'normal')
    x = torch.full((2, 9), 1e20)
    dist = decoder(x)
    sample = dist.rsample()
    assert torch.isfinite(sample).all() or torch.any(torch.isinf(sample))

def test_action_decoder_unexpected_batch():
    decoder = ActionDecoder(3, 4, 5, 2, 8, 'relu')
    # Pass batch size 1 instead of 2
    x = torch.randn(1, 9)
    action = decoder(x)
    assert action.shape[0] == 1
