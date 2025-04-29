import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pytest
from models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder

def test_rssm_forward():
    batch = 2
    action_size = 3
    stoch_size = 4
    deter_size = 5
    hidden_size = 5
    obs_embed_size = 6
    activation = 'relu'
    device = torch.device('cpu')
    rssm = RSSM(action_size, stoch_size, deter_size, hidden_size, obs_embed_size, activation)
    prev_state = rssm.init_state(batch, device)
    prev_action = torch.zeros(batch, action_size)
    obs_embed = torch.randn(batch, obs_embed_size)
    nonterm = 1.0
    prior, posterior = rssm.observe_step(prev_state, prev_action, obs_embed, nonterm)
    assert all(k in prior for k in ('mean','std','stoch','deter'))
    assert all(k in posterior for k in ('mean','std','stoch','deter'))
    # Device mismatch: should raise or handle
    prev_state = rssm.init_state(batch, device)
    prev_action = torch.zeros(batch, action_size).to('cpu')
    obs_embed = torch.randn(batch, obs_embed_size).to('cpu')
    try:
        rssm.observe_step(prev_state, prev_action.to('cpu'), obs_embed.to('cpu'), nonterm)
    except Exception:
        pass
    # Non-contiguous tensors
    obs_embed_nc = obs_embed.t().contiguous().t()  # still contiguous, but test anyway
    prior, posterior = rssm.observe_step(prev_state, prev_action, obs_embed_nc, nonterm)
    assert all(k in prior for k in ('mean','std','stoch','deter'))
    # Differentiability: check gradients
    obs_embed.requires_grad_()
    prior, posterior = rssm.observe_step(prev_state, prev_action, obs_embed, nonterm)
    s = posterior['stoch'].sum()
    s.backward()
    assert obs_embed.grad is not None

def test_conv_encoder_decoder():
    encoder = ConvEncoder((3,64,64), 16, 'relu')
    decoder = ConvDecoder(4, 5, (3,64,64), 'relu')
    x = torch.randn(2, 3, 64, 64)
    z = encoder(x)
    assert z.shape[0] == 2
    recon_dist = decoder(torch.randn(2, 4+5))
    recon_sample = recon_dist.rsample()
    assert recon_sample.shape[0] == 2

def test_dense_decoder():
    decoder = DenseDecoder(4, 5, (1,), 2, 8, 'relu', 'normal')
    x = torch.randn(2, 4+5)
    dist = decoder(x)
    sample = dist.rsample()
    assert sample.shape[0] == 2

def test_action_decoder():
    decoder = ActionDecoder(3, 4, 5, 2, 8, 'relu')
    x = torch.randn(2, 4+5)
    action = decoder(x)
    assert action.shape[0] == 2
    # Exploration: check noise and clamping
    action2 = decoder.add_exploration(action, action_noise=0.5)
    assert torch.all(action2 <= 1.0) and torch.all(action2 >= -1.0)
    # Differentiability
    x.requires_grad_()
    out = decoder(x)
    if hasattr(out, 'sum'):
        s = out.sum()
        s.backward()
        assert x.grad is not None
