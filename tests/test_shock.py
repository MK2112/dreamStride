import os
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Mock gym before importing spot_wrapper
sys.modules["gym"] = MagicMock()

from models import RSSM
from spot_wrapper import SpotControl


class TestSpotControlRobustness(unittest.TestCase):
    @patch("spot_wrapper.socket.socket")
    @patch("spot_wrapper.json.load")
    @patch("builtins.open")
    def test_initialization_timeout(self, mock_open, mock_json_load, mock_socket):
        # Mock config load
        mock_json_load.return_value = {
            "socket": {
                "host": "localhost",
                "port_controller": 9998,
                "port_daydreamer": 9999,
                "recv_size": 8,
                "buffer_size": 1024,
            }
        }

        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance

        mock_conn = MagicMock()
        mock_sock_instance.accept.return_value = (mock_conn, ("127.0.0.1", 12345))

        wrapper = SpotControl()

        # Verify timeout was set
        mock_conn.settimeout.assert_called_with(30.0)

    @patch("spot_wrapper.socket.socket")
    @patch("spot_wrapper.json.load")
    @patch("builtins.open")
    def test_receive_all_incomplete(self, mock_open, mock_json_load, mock_socket):
        mock_json_load.return_value = {
            "socket": {
                "host": "localhost",
                "port_controller": 9998,
                "port_daydreamer": 9999,
                "recv_size": 8,
                "buffer_size": 1024,
            }
        }

        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_conn = MagicMock()
        mock_sock_instance.accept.return_value = (mock_conn, ("127.0.0.1", 12345))

        wrapper = SpotControl()

        # Simulate receiving partial data and then closing
        mock_conn.recv.side_effect = [b"1234", b""]

        with self.assertRaisesRegex(RuntimeError, "Received data incomplete"):
            wrapper.receive_all(mock_conn, 8)


class TestModelRobustness(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.rssm = RSSM(
            action_size=12,
            stoch_size=32,
            deter_size=200,
            hidden_size=200,
            obs_embed_size=1024,
            activation="elu",
        ).to(self.device)

    def test_rssm_numerical_stability(self):
        batch_size = 5
        prev_state = self.rssm.init_state(batch_size, self.device)
        prev_action = torch.randn(batch_size, 12).to(self.device)
        obs_embed = torch.randn(batch_size, 1024).to(self.device)

        # Test with standard inputs
        prior, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)

        self.assertFalse(
            torch.isnan(posterior["mean"]).any(), "RSSM produced NaNs in mean"
        )
        self.assertFalse(
            torch.isnan(posterior["std"]).any(), "RSSM produced NaNs in std"
        )
        self.assertFalse(
            torch.isinf(posterior["mean"]).any(), "RSSM produced Infs in mean"
        )

        # Test with large inputs (potential overflow)
        obs_embed_large = obs_embed * 100.0
        prior, posterior = self.rssm.observe_step(
            prev_state, prev_action, obs_embed_large
        )
        self.assertFalse(
            torch.isnan(posterior["mean"]).any(), "RSSM unstable with large inputs"
        )

    def test_rssm_zero_inputs(self):
        batch_size = 5
        prev_state = self.rssm.init_state(batch_size, self.device)
        prev_action = torch.zeros(batch_size, 12).to(self.device)
        obs_embed = torch.zeros(batch_size, 1024).to(self.device)

        prior, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        self.assertFalse(
            torch.isnan(posterior["std"]).any(), "RSSM NaN with zero inputs"
        )
        self.assertTrue((posterior["std"] > 0).all(), "Std deviation must be positive")


class TestConfigRobustness(unittest.TestCase):
    def test_config_params(self):
        # Load the actual config file
        import json

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        with open(config_path, "r") as f:
            config = json.load(f)

        required_keys = ["env", "socket", "algo"]
        for key in required_keys:
            self.assertIn(key, config)

        self.assertIn("host", config["socket"])


if __name__ == "__main__":
    unittest.main()
