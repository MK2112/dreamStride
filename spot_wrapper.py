import os
import gym
import json
import socket
import numpy as np


class SpotControl:
    def __init__(self, size=(64, 256)):
        self.size = size

        with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"),
            "r",
        ) as f:
            config = json.load(f)

        self.HOST = config["socket"]["host"]
        self.PORT_OUT = config["socket"]["port_controller"]
        self.PORT_IN = config["socket"]["port_daydreamer"]
        self.RECV_SIZE = config["socket"]["recv_size"]
        self.BUFFER_SIZE = config["socket"]["buffer_size"]
        self.world_iters = 256  # Training event limit
        self.iters_counter = 0  # Training event counter
        print("[+] Integrator started.")
        print(f"[!] One episode is set to {self.world_iters} iterations.")

        # Connect Socket to (HOST, PORT_DAYDREAMER) to listen to receiving data -> socket_in
        self._socket_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_in.bind((self.HOST, self.PORT_IN))
        # Connect Socket to (HOST, PORT_CONTROLLER) to send data -> socket_out
        self._socket_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_out.connect((self.HOST, self.PORT_OUT))
        print("[+] Reverse connection established")
        self._socket_in.listen()
        self.conn, self.addr = self._socket_in.accept()
        self.conn.settimeout(30.0)  # 30 seconds timeout
        print(f"[+] Connected to controller at {self.addr}")

    def receive_all(self, conn: socket.socket, length: int) -> bytes:
        """
        As it turns out socket.send_all() is a thing, but doesn't in fact send all with really sizeable data.
        I don't really know why that is, but this function is a 'workaround' to ensure all data is received.
        :param conn: We want to receive from here
        :param length: The length of the data we want to receive (this was announced before)
        :return: Concatenated, received data
        """
        received_data = b""
        while len(received_data) < length:
            remaining_length = length - len(received_data)
            chunk = conn.recv(min(self.BUFFER_SIZE, remaining_length))
            if not chunk:
                raise RuntimeError(
                    "Received data incomplete. Socket connection broken."
                )
            received_data += chunk
        return received_data

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self.size + (3,), dtype=np.uint8),
                "reset": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)

    def step(self, action):
        ## Actio: Send network output to WeBots
        # Action is {1: 0.1, 2: 0.2, ..., 12: 0.12, 'reset': 0}
        # If action does not contain 'reset' key, add it with value 0

        # Turn numpy array to list
        action = action.tolist()
        action = {str(i): action[i] for i in range(len(action))}

        action_data = json.dumps(action).encode()
        self._socket_out.sendall(
            len(action_data).to_bytes(self.RECV_SIZE, byteorder="big")
        )
        self._socket_out.sendall(action_data)

        ## Reactio: Receive observation from WeBots
        data_length = int.from_bytes(self.conn.recv(self.RECV_SIZE), byteorder="big")
        obs_data = self.receive_all(self.conn, data_length)
        obs = json.loads(obs_data.decode())

        # obs: Dict ['position': (3,), 'velocity': (2,), 'image': (3, 64, 64)], reward:0.9556776085533385, done:False, info:{'discount': array(1., dtype=float32)}
        # image: (3, 64, 64) [0, 255]

        done = obs["done"] or (self.iters_counter >= self.world_iters)
        self.iters_counter = (
            self.iters_counter + 1 if self.iters_counter < self.world_iters else 0
        )

        return (
            {
                "position": [0, 0, 0],
                "velocity": [0, 0],
                "image": np.array(obs["image"], dtype=np.uint8),
            },
            obs["reward"],
            done,
            {"discount": np.array(1.0, np.float32)},
        )

    def reset(self):
        ## Actio: Send network output to WeBots
        # Action is {1: 0.1, 2: 0.2, ..., 12: 0.12, 'reset': 0}

        # New episode resets to full self.world_iters-step budget
        self.iters_counter = 0

        np_actions = np.random.uniform(-1, 1, 12)
        action = {str(i): np_actions[i] for i in range(12)}
        action["reset"] = 1

        action_data = json.dumps(action).encode()
        self._socket_out.sendall(
            len(action_data).to_bytes(self.RECV_SIZE, byteorder="big")
        )
        self._socket_out.sendall(action_data)

        ## Reactio: Receive observation from WeBots
        data_length = int.from_bytes(self.conn.recv(self.RECV_SIZE), byteorder="big")
        obs_data = self.receive_all(self.conn, data_length)
        obs = json.loads(obs_data.decode())

        # obs: Dict ['position': (3,), 'velocity': (2,), 'image': (3, 64, 64)]
        return {
            "position": [0, 0, 0],
            "velocity": [0, 0],
            "image": np.array(obs["image"], dtype=np.uint8),
        }

    def render(self, *args, **kwargs):
        raise NotImplementedError
        # if kwargs.get('mode', 'rgb_array') != 'rgb_array':
        #   raise ValueError("Only render mode 'rgb_array' is supported.")
        # <class 'numpy.ndarray'> (64, 64, 3) [0, 255]
        # return self._env.physics.render(*self._size, camera_id=self._camera)
