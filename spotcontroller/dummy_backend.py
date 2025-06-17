import socket
import json
import numpy as np

HOST = "localhost"  # Standard loopback interface address (localhost)
PORT_CONTROLLER = 9998  # Port to connect to (action sent to here)
PORT_DAYDREAMER = 9999  # Port to listen on (observ emitted from here)
RECV_SIZE = 8
BUFFER_SIZE = 1024


def receive_all(conn: socket.socket, length: int) -> bytes:
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
        chunk = conn.recv(min(BUFFER_SIZE, remaining_length))
        if not chunk:
            raise RuntimeError("Received data incomplete. Socket connection broken.")
        received_data += chunk
    return received_data


def dummy_server():
    print("[+] Dummy server started.")

    # Connect Socket to (HOST, PORT_DAYDREAMER) to listen to receiving data -> socket_in
    socket_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_in.bind((HOST, PORT_DAYDREAMER))

    # Connect Socket to (HOST, PORT_CONTROLLER) to send data -> socket_out
    socket_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_out.connect((HOST, PORT_CONTROLLER))

    print("[+] Reverse connection established")

    socket_in.listen()
    conn, addr = socket_in.accept()

    print(f"[+] Connected to controller at {addr}")

    # This is just an endurance test loop counter
    counter = 600  # approx. 20 minutes, i.e. 2 seconds per loop

    while True and counter > 0:
        ## Actio: Send mocked action to controller

        # This is how we want data to be formatted when
        # sent to the controller from Dreamer

        # Mockup: Create 12 random actions [-1, 1]
        np_actions = np.random.uniform(-1, 1, 12)
        action = {str(i): np_actions[i] for i in range(12)}

        # Either 0 (act) or 1 (reset)
        action["reset"] = 0

        action_data = json.dumps(action).encode()
        socket_out.sendall(len(action_data).to_bytes(RECV_SIZE, byteorder="big"))
        socket_out.sendall(action_data)

        ## Reactio: Receive observation from controller

        data_length = int.from_bytes(conn.recv(RECV_SIZE), byteorder="big")
        obs_data = receive_all(conn, data_length)
        obs = json.loads(obs_data.decode())

        # This is what we want to input into Dreamer
        print(f"Observation: {obs}")

        print(type(obs))
        print("Counter:", counter)
        counter -= 1


# Start Spot_Controller.py first, then run this script
if __name__ == "__main__":
    dummy_server()
