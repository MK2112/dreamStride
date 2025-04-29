import socket
import json

HOST = "localhost"  # Standard loopback interface address (localhost)
PORT = 9999  # Port to listen on (non-privileged ports are > 1023)

def dummy_server():
    print("ALIVE")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                try:
                    message_length = int.from_bytes(conn.recv(4), byteorder="big")
                    full_data = conn.recv(message_length)
                    received_data = full_data.decode()
                    received_json = json.loads(received_data)
                    print(f"Received JSON: {received_json}")
                    # Process the received JSON object here
                except (ConnectionError, json.JSONDecodeError) as e:
                    print(f"Error: {e}")
                    break  # Handle errors and break the connection


if __name__ == "__main__":
    dummy_server()