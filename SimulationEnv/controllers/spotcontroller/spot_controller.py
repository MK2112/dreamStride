import socket
import json
import time
import numpy as np
import cv2
from controller import Robot, Motor, Camera, Supervisor, GPS

EPISODE_LIMIT = 10000
STEPS_PER_EPISODE = 200

"""
This is a controller that focuses solely on modular integration with WeBots.
DayDreamer is communicated to by "shoving" observations and environment-emitted rewards into a socket.
In return, we expect to be "shoved" back actions to be taken by the robot.
"""

HOST = "localhost"
PORT_CONTROLLER = 9998  # Port to listen on
PORT_DAYDREAMER = 9999  # Port to connect to
BUFFER_SIZE = 1024  # 1 KB buffer for incoming/sent data via socket

GPS_BEACON = [2.59, -3.01, 0.624]  # xyz
D_THRESHOLD = 0.5

spot_robot = Supervisor()  # Robot()

timestep = int(spot_robot.getBasicTimeStep())

# (12) Motors
f_l_s_a_motor = spot_robot.getDevice("front left shoulder abduction motor")
f_l_s_r_motor = spot_robot.getDevice("front left shoulder rotation motor")
f_l_e_motor = spot_robot.getDevice("front left elbow motor")
f_r_s_a_motor = spot_robot.getDevice("front right shoulder abduction motor")
f_r_s_r_motor = spot_robot.getDevice("front right shoulder rotation motor")
f_r_e_motor = spot_robot.getDevice("front right elbow motor")
r_l_s_a_motor = spot_robot.getDevice("rear left shoulder abduction motor")
r_l_s_r_motor = spot_robot.getDevice("rear left shoulder rotation motor")
r_l_e_motor = spot_robot.getDevice("rear left elbow motor")
r_r_s_a_motor = spot_robot.getDevice("rear right shoulder abduction motor")
r_r_s_r_motor = spot_robot.getDevice("rear right shoulder rotation motor")
r_r_e_motor = spot_robot.getDevice("rear right elbow motor")
# Input range is -1 to 1

# (5) Cameras
l_h_cam = spot_robot.getDevice("left head camera")  # 1080 720
r_h_cam = spot_robot.getDevice("right head camera")  # 1080 720
l_f_cam = spot_robot.getDevice("left flank camera")  # 1080 720
r_f_cam = spot_robot.getDevice("right flank camera")  # 1080 720
r_cam = spot_robot.getDevice("rear camera")  # 1080 720

# GPS
gps = spot_robot.getDevice("gps")
motors = [
    f_l_s_a_motor,
    f_l_s_r_motor,
    f_l_e_motor,
    f_r_s_a_motor,
    f_r_s_r_motor,
    f_r_e_motor,
    r_l_s_a_motor,
    r_l_s_r_motor,
    r_l_e_motor,
    r_r_s_a_motor,
    r_r_s_r_motor,
    r_r_e_motor,
]
cameras = [l_h_cam, r_h_cam, l_f_cam, r_f_cam, r_cam]  # Keep the power bill down

for camera in cameras:
    camera.enable(timestep)

gps.enable(timestep)

##### Socket-based interconnection ######

# Controller <- DayDreamer
# This has to be setup before going into waiting loop
# so that DayDreamer can connect to it
socket_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_in.bind((HOST, PORT_CONTROLLER))
socket_in.listen()

# Controller -> DayDreamer
# Yes, we loop the hell out of this until DayDreamer is ready to connect
# I really really really don't want to take any hostages, so this is the way to go
socket_out = None
while socket_out is None:
    try:
        socket_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_out.connect((HOST, PORT_DAYDREAMER))
        print("[+] Successfully connected to DayDreamer!")
    except Exception as e:
        print("[WeBots Controller] Connection pending ...")
        socket_out = None
        time.sleep(1)


def calculate_reward(gps, time_taken):
    base_reward = 100.0
    reward = base_reward - (0.075 * time_taken)
    if not (
        time_taken > 0
        and GPS_BEACON[0] + D_THRESHOLD > gps[0] > GPS_BEACON[0] - D_THRESHOLD
        and GPS_BEACON[1] + D_THRESHOLD > gps[1] > GPS_BEACON[1] - D_THRESHOLD
    ):
        # the closer the robot is to the goal, the higher the reward
        reward -= 0.75 * (abs(GPS_BEACON[0] - gps[0]) + abs(GPS_BEACON[1] - gps[1]))
    return reward


def act_out(action):
    raise NotImplementedError("This is a placeholder for the action realization")


def get_env_state(cameras):
    # I really tried with getImage, but patience wore thin and expectations for numpy were higher
    l_h_img = np.array(cameras[0].getImageArray()).astype(np.uint8)
    r_h_img = np.array(cameras[1].getImageArray()).astype(np.uint8)
    # We need to achieve a locally sound shape to correctly resize and concatenate the images
    # I just can't with this resizing stuff, takes eternities to run and barely gets the stuff anywhere effectively
    l_h_img = cv2.cvtColor(l_h_img, cv2.COLOR_BGR2RGB).reshape(720, 1080, 3)
    r_h_img = cv2.cvtColor(r_h_img, cv2.COLOR_BGR2RGB).reshape(720, 1080, 3)
    resized_l_h_img = cv2.resize(l_h_img, (128, 64), interpolation=cv2.INTER_AREA)
    resized_r_h_img = cv2.resize(r_h_img, (128, 64), interpolation=cv2.INTER_AREA)
    # Yes, it's right, then left image - don't ask me why
    return np.concatenate((resized_r_h_img, resized_l_h_img), axis=1)  # 64x256x3


try:
    time_taken = 0
    conn, addr = socket_in.accept()
    with conn:
        print(f"> Connection from {addr}")
        while spot_robot.step(timestep) != -1:
            try:
                full_data = socket_in.recv(
                    int.from_bytes(socket_in.recv(4), byteorder="big")
                )
                action_data = full_data.decode()
                action = json.loads(action_data)
                # Process the received JSON object here
            except (ConnectionError, json.JSONDecodeError) as e:
                print(f"[!] Error: {e}")
                break  # Handle errors and break the connection

            ## Actio
            if "reset" in action:
                print("Resetting ...")
                time_taken = 0
                spot_robot.simulationReset()
            else:
                act_out(action)

            ## Reactio
            # I found this structure within DayDreamer's ppo.py and in some other robot definitions
            # (xarm_demo.py and a1.py I think)
            obs = {
                "image": get_env_state(cameras),
                "reset": 0,  # TODO: Make this indicate 1 on successful, completed reset
                "reward": calculate_reward(
                    gps.getValues(), time_taken
                ),  # xyz, timestep
            }

            # General structure:
            # obs = {
            #   "image":  [np image array],
            #   "reset": 0,
            #   "reward": 100.0
            # }

            # Send to algorithm backend -> JSON string
            json_data = json.dumps(obs).encode()
            socket_out.sendall(len(json_data).to_bytes(4, byteorder="big"))
            socket_out.sendall(json_data)
            time_taken += 1
finally:
    socket_in.close()
    socket_out.close()
