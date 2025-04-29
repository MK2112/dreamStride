import socket
import json
import time
import numpy as np
import cv2
from controller import Robot, Motor, Camera, Supervisor, GPS

# This is indent ugly, but helps tons in PyCharm configuration
if __name__ == "__main__":
    """
    Webots-side controller, focuses solely on *modular* integration with WeBots.
    Per time instance, we expect to be receiving an action to be taken by the robot.
    DayDreamer in return is communicated to by emitting observations and environment-emitted rewards.
    All of this is done via socket-based communication to ensure a modular and extendable structure.
    """

    HOST = "localhost"
    PORT_CONTROLLER = 9998            # Port to listen on (action data sent to here)
    PORT_DAYDREAMER = 9999            # Port to connect to (observ data emitted from here)
    GPS_BEACON = [2.59, -3.01, 0.624] # [x,y,z] of goal position
    D_THRESHOLD = 0.5                 # Distance threshold for goal position
    RECV_SIZE = 8                     # Size of the received data
    BUFFER_SIZE = 1024                # Size of the buffer for sending data

    spot_robot = Supervisor() # Not Robot() since we need to control the env for reset
    timestep = int(spot_robot.getBasicTimeStep()) # 32ms

    ## Sensor Initialization ##

    # (12) Motors (controller.motor.Motor)
    f_l_s_a_motor = spot_robot.getDevice("front left shoulder abduction motor")
    f_l_s_r_motor = spot_robot.getDevice("front left shoulder rotation motor")
    f_l_e_motor   = spot_robot.getDevice("front left elbow motor")
    f_r_s_a_motor = spot_robot.getDevice("front right shoulder abduction motor")
    f_r_s_r_motor = spot_robot.getDevice("front right shoulder rotation motor")
    f_r_e_motor   = spot_robot.getDevice("front right elbow motor")
    r_l_s_a_motor = spot_robot.getDevice("rear left shoulder abduction motor")
    r_l_s_r_motor = spot_robot.getDevice("rear left shoulder rotation motor")
    r_l_e_motor   = spot_robot.getDevice("rear left elbow motor")
    r_r_s_a_motor = spot_robot.getDevice("rear right shoulder abduction motor")
    r_r_s_r_motor = spot_robot.getDevice("rear right shoulder rotation motor")
    r_r_e_motor   = spot_robot.getDevice("rear right elbow motor")
    # Input range is [-1;1]

    # (5) Cameras (controller.camera.Camera)
    l_h_cam = spot_robot.getDevice('left head camera')   # (1080,720)
    r_h_cam = spot_robot.getDevice('right head camera')  # (1080,720)
    l_f_cam = spot_robot.getDevice("left flank camera")  # (1080,720)
    r_f_cam = spot_robot.getDevice("right flank camera") # (1080,720)
    r_cam   = spot_robot.getDevice("rear camera")        # (1080,720)

    # (1) GPS (controller.gps.GPS)
    gps = spot_robot.getDevice("gps")

    # Lists of all motors and cameras
    motors: list = [f_l_s_a_motor, f_l_s_r_motor, f_l_e_motor, f_r_s_a_motor, f_r_s_r_motor, f_r_e_motor,
                    r_l_s_a_motor, r_l_s_r_motor, r_l_e_motor, r_r_s_a_motor, r_r_s_r_motor, r_r_e_motor]
    cameras: list = [l_h_cam, r_h_cam, l_f_cam, r_f_cam, r_cam]

    ## Sensor Activation ##

    for camera in cameras:
        camera.enable(timestep)

    gps.enable(timestep)

    # Move *one single timestep* to make GPS sensors not return NaN
    spot_robot.step(timestep)

    ## Helper Functions ##

    def calculate_reward(gps, time_taken, penalty) -> float:
        """
        Custom reward function; Distance-based, time-decaying
        The closer the robot is to the goal, the higher the reward
        :param gps: GPS sensor values [x,y,z]
        :param time_taken: Passed time in this episode
        :return: Reward value for the current state
        """
        base_reward = 100.0 # Max attainble reward

        # Penalize harshly for out-of-range actions
        base_reward -= penalty * 25.0

        reward = base_reward - (0.1 * time_taken)
        if not (time_taken > 0
                and GPS_BEACON[0] + D_THRESHOLD > gps[0] > GPS_BEACON[0] - D_THRESHOLD
                and GPS_BEACON[1] + D_THRESHOLD > gps[1] > GPS_BEACON[1] - D_THRESHOLD):
            reward -= 0.85 * (abs(GPS_BEACON[0] - gps[0]) + abs(GPS_BEACON[1] - gps[1]))
        return reward


    def act_out(action: dict) -> float:
        """
        Actuate the motors based on the received action
        As long as the action indices don't change within controller, this relationship should just be learnable
        :param action: Dict, expected to look like {'i': value, 'j': value, 'k': value} with i,j,k being motor indices
        :return: Penalty for out-of-range actions
        """
        total_penalty = 0.0
        for i, motor in enumerate(motors):
            motor_position = action[str(i)]
            min_position = motor.getMinPosition()
            max_position = motor.getMaxPosition()

            if not (min_position <= motor_position <= max_position):
                total_penalty += abs(min_position - motor_position) if motor_position < min_position else abs(motor_position - max_position)

        # Scale action from [-1, 1] to [min_position, max_position]
        scaled_position = min_position + (motor_position + 1) * 0.5 * (max_position - min_position)
        motor.setPosition(scaled_position)
        return total_penalty


    def get_env_state(cameras: list) -> list:
        """
        Get the environment state as a single image
        :param cameras: List of Camera objects
        :return: NumPy array of shape (64, 256, 3)
        """
        # I really tried with getImage, but patience wore thin, Numpy prevailed
        l_h_img = np.array(cameras[0].getImageArray()).astype(np.uint8)
        r_h_img = np.array(cameras[1].getImageArray()).astype(np.uint8)
        # I just can't with this resizing stuff, takes eternities to run and barely gets the stuff anywhere effectively
        l_h_img = cv2.cvtColor(l_h_img, cv2.COLOR_BGR2RGB).reshape(720, 1080, 3)
        r_h_img = cv2.cvtColor(r_h_img, cv2.COLOR_BGR2RGB).reshape(720, 1080, 3)
        resized_l_h_img = cv2.resize(l_h_img, (128, 64), interpolation=cv2.INTER_AREA)
        resized_r_h_img = cv2.resize(r_h_img, (128, 64), interpolation=cv2.INTER_AREA)
        return np.concatenate((resized_r_h_img, resized_l_h_img), axis=1).tolist() # 64x256x3; NumPy array would cause havoc in JSON serialization


    def shutdown() -> None:
        """
        Gracefully shut down the sockets and simulation
        """
        socket_in.close()
        socket_out.close()


    ## Socket-based interconnection ##

    print("[+] Spot Controller is alive. Waiting for connections...")

    # Controller <- DayDreamer
    # This is set up before going into waiting loop
    # so that DayDreamer can connect to it
    socket_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_in.bind((HOST, PORT_CONTROLLER))
    socket_in.listen()
    conn, addr = socket_in.accept()
    print(f"[+] [Controller <- DayDreamer] Downlink Connection from {addr}")

    # Controller -> DayDreamer
    # We loop the living hell out of this until DayDreamer is ready to connect
    # This seems the most universal way to go
    socket_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_out.connect((HOST, PORT_DAYDREAMER))
    print("[+] [Controller -> DayDreamer] Uplink Connection established")

    ## Main loop ##

    with conn:
        time_taken = 0
        while spot_robot.step(timestep) != -1:
            ########
            ## Actio
            ########
            try:
                full_data = conn.recv(int.from_bytes(conn.recv(RECV_SIZE), byteorder="big"))
                action_data = full_data.decode()
                action = json.loads(action_data)
                # Process the received JSON object here
            except (ConnectionError, json.JSONDecodeError) as e:
                print(f"[!] Downlink Error: [{time.time()}] {e}")
                shutdown()
                break

            # Timestep reset flag
            was_reset = 0
            # Penalty for out-of-range actions
            oor_penalty = 0.0

            # Check if action contains reset dict key
            if not "reset" in action:
                action["reset"] = 0

            if action["reset"] == 1:
                # Resetting the simulation and counters
                time_taken = 0
                was_reset = 1
                spot_robot.simulationReset()
            else:
                action.pop("reset")
                oor_penalty = act_out(action)

            ##########
            ## Reactio
            ##########
            # I found this structure within DayDreamer's ppo.py and in some other robot definitions
            # (xarm_demo.py and a1.py, I think)

            current_gps = gps.getValues()

            obs = {
                "image": get_env_state(cameras),
                "reset": was_reset,
                "done": True if current_gps and (GPS_BEACON[0] + D_THRESHOLD > current_gps[0] > GPS_BEACON[0] - D_THRESHOLD and GPS_BEACON[1] + D_THRESHOLD > current_gps[1] > GPS_BEACON[1] - D_THRESHOLD) else False,
                "reward": calculate_reward(current_gps, time_taken, oor_penalty) # https://cyberbotics.com/doc/reference/gps?tab-language=python#wb_gps_get_values
            }

            # Send to DayDreamer -> JSON object
            json_data = json.dumps(obs).encode()
            data_length = len(json_data)
            socket_out.sendall(data_length.to_bytes(RECV_SIZE, byteorder="big"))

            sent_bytes = 0
            while sent_bytes < data_length:
                chunk = json_data[sent_bytes:sent_bytes + BUFFER_SIZE]
                sent_bytes += socket_out.send(chunk)

            time_taken += 1