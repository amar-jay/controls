import subprocess
import cv2
from pymavlink import mavutil
import math
import time
import pymavlink.dialects.v20.all as dialect

def enable_streaming(world="our_runway", model_name="iris_with_gimbal", camera_link="pitch_link") -> bool:
    """
    Enable streaming for the camera in the Gazebo simulation.
    """
	
    command = [
        "gz",
        "topic",
        "-t",
        f"/world/{world}/model/{model_name}/model/gimbal/link/{camera_link}/sensor/camera/image/enable_streaming",
        # "/world/our_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming",
        "-m",
        "gz.msgs.Boolean",
        "-p",
        "data: 1",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("The current gimbal topic is", command[3])
        print("🦾 Gazebo gimbal streaming enabled...", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        print("The current topic is", command[2])
        return False


def point_gimbal_downward(topic="/gimbal/cmd_tilt", angle=0) -> bool:
    """
    Uses gz command line to point gimbal downward.
    """
    command = [
        "gz",
        "topic",
        "-t",
        f"{topic}",
        "-m",
        "gz.msgs.Double",
        "-p",
        f"data: {angle}",
    ]
	 
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[CAMERA] Gimbal pointed to angle:", angle, "degrees. On topic:", topic)
        return True
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return False


def arm(connection):
    """
    Arms the vehicle and sets it to GUIDED mode.

    Parameters:
        connection: The MAVLink connection object.
    """
    # Wait for a heartbeat from the vehicle
    print("Waiting for heartbeat...")
    connection.wait_heartbeat()
    print(f"Heartbeat received from system {connection.target_system}")

    # Set mode to GUIDED (or equivalent)
    mode = "GUIDED"
    mode_id = connection.mode_mapping()[mode]
    connection.set_mode(mode_id)

    # Arm the vehicle
    print("Arming motors...")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,  # Confirmation
        1,
        0,
        0,
        0,
        0,
        0,
        0,  # Arm (1 to arm, 0 to disarm)
    )

    # Wait for arming
    connection.motors_armed_wait()
    print("Motors armed!")

def takeoff(connection, target_altitude=5.0):
    """
    Initiates takeoff to target altitude in meters.
    """

    # Send takeoff command
    print(f"Taking off to {target_altitude} meters...")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,  # Confirmation
        0,
        0,
        0,
        0,
        0,
        0,
        target_altitude,  # Altitude
    )

    # Optional: wait for some time or monitor altitude via message stream
    time.sleep(20)  # crude wait; replace with altitude monitor if needed

    print("Takeoff command sent.")

class GazeboVideoCapture:
    def __init__(self):
        """
        Open a video stream from the Gazebo simulation.
        """
        pipeline = (
            "udpsrc port=5600 ! "
            "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "rtph264depay ! "
            "h264parse ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "appsink drop=1"
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream! Check sender or pipeline. pipeline=", pipeline)

    def get_capture(self):
        return self.cap

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fps = self.cap.get(cv2.CAP_PROP_FPS)
        return width, height, None

def get_current_gps_location(master, timeout=5.0):
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if not msg:
        print("❌ Timeout: Failed to receive GPS data.")
        return None

    lat = msg.lat / 1e7  # Convert from 1e7-scaled degrees to float degrees
    lon = msg.lon / 1e7
    alt = msg.alt / 1000.0  # Convert mm to meters (altitude AMSL)

    return lat, lon, alt


def goto_waypoint_basic(master, lat: float, lon: float, alt: float):
    """Send MAV_CMD_NAV_WAYPOINT to fly to (lat, lon, alt)."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0,  # confirmation
        0,
        0,
        0,
        0,  # params 1-4 unused
        lat,  # param 5: latitude
        lon,  # param 6: longitude
        alt,  # param 7: altitude (AMSL)
    )
    print(f"[MAVLink] Sent waypoint → lat: {lat}, lon: {lon}, alt: {alt}")


def goto_waypoint_sync(
    master, lat: float, lon: float, alt: float, radius_m=2.0, alt_thresh=1.0, timeout=20
):
    """
    Send drone to waypoint (lat, lon, alt) and wait until it's close enough.

    Args:
        master: MAVLink connection (pymavlink instance).
        lat, lon: Target latitude/longitude in degrees.
        alt: Target altitude in meters (AMSL).
        radius_m: Horizontal threshold in meters to consider "arrived".
        alt_thresh: Vertical (altitude) threshold in meters.
        timeout: Max seconds to wait for arrival.
    """
    # Send command
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0,
        0,
        0,
        0,
        0,
        lat,
        lon,
        alt,
    )
    print(f"[MAVLink] Sent waypoint → lat={lat}, lon={lon}, alt={alt}")

    # Convert lat/lon to scaled int32 used in GLOBAL_POSITION_INT
    target_lat = int(lat * 1e7)
    target_lon = int(lon * 1e7)
    target_alt = int(alt * 1000)  # in mm

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    start_time = time.time()
    while time.time() - start_time < timeout:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg:
            current_lat = msg.lat
            current_lon = msg.lon
            current_alt = msg.alt  # in mm

            # compute distance
            dist = haversine(
                current_lat / 1e7, current_lon / 1e7, target_lat, target_lon
            )
            alt_diff = abs(current_alt - target_alt) / 1000.0

            print(f"Distance: {dist:.1f} m, Alt diff: {alt_diff:.2f} m")

            if dist <= radius_m and alt_diff <= alt_thresh:
                print("✅ Reached waypoint.")
                return True
        else:
            print("⚠️ No GLOBAL_POSITION_INT received.")

    print("❌ Timeout: did not reach waypoint in time.")
    return False

def clear_mav_missions(connection):
    print("[MAVLink] Clearing all missions. Hack...")
    # Clear all missions to prevent interference
    connection.mav.mission_clear_all_send(connection.target_system, connection.target_component)
    time.sleep(0.5)  # Give the FCU some breathing room

    # Set to GUIDED mode explicitly (you can also use MAV_MODE_AUTO if that suits your logic)
    connection.set_mode("GUIDED")  # Or use command_long if you don't have helper



# Global state
_waypoint_state = {}
WAIT_FOR_PICKUP_CONFIRMATION_TIMEOUT = 10  # seconds
pickup_confirmation_counter = 0

def goto_waypoint(
    master, lat: float, lon: float, alt: float, radius_m=.5, alt_thresh=1.0, timeout=20, alt_compensation=0.0
):
    """
    Initiate waypoint navigation. This does not block.
    """
    print(
        f"[MAVLink] goto_waypoint: lat={lat}, lon={lon}, alt={alt}, radius_m={radius_m}, alt_thresh={alt_thresh}, timeout={timeout}"
    )

    # Check if a waypoint is already in progress
    if _waypoint_state:
        print(
            "❌ A waypoint is already in progress. Please wait until it is completed."
        )
        return

    # Send command to go to waypoint
    # long doesn't work in Gazebo, so we use MAV_CMD_NAV_WAYPOINT
    # instead of MAV_CMD_DO_SET_MODE
    # this is a workaround for the fact that
    # the drone doesn't have a mode for "goto waypoint"
    # and we can't use the standard MAVLink command
    # because Gazebo doesn't support it
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0,
        0,
        0,
        0,
        0,
        lat,
        lon,
        alt + alt_compensation,
    )



    # message = dialect.MAVLink_mission_item_int_message(
    #     target_system=master.target_system,
    #     target_component=master.target_component,
    #     seq=0,
    #     frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    #     command=dialect.MAV_CMD_NAV_WAYPOINT,
    #     current=2,
    #     autocontinue=0,
    #     param1=0,
    #     param2=0,
    #     param3=0,
    #     param4=0,
    #     x = int(lat * 1e7),
    #     y = int(lon * 1e7),
    #     z = int(alt), # DOESN'T TAKE alt/1000 nor compensated altitude 
    #     # z = int(alt)   # in mm
    # )

    master.mav.send(message)
    print(f"[MAVLink] Sent waypoint → lat={lat}, lon={lon}, alt={alt}")

    # Store state
    _waypoint_state["target_lat"] = int(lat * 1e7)
    _waypoint_state["target_lon"] = int(lon * 1e7)
    _waypoint_state["target_alt"] = int(alt)
    _waypoint_state["radius_m"] = radius_m
    _waypoint_state["alt_thresh"] = alt_thresh
    _waypoint_state["start_time"] = time.time()
    _waypoint_state["timeout"] = timeout
    _waypoint_state["master"] = master
    _waypoint_state["alt_compensation"] = alt_compensation


def check_waypoint_reached():
    """
    Check if the drone has reached the previously set waypoint.
    Returns:
        - True if reached
        - False if not yet reached
        - None if timed out or no waypoint started
    """
    if not _waypoint_state:
        return None  # No waypoint in progress

    master = _waypoint_state["master"]
    print("[DEBUG] Heartbeat timestamp:", master.wait_heartbeat)
    print("[DEBUG] System ID:", master.target_system)
    print("[DEBUG] Component ID:", master.target_component)
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)

    if not msg:
        msg = master.recv_match(blocking=False)
        if msg:
            print("[MAVLink 🔁]", msg.get_type(), msg.to_dict())
        else:
            print("[MAVLink 🚫] Nothing coming in")

        print("[DEBUG]⚠️ No GLOBAL_POSITION_INT received.")
        return False  # No new data yet

    current_lat = msg.lat
    current_lon = msg.lon
    current_alt = msg.alt/1000 - _waypoint_state["alt_compensation"]  # in m
    print(f"{current_alt=} {msg.alt=} {_waypoint_state['alt_compensation']=}")

    target_lat = _waypoint_state["target_lat"]
    target_lon = _waypoint_state["target_lon"]
    target_alt = _waypoint_state["target_alt"]

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    dist = haversine(
        current_lat / 1e7, current_lon / 1e7, target_lat / 1e7, target_lon / 1e7
    )
    alt_diff = abs(current_alt - target_alt) / 1000.0
    print(f"Current location: lat={current_lat}, lon={current_lon}, alt={current_alt}")
    print(f"Target location: lat={target_lat}, lon={target_lon}, alt={target_alt}")
    print(f"Distance: {dist:.1f} m, Alt diff: {alt_diff:.2f} m")
    print("Current Alt:", current_alt, "Target Alt:", target_alt)
    print()

    if (
        dist <= _waypoint_state["radius_m"]
        and alt_diff <= _waypoint_state["alt_thresh"]
    ):
        print("✅ Reached waypoint.")
        # if check_pickup_confirmation():
        _waypoint_state.clear()
        # clear_mav_missions(master) # This is a hack to allow multiple waypoints to while bypassing the mission manager
        return True


    if time.time() - _waypoint_state["start_time"] > _waypoint_state["timeout"]:
        print("❌ Timeout: did not reach waypoint in time.")
        _waypoint_state.clear()
        return None

    return False  # Still on the way



def check_pickup_confirmation():
    """
    Check if the pickup confirmation has been received.
    """
    global pickup_confirmation_counter
    pickup_confirmation_counter += 1

    if pickup_confirmation_counter >= WAIT_FOR_PICKUP_CONFIRMATION_TIMEOUT:
        print("❌ Timeout: did not receive pickup confirmation in time.")
        return False

    # Placeholder: always returns True for this example
    return True

def is_pickup_confirmation_received():
    """
    Check if the pickup confirmation has been received.
    """
    global pickup_confirmation_counter
    # This is a placeholder. In a real implementation, you would check
    # for a specific MAVLink message or condition that indicates
    # the pickup confirmation. For this placeholder we will wait for 10 captures
    # and then return True.

    if (pickup_confirmation_counter >= WAIT_FOR_PICKUP_CONFIRMATION_TIMEOUT):
        pickup_confirmation_counter = 0
        return True
    else:
        return False


alt_compensation = 0.0  # Global variable to store altitude compensation
def get_base_alt()->int:
    """
    Get the base altitude compensation value.
    """
    global alt_compensation
    return alt_compensation

def update_alt_compensation(alt: int) -> int:
    """
    Update the altitude compensation value.
    """
    global alt_compensation
    alt_compensation += alt
    return alt