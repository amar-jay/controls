import controls.mavlink.gz as gz
from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialect
import time


def _set_mode(master, mode):
    mode_id = master.mode_mapping()[mode]
    master.set_mode(mode_id)


def ack_sync(master, msg):
    while True:
        m = master.recv_match(type=msg, blocking=True)
        if m is not None:
            if m.get_type() == msg:
               return
            print(f"Received {m.get_type()} instead of {msg}")
        else:
            continue


def arm(master):
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Arming vehicle...")

    master.motors_armed_wait()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
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
    print("Vehicle armed.")


def takeoff(master, target_altitude):
    print(f"Taking off to {target_altitude} m...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,  # Confirmation
        0,  # Param1: unused
        0,  # Param2: unused
        0,  # Param3: unused
        0,  # Param4: unused
        0,  # Param5: unused
        0,  # Param6: unused
        target_altitude,  # Param7: target altitude in meters
    )
    ack_sync(master, "COMMAND_ACK")
    time.sleep(10)


def start_mission(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_MISSION_START,
        0,  # Confirmation
        0,  # Param1: unused
        0,  # Param2: unused
        0,  # Param3: unused
        0,  # Param4: unused
        0,  # Param5: unused
        0,  # Param6: unused
        0,  # Param7: unused
    )
    ack_sync(master, "COMMAND_ACK")




def return_to_launch(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,  # Confirmation
        0,  # Param1: unused
        0,  # Param2: unused
        0,  # Param3: unused
        0,  # Param4: unused
        0,  # Param5: unused
        0,  # Param6: unused
        0,  # Param7: unused
    )
    ack_sync(master, "COMMAND_ACK")


def upload_mission(master, waypoints):
    num_wp = len(waypoints)
    print(f"Uploading {num_wp} waypoints...")

    # send mission count
    master.mav.mission_count_send(master.target_system, master.target_component, num_wp)
    ack_sync(master, "MISSION_REQUEST")

    for i, waypoint in enumerate(waypoints):
        if isinstance(waypoint, hover):
            print(f"Hovering for {waypoint.time} seconds at waypoint {i}")
            master.mav.mission_item_send(
					 target_system=master.target_system,  # System ID
					 target_component=master.target_component,  # Component ID
					 seq=i,  # Sequence number for item within mission (indexed from 0).
					 frame=0,  # The coordinate system of the waypoint.
					 command=dialect.MAV_CMD_NAV_LOITER_TIME,
					 current=1 if i == 0 else 0,
					 autocontinue=1,
					 param1=waypoint.time, 
					 param2=0,
					 param3=0,
					 param4=0,
					 x=0,
					 y=0,
					 z=0,
				)
            ack_sync(master, "MISSION_REQUEST")
            continue
		
		  # send mission item
        master.mav.mission_item_send(
            target_system=master.target_system,  # System ID
            target_component=master.target_component,  # Component ID
            seq=i,  # Sequence number for item within mission (indexed from 0).
            frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # The coordinate system of the waypoint.
            command=dialect.MAV_CMD_NAV_WAYPOINT,
            current=1
            if i == 0
            else 0,  # 1 if this is the current waypoint, 0 otherwise.
            autocontinue=0,
            param1=0,  # 	Hold time. (ignored by fixed wing, time to stay at waypoint for rotary wing)
            param2=0,  # Acceptance radius (if the sphere with this radius is hit, the waypoint counts as reached)
            param3=0,  # 	Pass the waypoint to the next waypoint (0 = no, 1 = yes)
            param4=0,  # Desired yaw angle at waypoint (rotary wing). NaN to use the current system yaw heading mode (e.g. yaw towards next waypoint, yaw to home, etc.).
            x=waypoint.x,  # Latitude in degrees * 1E7
            y=waypoint.y,  # Longitude in degrees * 1E7
            z=waypoint.z,  # DOESN'T TAKE alt/1000 nor compensated altitude
        )

        if i != num_wp - 1:
            ack_sync(master, "MISSION_REQUEST")
            print(f"Waypoint {i} uploaded: {waypoint}")

    ack_sync(master, "MISSION_ACK")


class waypoint:
    def __init__(self, lat, lon, alt, relative_to=None):
        if relative_to is None:
            self.x = lat
            self.y = lon
            self.z = alt
            return
        else:
            self.set_gps(lat, lon, alt, relative_to)

    def set_gps(self, lat, lon, alt, relative_to=(0, 0, 0)):
        self.x = lat - relative_to[0]
        self.y = lon - relative_to[1]
        self.z = alt

    def __repr__(self):
        return f"Waypoint(lat={self.x}, lon={self.y}, alt={self.z})"

class hover:
	 def __init__(self, time):
		  self.time = time

def check_waypoint_reached(master, i, lat, lon, alt):
	 # Check if the vehicle has reached the waypoint
	 # This is a placeholder function. You should implement the actual logic to check if the waypoint is reached.
	 
	reached = master.recv_match(type="MISSION_ITEM_REACHED", condition=f"MISSION_ITEM_REACHED.seq == {i}", blocking=True)
	return str(reached)



if __name__ == "__main__":
    master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    master.wait_heartbeat(timeout=10)

    _set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, 10)

    _set_mode(master, "AUTO")

    location = gz.get_current_gps_location(master)
    if location is None:
        print("❌ Failed to get current GPS location.")
        exit(1)

    lat, lon, alt = location
    print(f"📍Initial Current location → lat: {lat}, lon: {lon}, actual alt: {alt}")

    waypoints = [
        waypoint(lat + .00001, lon + .00001, 3),
        waypoint(lat + .00001, lon, 3),
        waypoint(lat - .00001, lon, 3),
        waypoint(lat, lon + .00001, 3),
        waypoint(lat, lon - .00001, 3),
		#   hover(5),
      #   waypoint(-35.36312509, 149.16519576, 5),
		#   hover(.5),
      #   waypoint(lat, lon+0.001, 3, relative_to=location),
      # #   waypoint(-34.36361444, 150.16469740, 20, relative_to=location),
      #   waypoint(lat, lon + .001, 13, relative_to=location),
		#   waypoint(0, 0, 0),
    ]
    print(f"📍 Waypoints: {waypoints}")

    try:
        clear_mission(master)
        time.sleep(1)
        upload_mission(master, waypoints)
        time.sleep(1)
        start_mission(master)

        for i in range(len(waypoints)):
            print(f"Waiting for waypoint {i} to be reached...")

            # reached = check_waypoint_reached(master, i+1, waypoint.x, waypoint.y, waypoint.z)
            ack_sync(master, "MISSION_ITEM_REACHED")
            time.sleep(1)
            # if reached:
            #    print(f"Waypoint {i} reached.")
            # else:
            #    print(f"Waypoint {i} not reached yet.")
            #    time.sleep(1)
        return_to_launch(master)
    except Exception as e:
        print(f"Error during mission upload: {e}")
