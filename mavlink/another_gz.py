import time
from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialect
import gz
from mission_types import Waypoint


def return_to_launch(connection):
	connection.mav.command_long_send(
		connection.target_system,
		connection.target_component,
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
	ack_sync(connection, "COMMAND_ACK")


def check_waypoint_reached(connection, i, lat, lon, alt):
	# Check if the vehicle has reached the waypoint
	# This is a placeholder function. You should implement the actual logic to check if the waypoint is reached.

	reached = connection.recv_match(
		type="MISSION_ITEM_REACHED",
		condition=f"MISSION_ITEM_REACHED.seq == {i}",
		blocking=True,
	)
	return str(reached)


if __name__ == "__main__":
	connection = gz.GazeboConnection("udp:127.0.0.1:14550")

	connection.arm()
	connection.takeoff(10)
	lat, lon, alt = connection.get_current_gps_location()
	connection.log(f"📍Initial Current location → lat: {lat}, lon: {lon}, actual alt: {alt}")

	_waypoints = [
		Waypoint(lat + 0.00001, lon + 0.00001, 3),
		Waypoint(lat + 0.00001, lon, 3),
		Waypoint(lat - 0.00001, lon, 3),
		Waypoint(lat, lon + 0.00001, 3),
		Waypoint(lat, lon - 0.00001, 3),
	]

	connection.upload_mission(_waypoints)
	time.sleep(1)

	connection.start_mission()
	time.sleep(1)
	connection.log("Mission started successfully.")

	# for i in range(5):
	# 	connection.check_mission_waypoint_reached(i)
	# 	time.sleep(1)
	connection.monitor_mission_progress()
	time.sleep(10)
	try:
		connection.clear_mission()
		time.sleep(1)
		connection.return_to_launch()
	except (mavutil.mavlink.MAVError, IOError) as e:
		connection.log(f"Error during mission upload: {e}")