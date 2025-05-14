import math
import time
from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialect
from .mission_types import Waypoint, deprecated_method

# ========== ========= ========= =========
# ========== Global Variables ==========
# ========== ========= ========= =========
_waypoint_state = {}
WAIT_FOR_PICKUP_CONFIRMATION_TIMEOUT = 10  # seconds
pickup_confirmation_counter = 0
alt_compensation = 0.0  # to store altitude compensation
# ========= ========= ======== =========
# ========== ========= ========= =========


class ArdupilotConnection:
	def __init__(self, connection_string, wait_heartbeat=10, logger=None):
		self.connection_string = connection_string
		self.target_system = 1
		self.target_component = 1
		self.master = mavutil.mavlink_connection(connection_string)
		self.master.wait_heartbeat(wait_heartbeat)
		self.log = lambda *args: logger(*args) if logger else print("[MAVLink] ", *args)
		self.log(
			f"Connected to {self.connection_string} with system ID {self.master.target_system}"
		)

		self.home_position = self.get_current_gps_location()

	def _set_mode(self, mode):
		mode_id = self.master.mode_mapping()[mode]
		self.master.set_mode(mode_id)

	def ack_sync(self, msg):
		while True:
			m = self.master.recv_match(type=msg, blocking=True)
			if m is not None:
				if m.get_type() == msg:
					return m
				self.log(f"Received {m.get_type()} instead of {msg}")
			else:
				continue

	def arm(self):
		"""
		Arms the vehicle and sets it to GUIDED mode.
		"""
		# Wait for a heartbeat from the vehicle
		self.log("Waiting for heartbeat...")
		self.master.wait_heartbeat()
		self.log(f"Heartbeat received from system {self.master.target_system}")

		# Set mode to GUIDED (or equivalent)
		mode = "GUIDED"
		self._set_mode(mode)

		# Arm the vehicle
		self.log("Arming motors...")
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
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

		self.master.motors_armed_wait()
		# Wait for arming
		self.log("Vehicle armed!")

	def disarm(self):
		"""
		Disarms the vehicle.
		"""
		self.log("Disarming motors...")
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
			0,  # Confirmation
			0,
			0,
			0,
			0,
			0,
			0,  # Disarm (1 to arm, 0 to disarm)
		)

		self.master.motors_disarmed_wait()
		self.log("Vehicle disarmed!")

	def takeoff(self, target_altitude=5.0, wait_time=10):
		"""
		Initiates takeoff to target altitude in meters.
		"""

		# Send takeoff command
		self.log(f"Taking off to {target_altitude} meters...")
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
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
		time.sleep(wait_time)  # crude wait; replace with altitude monitor if needed

		self.log("Takeoff command sent.")

	def return_to_launch(self):
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
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
		self.ack_sync("COMMAND_ACK")

	def upload_mission(self, waypoints: list[Waypoint], relative=False):
		num_wp = len(waypoints)
		self.log(f"Uploading {num_wp} waypoints...")

		# send mission count
		self.master.mav.mission_count_send(
			self.master.target_system, self.master.target_component, num_wp
		)
		self.ack_sync("MISSION_REQUEST")
		for i, waypoint in enumerate(waypoints):
			print(
				f"Uploading waypoint {i}: lat={waypoint.lat}, lon={waypoint.lon}, alt={waypoint.alt}, hold={waypoint.hold}"
			)
			# send mission item
			self.master.mav.mission_item_send(
				target_system=self.master.target_system,  # System ID
				target_component=self.master.target_component,  # Component ID
				seq=i,  # Sequence number for item within mission (indexed from 0).
				frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # The coordinate system of the waypoint.
				command=dialect.MAV_CMD_NAV_WAYPOINT,
				current=1
				if i == 0
				else 0,  # 1 if this is the current waypoint, 0 otherwise.
				autocontinue=0,
				param1=waypoint.hold,  # 	Hold time. (ignored by fixed wing, time to stay at waypoint for rotary wing)
				param2=0,  # Acceptance radius (if the sphere with this radius is hit, the waypoint counts as reached)
				param3=0,  # 	Pass the waypoint to the next waypoint (0 = no, 1 = yes)
				param4=0,  # Desired yaw angle at waypoint (rotary wing). NaN to use the current system yaw heading mode (e.g. yaw towards next waypoint, yaw to home, etc.).
				x=waypoint.lat + self.home_position[0],  # Latitude in degrees * 1E7
				y=waypoint.lon + self.home_position[1],  # Longitude in degrees * 1E7
				z=waypoint.alt,  # Altitude in meters (AMSL) DOESN'T TAKE alt/1000 nor compensated altitude
			)
			if i != num_wp - 1:
				self.ack_sync("MISSION_REQUEST")
				self.log(f"Waypoint {i} uploaded: {waypoint}")

		self.ack_sync("MISSION_ACK")
		self.log("Mission upload complete.")

	def clear_mission(self):
		# Clear mission
		self.log("Clearing all missions. Hack...")
		self.master.mav.mission_clear_all_send(
			self.master.target_system, self.master.target_component
		)
		# time.sleep(0.5)  # Give the FCU some breathing room
		self.ack_sync("MISSION_ACK")

		# Set to GUIDED mode explicitly (you can also use MAV_MODE_AUTO if that suits your logic)
		# self.master.set_mode("GUIDED")  # Or use command_long if you don't have helper

	def start_mission(self):
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
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
		self.ack_sync("COMMAND_ACK")

	def get_current_gps_location(self, timeout=5.0):
		msg = self.master.recv_match(
			type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout
		)
		if not msg:
			self.log("❌ Timeout: Failed to receive GPS data.")
			return None

		lat = msg.lat / 1e7  # Convert from 1e7-scaled degrees to float degrees
		lon = msg.lon / 1e7
		alt = msg.alt / 1000.0  # Convert mm to meters (altitude AMSL)

		return lat, lon, alt

	def get_status(self):
		status = {
			"connected": False,
			"armed": False,
			"flying": False,
			"position": None,
			"mission_active": False,
			"current_waypoint": None,
			"total_waypoints": 0,
			"battery": None,
		}

		# Try receiving a few messages quickly
		for _ in range(20):
			msg = self.master.recv_match(
				type=[
					"HEARTBEAT",
					"GLOBAL_POSITION_INT",
					"MISSION_CURRENT",
					"MISSION_COUNT",
					"BATTERY_STATUS",
				],
				blocking=False,
			)

			if not msg:
				continue

			if msg.get_type() == "HEARTBEAT":
				status["connected"] = True
				status["armed"] = bool(
					msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
				)
				status["flying"] = msg.system_status == mavutil.mavlink.MAV_STATE_ACTIVE

			elif msg.get_type() == "GLOBAL_POSITION_INT":
				status["position"] = {
					"lat": msg.lat / 1e7,
					"lon": msg.lon / 1e7,
					"alt": msg.alt / 1e3,
				}

			elif msg.get_type() == "MISSION_CURRENT":
				status["current_waypoint"] = msg.seq
				status["mission_active"] = msg.seq > 0  # or some other logic

			elif msg.get_type() == "MISSION_COUNT":
				status["total_waypoints"] = msg.count

			elif msg.get_type() == "BATTERY_STATUS":
				voltages = [v for v in msg.voltages if v != 0xFFFF]
				status["battery"] = {
					"voltage": sum(voltages) / 1000 if voltages else None,
					"current": msg.current_battery / 100.0,
					"remaining": msg.battery_remaining,
				}

		return status

	def close(self):
		self.master.close()
		self.master = None
		self.log("Connection closed.")

	@deprecated_method
	def goto_waypoint(
		self,
		lat: float,
		lon: float,
		alt: float,
		radius_m=0.5,
		alt_thresh=1.0,
		timeout=20,
		alt_compensation=0.0,
	):
		"""
		Initiate waypoint navigation. This does not block.
		"""
		self.log(
			f"goto_waypoint: lat={lat}, lon={lon}, alt={alt}, radius_m={radius_m}, alt_thresh={alt_thresh}, timeout={timeout}"
		)

		# Check if a waypoint is already in progress
		if _waypoint_state:
			self.log(
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
		# master.mav.command_long_send(
		#     master.target_system,
		#     master.target_component,
		#     mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
		#     0,
		#     0,
		#     0,
		#     0,
		#     0,
		#     lat,
		#     lon,
		#     alt + alt_compensation,
		# )

		message = dialect.MAVLink_mission_item_int_message(
			target_system=self.master.target_system,
			target_component=self.master.target_component,
			seq=0,
			frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
			command=dialect.MAV_CMD_NAV_WAYPOINT,
			current=2,
			autocontinue=0,
			param1=0,
			param2=0,
			param3=0,
			param4=0,
			x=int(lat * 1e7),
			y=int(lon * 1e7),
			z=int(alt),  # DOESN'T TAKE alt/1000 nor compensated altitude
			# z = int(alt)   # in mm
		)

		self.master.mav.send(message)
		self.log(f"🛫 Sent waypoint → lat={lat}, lon={lon}, alt={alt}")

		# Store state
		_waypoint_state["target_lat"] = int(lat * 1e7)
		_waypoint_state["target_lon"] = int(lon * 1e7)
		_waypoint_state["target_alt"] = int(alt)
		_waypoint_state["radius_m"] = radius_m
		_waypoint_state["alt_thresh"] = alt_thresh
		_waypoint_state["start_time"] = time.time()
		_waypoint_state["timeout"] = timeout
		_waypoint_state["alt_compensation"] = alt_compensation

	def goto_waypointv2(
		self,
		lat: float,
		lon: float,
		alt: float,
		timeout=20,
		speed=0,
	):
		"""
		Initiate waypoint navigation. This does not block.
		"""
		self.log(f"goto_waypoint: lat={lat}, lon={lon}, alt={alt}, timeout={timeout}")

		# alt = self.master.location(relative_alt=True).alt
		# Send command to move to the specified latitude, longitude, and current altitude
		self.master.mav.command_int_send(
			self.master.target_system,
			self.master.target_component,
			dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,
			# “frame” = 0 or 3 for alt-above-sea-level, 6 for alt-above-home or 11 for alt-above-terrain
			dialect.MAV_CMD_DO_REPOSITION,
			0,  # Current
			0,  # Autocontinue
			speed,
			0,
			0,
			0,  # Params 2-4 (unused)
			int(lat * 1e7),
			int(lon * 1e7),
			alt,
		)

		self.log(f"🛫 Sent waypoint → lat={lat}, lon={lon}, alt={alt}")
		return self.check_reposition_reached()

	def check_reposition_reached(self):
		# Check for COMMAND_ACK message
		msg = self.master.recv_match(type="COMMAND_ACK", blocking=False)
		if msg:
			# Check if it's the reposition command and if it was acknowledged
			if msg.command == dialect.MAV_CMD_DO_REPOSITION:
				if msg.result == dialect.MAV_RESULT_ACCEPTED:
					self.log("✅ Reposition command accepted!")
					return True
				else:
					# Command was rejected
					result_codes = {
						dialect.MAV_RESULT_DENIED: "DENIED",
						dialect.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
						dialect.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
						dialect.MAV_RESULT_FAILED: "FAILED",
						dialect.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
					}
					result_str = result_codes.get(msg.result, f"Unknown ({msg.result})")
					self.log(f"❌ Reposition command rejected: {result_str}")
					return False
		return False

	def check_reposition_reached_timeout(self, ack_timeout=10):
		# Wait for command acknowledgment (with timeout)
		ack_timeout = 5.0  # 5 seconds to wait for ACK
		start_time = time.time()
		while time.time() - start_time < ack_timeout:
			# Check for COMMAND_ACK message
			msg = self.master.recv_match(type="COMMAND_ACK", blocking=False)
			if msg:
				# Check if it's the reposition command and if it was acknowledged
				if msg.command == dialect.MAV_CMD_DO_REPOSITION:
					if msg.result == dialect.MAV_RESULT_ACCEPTED:
						self.log("✅ Reposition command accepted!")
						return True
					else:
						# Command was rejected
						result_codes = {
							dialect.MAV_RESULT_DENIED: "DENIED",
							dialect.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
							dialect.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
							dialect.MAV_RESULT_FAILED: "FAILED",
							dialect.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
						}
						result_str = result_codes.get(
							msg.result, f"Unknown ({msg.result})"
						)
						self.log(f"❌ Reposition command rejected: {result_str}")
						return False

			time.sleep(0.1)
		self.log("⏱️ Timeout waiting for command acknowledgment")
		return True

	def check_mission_waypoint_reached(self, waypoint_seq, timeout=50.0):
		"""
		Check if a specific mission waypoint has been reached.
		Args:
		    waypoint_seq: The sequence number of the waypoint to check
		    timeout: Maximum time to wait for the message
		"""
		start_time = time.time()
		while time.time() - start_time < timeout:
			# Check for MISSION_ITEM_REACHED message
			msg = self.master.recv_match(type="MISSION_ITEM_REACHED", blocking=False)
			if msg and msg.seq == waypoint_seq:
				self.log(f"✅ Mission waypoint {waypoint_seq} reached!")
				return True

				# Alternative: Check MISSION_CURRENT to see if we've moved past this waypoint
			msg = self.master.recv_match(type="MISSION_CURRENT", blocking=False)
			if msg and msg.seq > waypoint_seq:
				self.log(f"✅ Mission has progressed past waypoint {waypoint_seq}!")
				return True

			time.sleep(0.1)
		self.log(f"❌ Timeout: Mission waypoint {waypoint_seq} not reached in time.")

		return False

	def monitor_mission_progress(self, timeout=600, _update_status_hook=None):
		self.log("Starting mission monitoring...")
		start_time = time.time()
		current_waypoint = 0
		total_waypoints = None

		while time.time() - start_time < timeout:
			msg = self.master.recv_match(
				type=["MISSION_CURRENT", "MISSION_COUNT"], blocking=False
			)

			if not msg:
				time.sleep(0.1)
				continue

			if msg.get_type() == "MISSION_COUNT":
				total_waypoints = msg.count
				self.log(f"Mission has {total_waypoints} waypoints")

			elif msg.get_type() == "MISSION_CURRENT":
				if msg.seq > current_waypoint:
					current_waypoint = msg.seq
					self.log(f"Reached waypoint {current_waypoint}")
					if _update_status_hook:
						_update_status_hook(current_waypoint, False)

					# Check if we've reached the final waypoint
					if total_waypoints and current_waypoint >= total_waypoints - 1:
						self.log("✅ Mission completed!")
						if _update_status_hook:
							_update_status_hook(current_waypoint, True)
						return True

			time.sleep(0.1)

		self.log("❌ Mission monitoring timed out")
		return False


if __name__ == "__main__":
	# Example usage
	connection = ArdupilotConnection("udp:127.0.0.1:14550")

	connection.arm()
	connection.takeoff(10)
	lat, lon, alt = connection.get_current_gps_location()

	connection.goto_waypointv2(lat + 0.00001, lon + 0.00001, 3)
	connection.goto_waypointv2(lat - 0.00002, lon - 0.00002, 3)
	connection.goto_waypointv2(lat + 0.00002, lon + 0.00002, 3)
	connection.goto_waypointv2(lat - 0.00001, lon + 0.00001, 3)
	connection.goto_waypointv2(lat + 0.00002, lon + 0.00001, 3)
	connection.goto_waypointv2(lat + 0.00002, lon - 0.00002, 3)

	time.sleep(5)
	connection.log("Waiting for waypoint to be reached...")

	connection.return_to_launch()
	connection.master.close()
	connection.log("Connection closed.")
	connection.log("Ardupilot connection example completed.")
