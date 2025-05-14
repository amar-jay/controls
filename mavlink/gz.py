from .ardupilot import ArdupilotConnection
import time
import subprocess
import math
import cv2
from pymavlink import mavutil


class GazeboVideoCapture:
	def __init__(self, camera_port=5600):
		"""
		Open a video stream from the Gazebo simulation.
		"""
		pipeline = (
			f"udpsrc port={camera_port} ! "
			"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
			"rtph264depay ! "
			"h264parse ! "
			"avdec_h264 ! "
			"videoconvert ! "
			"appsink drop=1"
		)

		self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

		if not self.cap.isOpened():
			raise RuntimeError(
				"Failed to open stream! Check sender or pipeline. pipeline=", pipeline
			)
		# move all methods of self.cap to self

	def __getattr__(self, name):
		"""
		        Forward any undefined attribute access to self.cap
		This will automatically delegate any method calls not defined in this class
		to the underlying cv2.VideoCapture object.
		"""
		return getattr(self.cap, name)

	def get_capture(self):
		return self.cap

	def get_frame_size(self):
		width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# fps = self.cap.get(cv2.CAP_PROP_FPS)
		return width, height, None


class GazeboConnection(ArdupilotConnection):
	def __init__(
		self,
		connection_string,
		model_name="iris_with_gimbal",
		camera_link="pitch_link",
		world="our_runway",
		camera_port=5600,
		logger=None,
	):
		super().__init__(
			connection_string,
			logger=lambda *args: logger(*args)
			if logger
			else print("[GazeboConnection]", *args),
		)
		self.model_name = model_name
		self.camera_link = camera_link
		self.world = world
		self.cap = None
		self.camera_port = camera_port

	def enable_streaming(self) -> bool:
		"""
		Enable streaming for the camera in the Gazebo simulation.
		"""
		world = self.world
		model_name = self.model_name
		camera_link = self.camera_link

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
			self.log("The current gimbal topic is", command[3])
			time.sleep(0.5)
			self.cap = GazeboVideoCapture(camera_port=self.camera_port)

			self.log("🦾 Gazebo gimbal streaming enabled...", result.stdout)

			return True
		except subprocess.CalledProcessError as e:
			self.log("Error:", e.stderr)
			self.log("The current topic is", command[2])
			return False
		except Exception as e:
			self.log("Error:", e)

	def point_gimbal_downward(self, topic="/gimbal/cmd_tilt", angle=0) -> bool:
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
			print(
				"[CAMERA] Gimbal pointed to angle:", angle, "degrees. On topic:", topic
			)
			return True
		except subprocess.CalledProcessError as e:
			print("Error:", e.stderr)
			return False


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

	Args
	    master: MAVLink Connection (pymavlink instance).
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


if __name__ == "__main__":
	# Example usage
	# Example usage
	connection = GazeboConnection(
		connection_string="udp:127.0.0.1:14550",
		world="delivery_runway",
		camera_link="tilt_link",
		model_name="iris_with_stationary_gimbal",
	)

	connection.arm()
	connection.takeoff(10)
	if not connection.enable_streaming():
		exit(1)

	connection.point_gimbal_downward()
	connection._set_mode("AUTO")

	_lat, _lon, _ = connection.get_current_gps_location()

	connection.goto_waypoint(_lat + 0.00001, _lon + 0.00001, 3)

	camera = connection.cap.get_capture()

	while True:
		if connection.check_reposition_reached():
			connection.log("Waypoint reached!")
			break
		ret, frame = camera.read()
		if not ret:
			connection.log("Failed to capture frame.")
			break
		cv2.imshow("Gazebo Video Stream", frame)
		# Process window events and check for exit key (q)
		key = cv2.waitKey(30) & 0xFF  # 30ms delay between frames
		if key == ord("q"):
			connection.log("User exited video stream.")
			break
	connection.return_to_launch()
	connection.clear_mission()
	connection.master.close()
	connection.log("Connection closed.")
	connection.log("Ardupilot connection example completed.")
