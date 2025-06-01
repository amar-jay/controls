import cv2
import numpy as np
from ultralytics import YOLO  # Requires `pip install ultralytics`
import math
from typing import Tuple


# get current working directory
class YoloObjectTracker:
	def __init__(
		self,
		hfov_rad,
		model_path="detection/best.pt",
		frame_width=640,
		frame_height=640,
		log=None,
	):
		self.model = YOLO(model_path)
		self.hfov_rad = hfov_rad
		self.frame_width = frame_width
		self.frame_height = frame_height
		self.log = print if log is None else log

	def detect_helipad(
		self,
		image: np.ndarray,
		confidence_threshold: float = 0.5,
		object_class="helipad",
	):
		# Run YOLO detection
		results = self.model(image, conf=confidence_threshold)

		# For this example, we'll look for any circular/landing pad-like objects
		# In practice, you'd train YOLO specifically on helipad data
		for result in results:
			boxes = result.boxes
			if boxes is not None:
				for box in boxes:
					if self.model.names[int(box.cls[0])] == object_class:
						x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
						confidence = box.conf[0].cpu().numpy()
						class_id = int(box.cls[0].cpu().numpy())

						# Calculate center point of detected object
						center_x = int((x1 + x2) / 2)
						center_y = int((y1 + y2) / 2)
						width = x2 - x1
						height = y2 - y1

						return {
							"center_pixel": (center_x, center_y),
							"bbox": (int(x1), int(y1), int(x2), int(y2)),
							"confidence": float(confidence),
							"class_id": class_id,
							"size": (width, height),
						}

		return None

	def pixel_to_gps(
		self,
		pixel_coords: Tuple[int, int],
		drone_gps: Tuple[float, float, float],
		drone_attitude: Tuple[float, float, float],  # (roll, pitch, yaw) in radians
	):
		"""
		Estimate GPS location of target seen in camera view.

		Parameters:
		- pixel_coords: (x, y) pixel coordinates in the image
		- drone_gps: (latitude, longitude, altitude) of the drone
		- drone_pose: (roll, pitch, yaw) of the drone in radians

		Returns:
		- (target_lat, target_lon): estimated GPS coordinates
		"""

		drone_lat, drone_lon, drone_alt = drone_gps
		roll, pitch, yaw = drone_attitude
		# 1. Camera intrinsics
		fx = (self.frame_width / 2) / math.tan(self.hfov_rad / 2)
		fy = fx * (self.frame_width / self.frame_height)  # Assuming square pixels
		cx, cy = self.frame_width / 2, self.frame_height / 2

		# 2. Convert pixel to normalized camera coordinates
		x = (pixel_coords[0] - cx) / fx
		y = (pixel_coords[1] - cy) / fy
		dir_cam = np.array([x, y, -1.0])
		dir_cam = dir_cam / np.linalg.norm(dir_cam)

		# 3. Rotation from camera to world
		def rotation_matrix(roll, pitch, yaw):
			R_x = np.array(
				[
					[1, 0, 0],
					[0, math.cos(roll), -math.sin(roll)],
					[0, math.sin(roll), math.cos(roll)],
				]
			)
			R_y = np.array(
				[
					[math.cos(pitch), 0, math.sin(pitch)],
					[0, 1, 0],
					[-math.sin(pitch), 0, math.cos(pitch)],
				]
			)
			R_z = np.array(
				[
					[math.cos(yaw), -math.sin(yaw), 0],
					[math.sin(yaw), math.cos(yaw), 0],
					[0, 0, 1],
				]
			)
			return R_z @ R_y @ R_x

		R = rotation_matrix(roll, pitch, yaw)
		dir_world = R @ dir_cam
		if dir_world[2] >= 0:
			self.log(
				"⚠️ Ray does not point downwards, cannot compute GPS. Check drone attitude."
			)
			return None  # Or handle this case differently

		# 4. Intersect ray with ground (flat)
		t = drone_alt / -dir_world[2]
		offset_ned = t * dir_world  # [north, east, down]

		north = offset_ned[0]
		east = offset_ned[1]

		# 5. Convert NED offset to GPS
		def offset_gps(lat, lon, dn, de):
			dLat = dn / 6378137.0
			dLon = de / (6378137.0 * math.cos(math.radians(lat)))
			return lat + math.degrees(dLat), lon + math.degrees(dLon)

		target_lat, target_lon = offset_gps(drone_lat, drone_lon, north, east)

		return target_lat, target_lon

	def _haversine_distance(self, lat1, lon1, lat2, lon2):
		"""
		Calculate the great-circle distance between two points
		on the Earth's surface given their latitude and longitude in decimal degrees.
		"""
		# Convert decimal degrees to radians
		lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

		# Haversine formula
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = (
			math.sin(dlat / 2) ** 2
			+ math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
		)
		c = 2 * math.asin(math.sqrt(a))

		# Radius of Earth in kilometers (use 3956 for miles)
		r = 6371

		return c * r

	def gps_loss(self, pred_lat, pred_lon, gt_lat, gt_lon):
		return self._haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

	def __pixel_to_gps(
		self,
		pixel_coords: Tuple[int, int],
		drone_gps: Tuple[float, float, float],
	) -> Tuple[float, float]:
		"""
		Convert pixel coordinates in image to GPS coordinates on the ground
		assuming camera facing straight down (nadir).

		Args:
		    pixel_coords: (x, y) pixel coordinates in the image
		    drone_gps: (latitude, longitude, altitude_meters)
		    frame_width: image width in pixels
		    frame_height: image height in pixels
		    diagonal_fov_rad: camera diagonal FOV in radians

		Returns:
		    (latitude, longitude) of target point on ground
		"""
		px_x, px_y = pixel_coords
		lat, lon, altitude = drone_gps

		frame_width = self.frame_width
		frame_height = self.frame_height
		# Compute aspect ratio
		aspect_ratio = frame_width / frame_height

		horizontal_fov_rad = self.hfov_rad
		vertical_fov_rad = self.hfov_rad * aspect_ratio

		# Normalize pixel coords to [-1, 1] with pixel centers as reference
		norm_x = (px_x - (frame_width - 1) / 2) / ((frame_width - 1) / 2)
		norm_y = (px_y - (frame_height - 1) / 2) / ((frame_height - 1) / 2)

		# Calculate ground coverage (meters) at given altitude
		ground_width_m = 2 * altitude * math.tan(horizontal_fov_rad / 2)
		ground_height_m = 2 * altitude * math.tan(vertical_fov_rad / 2)

		# Compute offsets from drone GPS position in meters
		ground_x_m = norm_x * (ground_width_m / 2)  # East-West offset
		ground_y_m = norm_y * (ground_height_m / 2)  # North-South offset

		# Earth radius varies with latitude; use mean Earth radius for precision
		earth_radius_m = 6378137  # meters (WGS-84 standard)

		# Convert meter offsets to degrees
		delta_lat = (ground_y_m / earth_radius_m) * (180 / math.pi)
		delta_lon = (ground_x_m / (earth_radius_m * math.cos(math.radians(lat)))) * (
			180 / math.pi
		)

		# Note: y increases downwards in image, so subtract delta_lat
		helipad_lat = lat - delta_lat
		helipad_lon = lon + delta_lon

		# frame dim
		print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
		print(f"Pixel coordinates: {pixel_coords}")
		# fov
		print(f"FOV: {self.hfov_rad}")
		print(f"Normalized coordinates: ({norm_x:.2f}, {norm_y:.2f})")
		# detected helapad GPS
		print(f"Predicted Helipad GPS: ({helipad_lat:.6f}, {helipad_lon:.6f})")
		print(f"Drone GPS: ({lat:.6f}, {lon:.6f}, {altitude:.2f})")
		print("-" * 40)

		return helipad_lat, helipad_lon

	def _pixel_to_gps(
		self, pixel_coords: Tuple[int, int], drone_gps: Tuple[int, int, int]
	) -> Tuple[float, float]:
		"""
		Convert pixel coordinates to GPS coordinates using FOV

		Args:
		    pixel_coords: (x, y) pixel coordinates in image

		Returns:
		    (latitude, longitude) GPS coordinates
		"""
		px_x, px_y = pixel_coords
		lat, lon, altitude = drone_gps

		# Convert pixel coordinates to normalized coordinates (-1 to 1)
		# Center of image is (0, 0)
		norm_x = (px_x - (self.frame_width - 1) / 2) / ((self.frame_width - 1) / 2)
		norm_y = (px_y - (self.frame_height - 1) / 2) / ((self.frame_height - 1) / 2)

		# Calculate ground coverage at current altitude using FOV
		# Ground width/height covered by the camera at current altitude
		horizontal_fov_rad = self.hfov_rad
		vertical_fov_rad = self.hfov_rad * (
			(self.frame_height) / (self.frame_width)
		)  # Maintain aspect ratio

		# Total ground coverage (width and height in meters)
		ground_width_total = 2 * altitude * math.tan(horizontal_fov_rad / 2)
		ground_height_total = 2 * altitude * math.tan(vertical_fov_rad / 2)

		# Distance from drone center to helipad on ground (in meters)
		ground_x = norm_x * (ground_width_total / 2)  # East-West offset
		ground_y = norm_y * (ground_height_total / 2)  # North-South offset

		# Convert ground distance to GPS coordinates
		# Approximate conversion (more accurate methods would use proper geodetic calculations)
		lat_per_meter = 1 / 111320.0  # Approximate meters per degree latitude
		lon_per_meter = 1 / (
			111320.0 * math.cos(math.radians(lat))
		)  # Longitude varies with latitude

		# Calculate helipad GPS coordinates
		# Note: In image coordinates, Y increases downward, but in GPS, latitude increases northward
		helipad_lat = lat - (
			ground_y * lat_per_meter
		)  # Negative because image y increases downward
		helipad_lon = lon + (ground_x * lon_per_meter)
		# frame dim
		print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
		print(f"Pixel coordinates: {pixel_coords}")
		# fov
		print(f"FOV: {self.hfov_rad}")
		print(f"Normalized coordinates: ({norm_x:.2f}, {norm_y:.2f})")
		# detected helapad GPS
		print(f"Predicted Helipad GPS: ({helipad_lat:.6f}, {helipad_lon:.6f})")
		print(f"Drone GPS: ({lat:.6f}, {lon:.6f}, {altitude:.2f})")

		return helipad_lat, helipad_lon

	def gps_to_altitude(
		self,
		pixel_coords: Tuple[int, int],
		drone_gps: Tuple[float, float, float],
		helipad_gps: Tuple[float, float],
	) -> float:
		"""
		Estimate the altitude of the drone given its GPS, a pixel coordinate in the image (e.g., where the helipad is),
		and the GPS location of the helipad.

		Args:
		    pixel_coords: (x, y) pixel coordinates where helipad appears in the image
		    drone_gps: (lat, lon, alt) GPS and altitude of the drone
		    helipad_gps: (lat, lon) GPS coordinates of the helipad

		Returns:
		    Estimated altitude (in meters)
		"""

		px_x, px_y = pixel_coords
		drone_lat, drone_lon, _ = drone_gps
		helipad_lat, helipad_lon = helipad_gps
		# actual helipad GPS
		# print(f"Actual Helipad GPS: ({helipad_lat:.8f}, {helipad_lon:.6f})")

		# Convert pixel to normalized image coordinates
		norm_x = (px_x - (self.frame_width - 1) / 2) / ((self.frame_width - 1) / 2)
		norm_y = (px_y - (self.frame_height - 1) / 2) / ((self.frame_height - 1) / 2)

		# Approximate lat/lon distance in meters
		delta_lat = drone_lat - helipad_lat  # Positive if helipad is south
		delta_lon = helipad_lon - drone_lon  # Positive if helipad is east

		lat_per_meter = 1 / 111320.0
		lon_per_meter = 1 / (111320.0 * math.cos(math.radians(drone_lat)))

		ground_y = delta_lat / lat_per_meter  # North-South offset in meters
		ground_x = delta_lon / lon_per_meter  # East-West offset in meters

		# Reconstruct total ground width and height (from normalized offsets and actual offsets)
		ground_width_total = 2 * ground_x / norm_x if norm_x != 0 else float("inf")
		ground_height_total = 2 * ground_y / norm_y if norm_y != 0 else float("inf")

		# FOV in radians
		horizontal_fov_rad = self.hfov_rad
		vertical_fov_rad = self.hfov_rad * (self.frame_height / self.frame_width)

		# Compute altitude based on either horizontal or vertical FOV (choose the one with less division noise)
		if abs(norm_x) > abs(norm_y):  # Use horizontal
			altitude = ground_width_total / (2 * math.tan(horizontal_fov_rad / 2))
		else:  # Use vertical
			altitude = ground_height_total / (2 * math.tan(vertical_fov_rad / 2))

		return altitude

	def process_frame(
		self,
		frame,
		drone_gps: Tuple[float, float, float],
		drone_attitude: Tuple[float, float, float],
		object_class="helipad",
		threshold=0.5,
	):
		detection = self.detect_helipad(frame, confidence_threshold=threshold)
		if detection is None:
			print("No helipad detected")
			return frame, None, None

		x1, y1, x2, y2 = detection["bbox"]
		conf = detection["confidence"]
		color = (100, 255, 0)

		annotated_frame = frame.copy()
		cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
		cv2.circle(annotated_frame, detection["center_pixel"], 8, (255, 0, 255), -1)
		cv2.putText(
			annotated_frame,
			f"{object_class}: {conf:.2f}",
			(x1, y1 - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			color,
			2,
		)

		lat, lon = self.pixel_to_gps(
			pixel_coords=detection["center_pixel"],
			drone_gps=drone_gps,
			drone_attitude=drone_attitude,
		)
		return annotated_frame, (lat, lon), detection["center_pixel"]


if __name__ == "__main__":
	import logging

	logging.getLogger("ultralytics").setLevel(logging.WARNING)

	input_video_path = "assets/input_video2.mp4"  # Path to your input video
	output_video_path = "assets/output.mp4"  # Path to your input video

	# Read the image using PIL and convert to numpy array
	# image_np = np.array(Image.open(random_test_image_path).convert("RGB"))
	cap = cv2.VideoCapture(input_video_path)
	if not cap.isOpened():
		print(f"❌ Error opening video file: {input_video_path}")
		exit(1)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	out = cv2.VideoWriter(
		output_video_path,
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)

	# Load YOLO model
	estimator = YoloObjectTracker(1, "detection/best.pt")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Run inference
		try:
			coords, center_pose, annotated_frame = estimator.process_frame(
				frame,
				(0, 0, 10),
				(0, 0, 0),  # Dummy GPS and attitude for testing
			)
			if coords is None or center_pose is None:
				cv2.putText(
					annotated_frame,
					"No target detected",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(0, 0, 255),
					2,
					cv2.LINE_AA,
				)
				out.write(annotated_frame)
				continue

			# Extract coordinates and center pose
			target_lat, target_lon = coords
			cx, cy = center_pose
			text_latlon = f"Lat: {target_lat:.6f}, Lon: {target_lon:.6f}"
			text_center = f"Center: ({cx}, {cy})"

			pos1 = (10, 30)
			pos2 = (10, 60)

			cv2.putText(
				annotated_frame,
				text_latlon,
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)
			cv2.putText(
				annotated_frame,
				text_center,
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)

			out.write(annotated_frame)

		except ValueError as e:
			print(f"❌ {e}")
			# If no target detected, just write the original frame
			out.write(frame)

	cap.release()
	out.release()
	print(f"✅ Output saved to {output_video_path}")
