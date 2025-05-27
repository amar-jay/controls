import cv2
import numpy as np
from ultralytics import YOLO  # Requires `pip install ultralytics`


# get current working directory
class YoloObjectTracker:
	def __init__(
		self,
		model_path="detection/best.pt",
		fov_deg=1,
		frame_width=640,
		frame_height=640,
	):
		self.model = YOLO(model_path)
		self.fov_deg = fov_deg
		self.frame_width = frame_width
		self.frame_height = frame_height

	def detect(self, frame):
		results = self.model(frame)  # TODO: add some confidence threshold
		# results = self.model.predict(frame, conf=0.5)  # Uncomment if using older YOLOv8
		return results[0]  # single frame

	def get_pixel_offset(self, cx, cy):
		dx = cx - self.frame_width / 2
		dy = cy - self.frame_height / 2
		return dx, dy

	def pixel_to_angle(self, dx, dy):
		"""Convert pixel offset to angular offset."""
		fx = self.fov_deg / self.frame_width
		fy = self.fov_deg / self.frame_height
		return dx * fx, dy * fy

	def estimate_gps_offset(self, angle_dx, angle_dy, altitude):
		"""Estimate how far the object is from the center in meters, assuming drone is fixed."""
		# Simple trig, assuming nadir (straight-down) view
		offset_x = np.tan(np.radians(angle_dx)) * altitude
		offset_y = np.tan(np.radians(angle_dy)) * altitude
		return offset_x, offset_y  # meters in X/Y plane

	def meters_to_gps(self, current_lat, current_lon, dx, dy):
		"""Convert x/y offsets in meters to latitude and longitude offset."""
		# Approximate conversion assuming small angles and distance
		earth_radius = 6378137  # in meters

		dlat = dy / earth_radius
		dlon = dx / (earth_radius * np.cos(np.pi * current_lat / 180))

		new_lat = current_lat + (dlat * 180 / np.pi)
		new_lon = current_lon + (dlon * 180 / np.pi)

		return new_lat, new_lon
	
	def plot(self, frame, detections, object_class, threshold, color=(100, 255, 0)):
		if not hasattr(detections, "boxes"):
			return frame, None

		boxes = detections.boxes[detections.boxes.conf >= threshold]
		for box in boxes:
			if self.model.names[int(box.cls[0])] == object_class:
				conf = box.conf.cpu().numpy()[0]
				x1, y1, x2, y2 = map(int, box.xyxy[0])
				center = ((x1 + x2) // 2, (y1 + y2) // 2)

				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
				cv2.circle(frame, center, 4, (0, 0, 255), -1)
				cv2.putText(frame, f"{object_class}: {conf:.2f}", (x1, y1 - 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
				return frame, center
				
		return frame, None


	def process_frame(
		self,
		frame,
		detections,
		current_lat,
		current_lon,
		altitude,
		object_class="helipad",
		threshold=0.5,
	):
		annotated_frame, best_center = self.plot(
			frame, detections, object_class, threshold
		)
		if best_center is None:
			return annotated_frame, None, None
		offset = self.get_pixel_offset(*best_center)
		angle_dx, angle_dy = self.pixel_to_angle(*offset)
		offset_x, offset_y = self.estimate_gps_offset(angle_dx, angle_dy, altitude)
		target_gps = self.meters_to_gps(current_lat, current_lon, offset_x, offset_y)
		return annotated_frame, target_gps, best_center






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
	estimator = YoloObjectTracker("detection/best.pt")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Run inference
		results = estimator.detect(frame)
		try:
			coords, center_pose, annotated_frame = estimator.process_frame(frame, results, 0, 0, 10)
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

	# Load the original image
	# image = cv2.imread(random_test_image_path)

	# res = results.plot()
	# cv2.imwrite("output.jpg", annotated_frame)

	# # Draw boxes and labels
	# for box in results.boxes:
	#     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
	#     conf = box.conf[0].item()
	#     cls = int(box.cls[0].item())
	#     label = estimator.model.names[cls]

	#     print(f"→ {label} with {conf:.2%} confidence")
	#     # Draw rectangle and label
	#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
	#     cv2.putText(
	#         image,
	#         f"{label} {conf:.2f}",
	#         (x1, y1 - 10),
	#         cv2.FONT_HERSHEY_SIMPLEX,
	#         0.6,
	#         (0, 255, 0),
	#         2,
	#     )

	# # Save the annotated image
	# cv2.imwrite("output.jpg", image)
	# print(f"Annotated image saved to output.jpg")

	# # Create tracker and run detection
	# estimator = YoloObjectTracker("detection/best.pt")
	# result_frame = estimator.detect(image_np)  # detect expects numpy frame
	# print(f"{result_frame=}")

	# Save the resulting image using PIL (not OpenCV)
	# result_image = Image.fromarray(result_frame)
	# output_path = "./output_detected.jpg"
	# result_image.save(output_path, format="JPEG")
	# print(f"Saved output to {output_path}")
	# random_test_image = random.choice(os.listdir("./frames"))
	# print("running inference on " + random_test_image)
