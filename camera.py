import os
import cv2
from pymavlink import mavutil   
import math
import time
import yolo
import gz
from gz import GazeboVideoCapture
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)


CONNECTION_STR = "udp:127.0.0.1:14550"
SAVE_DIR = "captures"  # directory to save images
PER_CAPTURE = 1  # time in seconds to wait between captures

master = mavutil.mavlink_connection(CONNECTION_STR)
master.wait_heartbeat()
print(
    f"[MAVLink] Heartbeat from system {master.target_system}, component {master.target_component}"
)


gz.arm_and_takeoff(master, target_altitude=10.0)

done = gz.point_gimbal_downward()
if not done:
    print("❌ Failed to point gimbal downward.")
    exit(1)

done = gz.enable_streaming(
    world="delivery_runway",
    model_name="iris_with_stationary_gimbal",
    camera_link="tilt_link")
if not done:
    print("❌ Failed to enable streaming.")
    exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)

location = gz.get_current_gps_location(master)
if location is None:
    print("❌ Failed to get current GPS location.")
    exit(1)

lat, lon, alt = location
print(f"📍 Current location → lat: {lat}, lon: {lon}, alt: {alt}")

waypoints = [
    (lat, lon, 15),
    (lat + 0.00001, lon + 0.0001, 5),
    (lat, lon, 10),
    (lat + 0.00001, lon - 0.00001, 5),
    (lat, lon, 10),
    (lat - 0.00001, lon - 0.00001, 5),
    (lat, lon, 10),
    (lat - 0.00001, lon + 0.00001, 5),
    (lat, lon, 10),
]


print("📸 Starting video stream...")
camera = GazeboVideoCapture()
width, height, fps = camera.get_frame_size()
cap = camera.get_capture()

print(f"[CAMERA]  → width: {width}, height: {height}, fps: {fps}")

estimator = yolo.YoloObjectTracker(
    "best.pt", 
    frame_height=height,
    frame_width=width,
    fov_deg=2, # field of view in degrees, got from the gimbal model.sdf file TODO: check if its in degrees/radians
    )
for idx, (lat, lon, alt) in enumerate(waypoints):
    # fly to next waypoint
    gz.goto_waypoint(master, lat, lon, alt, timeout=100)
    print(f"[MAVLINK]  → flying to waypoint {idx} at lat: {lat}, lon: {lon}, alt: {alt}")

    while True:
        # grab one frame
        ret, frame = cap.read()
        if not ret:
            print(f"[CAMERA]  →   ⚠️ failed to grab frame at waypoint {idx}, skipping slice.")
            continue

        cv2.imshow("Stream", frame)

        current_lat, current_lon, current_alt = gz.get_current_gps_location(master)
        if current_lat is None or current_lon is None:
            print(f"[MAVLINK]  →   ⚠️ failed to get current GPS location at waypoint {idx}, skipping slice.")
            continue

        results = estimator.detect(frame)
        coords, center_pose, annotated_frame = estimator.process_frame(results, current_lat, current_lon, alt, object_class="helipad")

        text_latlon_base = f"current Lat: {current_lat:.6f}, Lon: {current_lon:.6f}"
        if coords is None or center_pose is None:
            cv2.putText(annotated_frame, text_latlon_base, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "No target detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Annotated Stream", annotated_frame)
            continue

        target_lat, target_lon = coords
        cx, cy = center_pose
        text_latlon_base = f"current Lat: {current_lat:.6f}, Lon: {current_lon:.6f}"
        text_latlon = f"helipad Lat: {target_lat:.6f}, Lon: {target_lon:.6f}"
        text_center = f"helipad Center: ({cx}, {cy})"

        cv2.putText(annotated_frame, text_latlon_base, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_frame, text_latlon, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, text_center, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # wait for a bit before capturing the next frame
        cv2.imshow("Annotated Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        reached = gz.check_waypoint_reached()
        if reached:
            print(f"  ✅ reached waypoint {idx} at lat: {lat}, lon: {lon}, alt: {alt}")
            if gz.is_pickup_confirmation_received():
                print("  ✅ pickup confirmation received.")
                break
        if reached is None:
            print(
                f"  ❌ failed to reach waypoint {idx} in time going to next waypoint, skipping slice."
            )
            break


        time.sleep(PER_CAPTURE)

cap.release()
cv2.destroyAllWindows()
