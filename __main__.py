import os
import cv2
from pymavlink import mavutil   
import math
import time
import detection.yolo as yolo
from mavlink import gz
import logging
from gps.ekf import GeoFilter

logging.getLogger("ultralytics").setLevel(logging.WARNING)


CONNECTION_STR = "udp:127.0.0.1:14550"
SAVE_DIR = "captures"  # directory to save images
PER_CAPTURE = .3  # time in seconds to wait between captures

master = mavutil.mavlink_connection(CONNECTION_STR)
master.wait_heartbeat()
print(
    f"[MAVLink] Heartbeat from system {master.target_system}, component {master.target_component}"
)

takeoff_altitude = 10.0

gz.arm(master)

location = gz.get_current_gps_location(master)
if location is None:
    print("❌ Failed to get current GPS location.")
    exit(1)

lat, lon, alt = location
gz.update_alt_compensation(alt)
alt_compensation = gz.get_base_alt()
print(f"📍Initial Current location → lat: {lat}, lon: {lon}, actual alt: {alt} compensated alt: {alt_compensation}")

gz.takeoff(master, target_altitude=takeoff_altitude)
print(f"📍 Takeoff {takeoff_altitude} m location → lat: {lat}, lon: {lon}, actual alt: {takeoff_altitude}")

done = gz.enable_streaming(
    world="delivery_runway",
    model_name="iris_with_stationary_gimbal",
    camera_link="tilt_link")
if not done:
    print("❌ Failed to enable streaming.")
    exit(1)

# os.makedirs(SAVE_DIR, exist_ok=True)

waypoints = [
    # (lat + 0.0001, lon + 0.0001, 7),
    (lat, lon, 10),
    (lat, lon, 13),
    (lat, lon, 15),
    (lat, lon, 10),
    (lat, lon, 13),
    # (lat + 0.0001, lon + 0.0001, -5),
    # (lat, lon, 10),
    # (lat + 0.0001, lon - 0.0001, 5),
    # (lat, lon, 10),
    # (lat - 0.0001, lon - 0.0001, 5),
    # (lat, lon, 10),
    # (lat - 0.0001, lon + 0.0001, 5),
    # (lat, lon, 10),
]


print("📸 Starting video stream...")
camera = gz.GazeboVideoCapture()
width, height, _ = camera.get_frame_size()
cap = camera.get_capture()

print(f"[CAMERA]  → width: {width}, height: {height}")

current_dir = os.path.dirname(os.path.abspath(__file__))
estimator = yolo.YoloObjectTracker(
    current_dir + "/detection/best.pt", 
    frame_height=height,
    frame_width=width,
    fov_deg=2, # field of view in degrees, got from the gimbal model.sdf file TODO: check if its in degrees/radians
    )

_filter = GeoFilter()

for idx, (lat, lon, alt) in enumerate(waypoints):
    # fly to next waypoint
    gz.goto_waypoint(master, lat, lon, alt, timeout=100, alt_compensation=alt_compensation)
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
        coords, center_pose, annotated_frame = estimator.process_frame(results, current_lat, current_lon, alt, object_class="helipad", threshold=0.45)
        # print(f"[YOLO]  →   detected {len(results)} objects.")
        # if coords is not None:
        #     # print(f"[YOLO]  →   detected helipad at {coords} with center {center_pose}")
        # else:
        #     print(f"[YOLO]  →   no helipad detected.")

        text_latlon_base = f"current Lat: {current_lat:.6f}, Lon: {current_lon:.6f}"
        if coords is None or center_pose is None:
            cv2.putText(annotated_frame, text_latlon_base, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "No target detected", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Annotated Stream", annotated_frame)
            cv2.waitKey(1) # NOTE: this is important to show the frame, otherwise it will block the stream
            continue

        target_lat, target_lon = coords


        target_lat, target_lon, alt = _filter.compute_gps((target_lat, target_lon, 0))

        cx, cy = center_pose
        text_latlon_base = f"current Lat: {current_lat:.6f}, Lon: {current_lon:.6f}"
        text_latlon = f"helipad Lat: {target_lat:.6f}, Lon: {target_lon:.6f}"
        text_center = f"helipad Center: ({cx}, {cy})"

        cv2.putText(annotated_frame, text_latlon_base, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_frame, text_latlon, (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, text_center, (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # wait for a bit before capturing the next frame
        cv2.imshow("Annotated Stream", annotated_frame)
        cv2.waitKey(1) # NOTE: this is important to show the frame, otherwise it will block the stream


        reached = gz.check_waypoint_reached()
        if reached:
            print(f"  ✅ reached waypoint {idx} at lat: {lat}, lon: {lon}, alt: {alt}")
            # if gz.is_pickup_confirmation_received():
            #     print("  ✅ pickup confirmation received.")
            break
        if reached is None:
            print(
                f"  ❌ failed to reach waypoint {idx} in time going to next waypoint, skipping slice."
            )
            break



        time.sleep(PER_CAPTURE)
    alt_compensation = gz.get_current_gps_location(master)[2]
    print("[WAITING].....") #TODO: with emoji
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break   time.sleep(10)

cap.release()
cv2.destroyAllWindows()
