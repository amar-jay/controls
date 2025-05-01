import cv2
import time
import mavlink.gz as gz
import detection.yolo as yolo
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
    "rtph264depay ! "
    "h264parse ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)

# done = gz.point_gimbal_downward(
#     topic="/gimbal/cmd_tilt",
#     angle=1.57)
# if not done:
#     print("❌ Failed to point gimbal downward.")
#     exit(1)

done = gz.enable_streaming(
    world="delivery_runway",
    model_name="iris_with_stationary_gimbal",
    camera_link="tilt_link")
if not done:
    print("❌ Failed to enable streaming.")
    exit(1)




camera = gz.GazeboVideoCapture()

width, height, _ = camera.get_frame_size()
cap = camera.get_capture()

estimator = yolo.YoloObjectTracker(
    "best.pt", 
    frame_height=height,
    frame_width=width,
    fov_deg=2, # field of view in degrees, got from the gimbal model.sdf file TODO: check if its in degrees/radians
    )

print(f"[CAMERA]  → width: {width}, height: {height}")

if not cap.isOpened():
    print("Failed to open stream! Check sender or pipeline.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Is the stream active?")
        break

    results = estimator.detect(frame)
    coords, center_pose, annotated_frame = estimator.process_frame(results, 0, 0, 10, object_class="helipad")
    # print(f"[YOLO]  →   detected {len(results)} objects.")
    if coords is None or center_pose is None:
        cv2.putText(annotated_frame, "No target detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Stream", frame)
        cv2.imshow("Annotated Stream", annotated_frame)
        continue


    target_lat, target_lon = coords
    cx, cy = center_pose
    text_latlon = f"helipad Lat: {target_lat:.6f}, Lon: {target_lon:.6f}"
    text_center = f"helipad Center: ({cx:.3f}, {cy:.3f})"

    cv2.putText(annotated_frame, text_center, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, text_latlon, (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Stream", frame)
    cv2.imshow("Annotated Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(.3)
cap.release()
cv2.destroyAllWindows()

