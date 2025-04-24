import cv2
import gz

pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
    "rtph264depay ! "
    "h264parse ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)

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



cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("Failed to open stream! Check sender or pipeline.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Is the stream active?")
        break

    cv2.imshow("Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

