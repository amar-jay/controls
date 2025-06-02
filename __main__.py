import logging
import os
import time

import cv2
import numpy as np

from src.controls.detection import yolo
from src.controls.gps.ekf import GeoFilter
from src.controls.mavlink.ardupilot import ArdupilotConnection, Waypoint
from src.controls.mavlink.gz import (  # , point_gimbal_downward
    GazeboVideoCapture,
    enable_streaming,
)

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Constants
GIMBAL_FOV_DEG = 0.85
HELIPAD_CLASS = "helipad"
DETECTION_THRESHOLD = 0.5


class StreamDisplay:
    def __init__(self, window_name="Drone Mission"):
        self.window_name = window_name
        self.goto_coords = None
        self.current_mode = "AUTO"  # Default mode
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def set_mode(self, mode):
        """Update the current flight mode"""
        self.current_mode = mode

    def update(
        self,
        frame,
        annotated_frame,
        curr_coords,
        detected_coords=None,
        center_pose=None,
        actual_coords=None,
        loss=None,
    ):
        """Update the display with the current frames and information"""
        # Create a combined display with raw and annotated frames side by side
        h, w = frame.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = annotated_frame
        combined[:, w:] = frame

        # Add dividing line
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)

        # Add labels for each view
        cv2.putText(
            combined,
            "Raw Feed",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            combined,
            "Annotated Feed",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Display flight mode with appropriate color

        # Orange for GUIDED, Blue for STABILIZE, Green for AUTO
        mode_color = (
            (0, 165, 255)
            if self.current_mode == "GUIDED"
            else (80, 80, 255) if self.current_mode == "STABILIZE" else (0, 255, 0)
        )
        # mode_color = (0, 165, 255) if self.current_mode == "GUIDED" else (0, 255, 0)  # Orange for GUIDED, Green for AUTO
        cv2.putText(
            combined,
            f"MODE: {self.current_mode}",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2,
        )

        # Current coordinates
        if actual_coords is not None:
            cv2.putText(
                combined,
                f"Actual    GPS: {actual_coords[0]:.8f}, {actual_coords[1]:.8f}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Display detected helipad info if available
        if detected_coords and center_pose:
            self.goto_coords = detected_coords
            cv2.putText(
                combined,
                f"Predicted GPS: {detected_coords[0]:.8f}, {detected_coords[1]:.8f}",
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2,
            )
            cv2.putText(
                combined,
                f"Center Offset: ({center_pose[0]:.1f}, {center_pose[1]:.1f})",
                (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
            if actual_coords:
                cv2.putText(
                    combined,
                    f"Actual distance: {(curr_coords[0] - actual_coords[0]):.8f}, {(curr_coords[1] - actual_coords[1]):.8f}",
                    (w + 10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    combined,
                    f"Predicted distance: {(curr_coords[0] - detected_coords[0]):.8f}, {(curr_coords[1] - detected_coords[1]):.8f}",
                    (w + 10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # calcuate percentage error
                loss = loss if loss is not None else 0
                cv2.putText(
                    combined,
                    f"Predicted Difference: {(actual_coords[0] - detected_coords[0]):.8f}, {(actual_coords[1] - detected_coords[1]):.8f}| {loss:.2f}m",
                    (w + 10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 165, 0),
                    2,
                )
        else:
            cv2.putText(
                combined,
                "No helipad detected",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Show the combined display
        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(5) & 0xFF
        return key

    def close(self):
        """Clean up resources"""
        cv2.destroyWindow(self.window_name)


def process_frame(
    camera,
    estimator: yolo.YoloObjectTracker,
    connection: ArdupilotConnection,
    geo_filter: GeoFilter,
    display: StreamDisplay,
    actual_coords=None,
):
    """Process a single frame from the camera and update the display"""
    ret, frame = camera.read()
    if not ret:
        connection.log("❌ Failed to capture frame.")
        return False

    curr_gps = connection.get_current_gps_location()
    curr_attitude = connection.get_current_attitude()
    # detections = estimator.detect(frame)
    annotated_frame, detected_coords, center_pose = estimator.process_frame(
        frame=frame,
        drone_gps=curr_gps,
        drone_attitude=curr_attitude,
        object_class=HELIPAD_CLASS,
        threshold=DETECTION_THRESHOLD,
    )

    if detected_coords:
        detected_coords = (*detected_coords, 0)
        loss = estimator.gps_loss(
            detected_coords[0], detected_coords[1], curr_gps[0], curr_gps[1]
        )
        # detected_coords = geo_filter.compute_gps(detected_coords)
        # if actual_coords is not None:
        # 	altitude = estimator.gps_to_altitude(
        # 		drone_gps=curr_coords,
        # 		helipad_gps=actual_coords[:2],
        # 		pixel_coords=center_pose,
        # 	)
        # 	print(f"Estimated altitude: {altitude:.2f}m")
        if process_frame.writer is not None and actual_coords is not None:
            # headers = ["pixel", "gps_true", "drone_gps", "drone_attitude"]
            process_frame.writer(
                _pixel=center_pose,
                _gps_true=actual_coords,
                drone_gps=detected_coords,
                drone_att=curr_attitude,
            )
        display.update(
            frame=frame,
            annotated_frame=annotated_frame,
            curr_coords=curr_gps,
            detected_coords=detected_coords,
            center_pose=center_pose,
            actual_coords=actual_coords,
            loss=loss,
        )
    else:
        display.update(
            frame=frame,
            annotated_frame=annotated_frame,
            curr_coords=curr_gps,
            detected_coords=detected_coords,
            center_pose=center_pose,
            actual_coords=actual_coords,
        )

    return True


def handle_waypoint_reached(
    seq,
    completed,
    connection,
    camera,
    estimator,
    geo_filter,
    display: StreamDisplay,
):
    if not completed and seq == handle_waypoint_reached.prev_seq + 1:
        connection.log(f"Reached waypoint {seq}. Switching to GUIDED mode.")
        connection._set_mode("GUIDED")
        display.set_mode("STABILIZE")

        # Stabilize the stream
        connection.log(f"Stabilizing stream for waypoint {seq}...")
        for _ in range(100):
            process_frame(
                camera,
                estimator,
                connection,
                geo_filter,
                display,
                handle_waypoint_reached.actual_coords,
            )
            time.sleep(0.05)

        display.set_mode("GUIDED")
        connection.log("Stabilization complete. Proceeding with repositioning.")
        if display.goto_coords:
            connection.log(f"Repositioning to {display.goto_coords} in GUIDED mode.")
            lat, lon, _ = display.goto_coords
            coords = connection.get_current_gps_location()
            connection.log(
                f"Difference from current location: {lat - coords[0]:.6f}, {lon - coords[1]:.6f}"
            )
            connection.goto_waypointv2(lat, lon, 5)

            # Wait for repositioning to complete
            # while not connection.check_reposition_reached(lat, lon, 20):
            # 	process_frame(camera, estimator, connection, geo_filter, display)
            # 	time.sleep(0.05)

        connection.log("Repositioning complete. Returning to AUTO mode.")
        time.sleep(5)
        connection._set_mode("AUTO")
        display.set_mode("AUTO")
        connection.log("Returned to AUTO after repositioning.")
        handle_waypoint_reached.prev_seq = seq
    # elif not completed:
    #     connection.log(f"Mission waypoint {seq} in session.")


handle_waypoint_reached.prev_seq = 1


def main():
    # SETUP
    connection = ArdupilotConnection("udp:127.0.0.1:14550")
    connection._set_mode("AUTO")
    connection.arm()
    connection.master.motors_armed_wait()
    enable_streaming()
    # point_gimbal_downward()

    # OBJECT TRACKER
    camera = GazeboVideoCapture()
    width, height, _ = camera.get_frame_size()
    print(f"Camera resolution: {width}x{height}")
    weights_path = os.path.join(os.path.dirname(__file__), "detection/best.pt")
    estimator = yolo.YoloObjectTracker(
        model_path=weights_path,
        hfov_rad=GIMBAL_FOV_DEG,
        frame_height=height,
        frame_width=width,
        log=connection.log,
    )

    geo_filter = GeoFilter()
    display = StreamDisplay()
    display.set_mode("AUTO")

    lat, lon, alt = connection.get_current_gps_location()
    connection.takeoff(10)

    mission_coords = [
        [lat + 0.00003, lon + 0.00003, 30],
        [lat - 0.00004, lon - 0.00002, 20],
        [lat, lon + 0.000005, 10],
        [lat - 0.00001, lon - 0.00001, 7.5],
        [lat + 0.000009, lon + 0.00001, 5],
        [lat + 0.000006, lon - 0.000000, 3],
        [lat, lon, alt + 10],  # Return to home
    ]

    writer, close_dset = estimator.build_dataset("dataset.csv")
    process_frame.writer = writer

    try:
        mission = [Waypoint(lat, lon, alt, hold=5) for lat, lon, alt in mission_coords]
        connection.upload_mission(mission)
        connection.start_mission()

        handle_waypoint_reached.actual_coords = (lat, lon, alt)

        def waypoint_callback(seq, completed):
            handle_waypoint_reached(
                seq,
                completed,
                connection,
                camera,
                estimator,
                geo_filter,
                display,
            )

        while not connection.monitor_mission_progress(waypoint_callback):
            process_frame(camera, estimator, connection, geo_filter, display)
            time.sleep(0.05)

    except Exception as e:
        connection.log(f"❌ Error: {e}")
    finally:
        close_dset()
        connection.clear_mission()
        connection.return_to_launch()
        connection.close()
        display.close()
        connection.log("✅ Mission complete and connection closed.")


if __name__ == "__main__":
    main()
