import os
import time
import logging
from .gps.ekf import GeoFilter
from .mavlink.ardupilot import ArdupilotConnection, Waypoint
from .mavlink.gz import enable_streaming, point_gimbal_downward, GazeboVideoCapture
from .detection import yolo
import cv2
import numpy as np
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
    
    def update(self, frame, annotated_frame, curr_coords, detected_coords=None, center_pose=None):
        """Update the display with the current frames and information"""
        # Create a combined display with raw and annotated frames side by side
        h, w = frame.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = frame
        combined[:, w:] = annotated_frame
        
        # Add dividing line
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add labels for each view
        cv2.putText(combined, "Raw Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Annotated Feed", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display flight mode with appropriate color

        # Orange for GUIDED, Blue for STABILIZE, Green for AUTO
        mode_color = (0, 165, 255) if self.current_mode == "GUIDED" else (80, 80, 255) if self.current_mode == "STABILIZE" else (0, 255, 0)  
        # mode_color = (0, 165, 255) if self.current_mode == "GUIDED" else (0, 255, 0)  # Orange for GUIDED, Green for AUTO
        cv2.putText(combined, f"MODE: {self.current_mode}", 
                  (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Current coordinates
        cv2.putText(combined, f"Current GPS: {curr_coords[0]:.8f}, {curr_coords[1]:.8f}, {curr_coords[2]:.1f}m", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display detected helipad info if available
        if detected_coords and center_pose:
            self.goto_coords = detected_coords
            cv2.putText(combined, f"Helipad GPS: {detected_coords[0]:.8f}, {detected_coords[1]:.8f}", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(combined, f"Center Offset: ({center_pose[0]:.1f}, {center_pose[1]:.1f})", 
                       (w+10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            cv2.putText(combined, "No helipad detected", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the combined display
        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(5) & 0xFF
        return key
    
    def close(self):
        """Clean up resources"""
        cv2.destroyWindow(self.window_name)


def process_frame(camera, estimator, connection, geo_filter, display):
    """Process a single frame from the camera and update the display"""
    ret, frame = camera.read()
    if not ret:
        connection.log("❌ Failed to capture frame.")
        return False
    
    curr_coords = connection.get_current_gps_location()
    detections = estimator.detect(frame)
    annotated_frame, coords, center_pose = estimator.process_frame(
        frame=frame,
        detections=detections,
        current_lat=curr_coords[0],
        current_lon=curr_coords[1],
        altitude=curr_coords[2],
        object_class=HELIPAD_CLASS,
        threshold=DETECTION_THRESHOLD,
    )
    
    if coords:
        filtered_coords = (*coords, 0) #geo_filter.compute_gps((*coords, 0))
        display.update(frame, annotated_frame, curr_coords, filtered_coords, center_pose)
    else:
        display.update(frame, annotated_frame, curr_coords)
    
    return True


def handle_waypoint_reached(seq, completed, connection, camera, estimator, geo_filter, display):
    if not completed and seq == handle_waypoint_reached.prev_seq + 1:
        connection.log(f"Reached waypoint {seq}. Switching to GUIDED mode.")
        connection._set_mode("GUIDED")
        display.set_mode("STABILIZE")
        
        # Stabilize the stream
        connection.log(f"Stabilizing stream for waypoint {seq}...")
        for _ in range(100):
            process_frame(camera, estimator, connection, geo_filter, display)
            time.sleep(0.05)
            
        display.set_mode("GUIDED")
        connection.log("Stabilization complete. Proceeding with repositioning.")
        if display.goto_coords:
            connection.log(f"Repositioning to {display.goto_coords} in GUIDED mode.")
            lat, lon, _ = display.goto_coords
            coords = connection.get_current_gps_location()
            connection.log(f"Difference from current location: {lat - coords[0]:.6f}, {lon - coords[1]:.6f}")
            connection.goto_waypointv2(lat, lon, 20)
            
            # Wait for repositioning to complete
            while not connection.check_reposition_reached(lat, lon, 20):
                process_frame(camera, estimator, connection, geo_filter, display)
                time.sleep(0.05)

        connection.log(f"Repositioning complete. Returning to AUTO mode.")
        time.sleep(5)
        connection._set_mode("AUTO")
        display.set_mode("AUTO")
        connection.log(f"Returned to AUTO after repositioning.")
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
    weights_path = os.path.join(os.path.dirname(__file__), "detection/best.pt")
    estimator = yolo.YoloObjectTracker(weights_path, height, width, GIMBAL_FOV_DEG)
    geo_filter = GeoFilter()
    display = StreamDisplay()
    display.set_mode("AUTO")

    lat, lon, alt = connection.get_current_gps_location()
    connection.takeoff(10)

    mission_coords = [
        [lat + 0.00001, lon + 0.00001, 20],
        [lat - 0.00002, lon - 0.00002, 20],
        [lat + 0.00002, lon + 0.00002, 20],
        [lat - 0.00001, lon + 0.00001, 20],
        [lat + 0.00002, lon + 0.00001, 20],
        [lat + 0.00002, lon - 0.00002, 20],
        [lat, lon, alt + 10]  # Return to home
    ]

    try:
        mission = [Waypoint(lat, lon, alt, hold=0) for lat, lon, alt in mission_coords]
        connection.upload_mission(mission)
        connection.start_mission()

        def waypoint_callback(seq, completed):
            handle_waypoint_reached(seq, completed, connection, camera, estimator, geo_filter, display)

        while not connection.monitor_mission_progress(waypoint_callback):
            process_frame(camera, estimator, connection, geo_filter, display)
            time.sleep(0.05)

    except Exception as e:
        connection.log(f"❌ Error: {e}")
    finally:
        connection.clear_mission()
        connection.return_to_launch()
        connection.close()
        display.close()
        connection.log("✅ Mission complete and connection closed.")


if __name__ == "__main__":
    main()