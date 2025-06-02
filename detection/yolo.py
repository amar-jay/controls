import csv
import math
import os
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO  # Requires `pip install ultralytics`

# If IS_CV_TRAINING is true (case-insensitive), import torch
if os.getenv("IS_CV_TRAINING", "false").lower() == "true":
    import torch

    Tensor = torch.Tensor
else:

    class Tensor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Tensor is not available in this environment")


class Dataset:
    "Dataset type for Rotation Optimization"

    pixel: Tuple[int, int]
    gps_true: Tuple[float, float]
    drone_gps: Tuple[float, float, float]
    drone_att: Tuple[float, float, float]


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
        results = self.model(image, conf=confidence_threshold)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if self.model.names[int(box.cls[0])] == object_class:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

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
        drone_attitude: Tuple[float, float, float],
    ):
        drone_lat, drone_lon, drone_alt = drone_gps
        roll, pitch, yaw = drone_attitude

        fx = (self.frame_width / 2) / math.tan(self.hfov_rad / 2)
        fy = fx * (self.frame_width / self.frame_height)
        cx, cy = self.frame_width / 2, self.frame_height / 2

        x = (pixel_coords[0] - cx) / fx
        y = (pixel_coords[1] - cy) / fy
        dir_cam = torch.tensor([x, y, -1.0])
        dir_cam = dir_cam / torch.linalg.norm(dir_cam)

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

        # input parameters for training
        R = torch.tensor(R, dtype=torch.double, requires_grad=False)
        dir_cam = torch.tensor(dir_cam, dtype=torch.double, requires_grad=False)

        if self.adjustment is not None:
            # R_adj = self.rpy_to_rotation_matrix(
            # 	self.adjustment[0],
            # 	self.adjustment[1],
            # 	self.adjustment[2],
            # )
            R_adj = self.adjustment
            R = R_adj @ R
        dir_world = R @ dir_cam
        # if dir_world[2] > 0:
        # 	self.log(
        # 		"⚠️ Ray does not point downwards, cannot compute GPS. Check drone attitude."
        # 	)
        # 	return None

        t = drone_alt / -dir_world[2]
        offset_ned = t * dir_world
        north = offset_ned[0]
        east = offset_ned[1]

        def offset_gps(lat, lon, dn, de):
            dLat = dn / 6378137.0
            dLon = de / (6378137.0 * torch.cos(torch.deg2rad(lat)))
            return lat + torch.rad2deg(dLat), lon + torch.rad2deg(dLon)

        target_lat, target_lon = offset_gps(drone_lat, drone_lon, north, east)
        return target_lat, target_lon

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(roll), -torch.sin(roll)],
                [0, torch.sin(roll), torch.cos(roll)],
            ]
        )
        Ry = torch.tensor(
            [
                [torch.cos(pitch), 0, torch.sin(pitch)],
                [0, 1, 0],
                [-torch.sin(pitch), 0, torch.cos(pitch)],
            ]
        )
        Rz = torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), 0],
                [torch.sin(yaw), torch.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        return Rz @ Ry @ Rx

    def optimize_adjustment(self, dataset: list[Dataset], verbose=False):
        self.adjustment = torch.nn.Parameter(
            torch.tensor(
                [[0.01, 0.01, 0.01]] * 3, dtype=torch.double, requires_grad=True
            )
        )
        # use random initialization for adjustment. for a simple 2 la

        optimizer = torch.optim.Adam([self.adjustment], lr=0.01)

        for epoch in range(100):
            total_loss = 0.0
            for pixel, gps_true, drone_gps, drone_att in dataset:
                pred_latlon = self.pixel_to_gps(pixel, drone_gps, drone_att)
                if pred_latlon is None:
                    continue
                pred_lat, pred_lon = pred_latlon
                gt_lat, gt_lon, _ = gps_true
                loss = self.gps_loss(pred_lat, pred_lon, gt_lat, gt_lon)
                print(f"Loss: {loss.item():.2f}")
                total_loss = loss
                loss = torch.nn.functional.mse_loss(
                    total_loss, torch.tensor(0.0, dtype=torch.double)
                )
                optimizer.zero_grad()
                if verbose and epoch == 0:
                    verbose = False
                    print(f"Loss tensor: {loss}")
                    print(f"Requires grad: {loss.requires_grad}")
                    print(f"Grad function: {loss.grad_fn}")

                    # Print computational graph
                    try:
                        from torchviz import make_dot

                        dot = make_dot(loss, params={"adjustment": self.adjustment})
                        dot.render("computational_graph", format="png")
                        print("Computational graph saved as computational_graph.png")
                    except ImportError:
                        print(
                            "Install torchviz to visualize computational graph: pip install torchviz"
                        )
                total_loss.backward()
                optimizer.step()
                self.log(
                    f"Epoch {epoch}, Loss: {total_loss.item():.2f}, Adjustment: {self.adjustment.data.tolist()}"
                )

        return self.adjustment.data

    def build_dataset(self, dataset_path: str):
        # TODO: reimplemnt this. I know this is not the best way to do it, a temporary solution for now.

        # open a csv file and write the dataseto
        try:
            _f = open(dataset_path, mode="a", newline="", encoding="utf-8")
            headers = ["pixel", "gps_true", "drone_gps", "drone_attitude"]
            writer = csv.writer(_f)
            writer.writerow(headers)

            def close_():
                _f.close()

            def write_data(_pixel, _gps_true, drone_gps, drone_att):
                writer.writerow([_pixel, _gps_true, drone_gps, drone_att])
                return close_

            # write_data.close = close_
            return write_data, close_
        except FileNotFoundError:
            return None

    def _haversine_distance(self, pred_lat, pred_lon, gt_lat, gt_lon):
        R = 6371000
        phi1 = torch.deg2rad(gt_lat)
        phi2 = torch.deg2rad(pred_lat)
        dphi = torch.deg2rad(pred_lat - gt_lat)
        dlambda = torch.deg2rad(pred_lon - gt_lon)
        a = (
            torch.sin(dphi / 2) ** 2
            + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlambda / 2) ** 2
        )
        return 2 * R * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    def gps_loss(self, pred_lat, pred_lon, gt_lat, gt_lon):
        return self._haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

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
    import ast
    import logging

    import torch

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    with open("dataset.csv", "r", encoding="utf-8") as f:
        dataset = csv.reader(f)
        # Skip header row
        next(dataset, None)

        # Initialize tracker
        tracker = YoloObjectTracker(
            model_path="src/controls/detection/best.pt",
            hfov_rad=0.85,
            frame_width=640,
            frame_height=480,
        )

        # Convert dataset to tensors
        tensor_data = []
        for row in dataset:
            if len(row) != 4:
                continue  # Skip invalid rows

            try:
                # Parse string representations of tuples/lists
                pixel = ast.literal_eval(row[0])
                gps_true = ast.literal_eval(row[1])
                gps_drone = ast.literal_eval(row[2])
                attitude = ast.literal_eval(row[3])

                # Convert to tensors
                tensor_data.append(
                    (
                        torch.tensor(pixel, dtype=torch.int32),
                        torch.tensor(gps_true, dtype=torch.double),
                        torch.tensor(gps_drone, dtype=torch.double),
                        torch.tensor(attitude, dtype=torch.double),
                    )
                )
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing row {row}: {e}")
                continue

        print(f"Loaded {len(tensor_data)} valid data points")

        if len(tensor_data) > 0:
            # Optimize adjustment
            tracker.optimize_adjustment(tensor_data, verbose=True)

            # Save final adjustment for inference
            torch.save(tracker.adjustment, "learned_rotation_adjustment.pt")
            print("✅ Saved learned adjustment:", tracker.adjustment.data.detach())
        else:
            print("❌ No valid data found in dataset")
