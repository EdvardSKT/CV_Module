#!/usr/bin/env python3
import sys, site
print("PYTHON:", sys.executable)
print("USER_SITE:", site.getusersitepackages(), "ENABLE:", site.ENABLE_USER_SITE)
print("PATH0:", sys.path[0])
print("SITE0:", sys.path[:3])

import os
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from ros_cv_module_interfaces.msg import DetectionArray, Detection
from ament_index_python.packages import get_package_share_directory

resx = 1920
resy = 1080

origin_coin_x = 518
origin_coin_y = 128


class ConverterNode(Node):
    def __init__(self):
        super().__init__("converter_node")
        self.sub = self.create_subscription(DetectionArray, "detections/pixels", self.cb, 10)
        self.pub = self.create_publisher(DetectionArray, "detections/meters", 10)

        self.K = None
        self.D = None

        self.load_config()
        self.load_fisheye_calibration()

    def load_config(self):
        """Load configuration parameters from external YAML file"""
        try:
            package_share_directory = get_package_share_directory("cv_module")
            config_path = os.path.join(package_share_directory, "config", "converter_config.yaml")

            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(__file__), "..", "config", "converter_config.yaml")

            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            self.scale_x = config.get("scale_x", 0.001)
            self.scale_y = config.get("scale_y", 0.001)

            self.robot_offset_x = config.get("cam_offset_rob_frame_x", 0.0)
            self.robot_offset_y = config.get("cam_offset_rob_frame_y", 0.0)

            # Path to your npz from cv2.fisheye.calibrate()
            # Put this in converter_config.yaml as:
            # fisheye_calib_npz: "/absolute/path/to/gopro_fisheye_calib.npz"
            self.fisheye_calib_npz = os.path.join(package_share_directory,"config","gopro_fisheye_calib.npz")

            # Camera height used in your existing math (meters)
            self.camera_height_m = float(config.get("camera_height_m", 0.39))

            self.get_logger().info(f"Loaded config from {config_path}")
            self.get_logger().info(f"Scale: x={self.scale_x}, y={self.scale_y}")
            self.get_logger().info(f"fisheye_calib_npz: {self.fisheye_calib_npz}")
            self.get_logger().info(f"camera_height_m: {self.camera_height_m}")

        except Exception as e:
            self.get_logger().warn(f"Failed to load config file: {e}")
            self.get_logger().info("Using default parameters")
            self.scale_x = 0.001
            self.scale_y = 0.001
            self.robot_offset_x = 0.0
            self.robot_offset_y = 0.0
            self.fisheye_calib_npz = ""
            self.camera_height_m = 0.39

    def load_fisheye_calibration(self):
        """Load K and D from npz saved by your fisheye calibration script."""
        if not self.fisheye_calib_npz:
            self.get_logger().warn("No fisheye_calib_npz set; distortion correction DISABLED.")
            return

        if not os.path.exists(self.fisheye_calib_npz):
            self.get_logger().warn(f"fisheye_calib_npz not found: {self.fisheye_calib_npz}")
            return

        data = np.load(self.fisheye_calib_npz)
        self.K = data["K"].astype(np.float64)
        self.D = data["D"].astype(np.float64)  # shape (4,1) typically

        # Optional: sanity-log
        self.get_logger().info(f"Loaded fisheye K:\n{self.K}")
        self.get_logger().info(f"Loaded fisheye D: {self.D.reshape(-1).tolist()}")

    def undistort_pixel_to_normalized(self, x_px: float, y_px: float):
        """
        Convert a distorted pixel coordinate to an undistorted *normalized* point (x, y)
        in the rectified pinhole camera coordinate system.

        Output units: "camera normalized" where Z=1 (i.e., x = X/Z, y = Y/Z)
        """
        if self.K is None or self.D is None:
            # Fallback: your previous "normalize by image size" (not physically correct)
            return (x_px - resx / 2) / resx, (y_px - resy / 2) / resy

        pts = np.array([[[x_px, y_px]]], dtype=np.float64)  # shape (1,1,2)

        # P=None -> returns normalized coords (not pixels)
        und = cv2.fisheye.undistortPoints(pts, self.K, self.D, R=np.eye(3), P=None)
        x_n = float(und[0, 0, 0])
        y_n = float(und[0, 0, 1])
        return x_n, y_n

    def cb(self, msg: DetectionArray):
        metric_array = DetectionArray()
        metric_array.header = msg.header

        # Undistort your origin coin pixel too (so both points are in same undistorted space)
        origin_x_u, origin_y_u = self.undistort_pixel_to_normalized(origin_coin_x, origin_coin_y)

        for det in msg.detections:
            m = Detection()
            m.class_id = det.class_id
            m.class_name = det.class_name
            m.confidence = det.confidence

            # Undistort detection center pixel to normalized coords
            x_u, y_u = self.undistort_pixel_to_normalized(det.x, det.y)

            # Your mapping: note you previously swapped axes when assigning m.y/m.x
            # Keep same convention but now using undistorted normalized coordinates.
            m.y = (x_u - origin_x_u) * self.camera_height_m + self.robot_offset_y
            m.x = (y_u - origin_y_u) * self.camera_height_m + self.robot_offset_x

            # Width/height scaling: these are still in pixel space; if you want these
            # to be distortion-correct, you’d need to undistort box corners too.
            m.width = det.width * self.scale_x
            m.height = det.height * self.scale_y

            metric_array.detections.append(m)

        self.pub.publish(metric_array)


def main(args=None):
    rclpy.init(args=args)
    node = ConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
