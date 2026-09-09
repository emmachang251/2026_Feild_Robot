#!/usr/bin/env python3

import math
import time

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32MultiArray, String
from ultralytics import YOLO


@dataclass
class ManureDetection:
    confidence: float
    forward_x_m: float
    lateral_y_m: float
    vertical_z_m: float
    depth_m: float
    ground_fraction: float
    x1: float
    y1: float
    x2: float
    y2: float


class ManureGrooveDetector(Node):

    def __init__(self) -> None:
        super().__init__('manure_groove_detector')

        # Model and inference settings.
        self.declare_parameter(
            'model_path',
            '/home/bme1234/rover_ws/models/manure_best_ncnn_model',
        )
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('iou_threshold', 0.50)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_hz', 2.0)

        # Aligned D435 topics.
        self.declare_parameter(
            'color_topic',
            '/camera/camera/color/image_raw',
        )
        self.declare_parameter(
            'depth_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
        )
        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/aligned_depth_to_color/camera_info',
        )

        # Camera pose in vehicle coordinates: x forward, y left, z up.
        self.declare_parameter('camera_x_m', 0.14425)
        self.declare_parameter('camera_y_m', 0.033)
        self.declare_parameter('camera_z_m', 0.410)
        self.declare_parameter('camera_pitch_down_deg', 39.11)

        # The current installation has approximately zero yaw and roll.
        self.declare_parameter('camera_yaw_deg', 0.0)
        self.declare_parameter('camera_roll_deg', 0.0)

        # 16UC1 depth uses millimetres. Floating depth is treated as metres.
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('minimum_depth_m', 0.20)
        self.declare_parameter('maximum_depth_m', 2.50)
        self.declare_parameter('minimum_depth_pixels', 20)

        # Only use an inset region of each YOLO box for depth measurements.
        self.declare_parameter('depth_roi_inset_ratio', 0.10)
        self.declare_parameter('depth_sample_step', 2)

        # Physical validity limits in the vehicle frame.
        self.declare_parameter('minimum_forward_m', 0.10)
        self.declare_parameter('maximum_forward_m', 2.00)
        self.declare_parameter('maximum_abs_lateral_m', 0.80)

        # Manure must contain enough measured points near the floor plane.
        # These initial values are deliberately tolerant of D435 noise and
        # must be tightened only after inspecting live z measurements.
        self.declare_parameter('minimum_ground_height_m', -0.08)
        self.declare_parameter('maximum_ground_height_m', 0.12)
        self.declare_parameter('minimum_ground_pixels', 15)
        self.declare_parameter('minimum_ground_fraction', 0.30)

        self.declare_parameter('side_deadband_m', 0.05)

        # Multi-frame confirmation.
        self.declare_parameter('history_size', 3)
        self.declare_parameter('required_hits', 2)
        self.declare_parameter('maximum_history_spread_m', 0.20)
        self.declare_parameter('sensor_timeout_sec', 3.0)
        self.declare_parameter('publish_debug_image', True)

        self.model_path = str(self.get_parameter('model_path').value)
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.iou_threshold = float(
            self.get_parameter('iou_threshold').value
        )
        self.image_size = int(self.get_parameter('image_size').value)
        self.inference_hz = float(
            self.get_parameter('inference_hz').value
        )

        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )

        self.camera_x_m = float(self.get_parameter('camera_x_m').value)
        self.camera_y_m = float(self.get_parameter('camera_y_m').value)
        self.camera_z_m = float(self.get_parameter('camera_z_m').value)
        self.camera_pitch_rad = math.radians(
            float(self.get_parameter('camera_pitch_down_deg').value)
        )
        self.camera_yaw_rad = math.radians(
            float(self.get_parameter('camera_yaw_deg').value)
        )
        self.camera_roll_rad = math.radians(
            float(self.get_parameter('camera_roll_deg').value)
        )

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.minimum_depth_m = float(
            self.get_parameter('minimum_depth_m').value
        )
        self.maximum_depth_m = float(
            self.get_parameter('maximum_depth_m').value
        )
        self.minimum_depth_pixels = int(
            self.get_parameter('minimum_depth_pixels').value
        )
        self.depth_roi_inset_ratio = float(
            self.get_parameter('depth_roi_inset_ratio').value
        )
        self.depth_sample_step = int(
            self.get_parameter('depth_sample_step').value
        )

        self.minimum_forward_m = float(
            self.get_parameter('minimum_forward_m').value
        )
        self.maximum_forward_m = float(
            self.get_parameter('maximum_forward_m').value
        )
        self.maximum_abs_lateral_m = float(
            self.get_parameter('maximum_abs_lateral_m').value
        )
        self.minimum_ground_height_m = float(
            self.get_parameter('minimum_ground_height_m').value
        )
        self.maximum_ground_height_m = float(
            self.get_parameter('maximum_ground_height_m').value
        )
        self.minimum_ground_pixels = int(
            self.get_parameter('minimum_ground_pixels').value
        )
        self.minimum_ground_fraction = float(
            self.get_parameter('minimum_ground_fraction').value
        )
        self.side_deadband_m = float(
            self.get_parameter('side_deadband_m').value
        )

        history_size = int(self.get_parameter('history_size').value)
        self.required_hits = int(self.get_parameter('required_hits').value)
        self.maximum_history_spread_m = float(
            self.get_parameter('maximum_history_spread_m').value
        )
        self.sensor_timeout_sec = float(
            self.get_parameter('sensor_timeout_sec').value
        )
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value
        )

        self.validate_parameters(history_size)
        self.inference_period_sec = 1.0 / self.inference_hz

        self.bridge = CvBridge()
        self.history = deque(maxlen=history_size)

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.last_inference_time = 0.0
        self.last_sensor_time = 0.0
        self.last_log_time = 0.0
        self.sensor_timeout_published = False
        self.last_yolo_box_count = 0
        self.last_geometry_pass_count = 0

        self.detected_publisher = self.create_publisher(
            Bool,
            '/level_a/manure_detected',
            10,
        )
        self.side_publisher = self.create_publisher(
            String,
            '/level_a/manure_side',
            10,
        )
        self.detection_publisher = self.create_publisher(
            Float32MultiArray,
            '/level_a/manure_detection',
            10,
        )
        self.debug_publisher = self.create_publisher(
            Image,
            '/level_a/manure_debug_image',
            2,
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )
        self.color_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.color_topic,
            qos_profile=sensor_qos,
        )
        self.depth_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=sensor_qos,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            queue_size=30,
            slop=0.20,
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

        self.timeout_timer = self.create_timer(0.2, self.timeout_callback)

        self.get_logger().info(f'Loading NCNN model: {self.model_path}')
        self.model = YOLO(self.model_path, task='detect')

        dummy_image = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        self.model.predict(
            source=dummy_image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device='cpu',
            save=False,
            verbose=False,
        )

        self.get_logger().info('NCNN model warm-up complete')
        self.get_logger().info(
            'Manure detector ready: '
            f'conf={self.confidence_threshold:.2f}, '
            f'inference={self.inference_hz:.2f} Hz'
        )

    def validate_parameters(self, history_size: int) -> None:
        if self.inference_hz <= 0.0:
            raise ValueError('inference_hz must be greater than zero')
        if self.depth_sample_step < 1:
            raise ValueError('depth_sample_step must be at least one')
        if not 0.0 <= self.depth_roi_inset_ratio < 0.50:
            raise ValueError('depth_roi_inset_ratio must be in [0, 0.5)')
        if not 0.0 <= self.minimum_ground_fraction <= 1.0:
            raise ValueError('minimum_ground_fraction must be in [0, 1]')
        if history_size < 1:
            raise ValueError('history_size must be at least one')
        if not 1 <= self.required_hits <= history_size:
            raise ValueError('required_hits must be in [1, history_size]')

    def camera_info_callback(self, message: CameraInfo) -> None:
        if len(message.k) < 9:
            return
        self.fx = float(message.k[0])
        self.fy = float(message.k[4])
        self.cx = float(message.k[2])
        self.cy = float(message.k[5])

    def synchronized_callback(
        self,
        color_message: Image,
        depth_message: Image,
    ) -> None:
        now = time.monotonic()
        self.last_sensor_time = now
        self.sensor_timeout_published = False

        if now - self.last_inference_time < self.inference_period_sec:
            return
        self.last_inference_time = now

        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().warning('Waiting for D435 CameraInfo')
            return

        try:
            color_image = self.bridge.imgmsg_to_cv2(
                color_message,
                desired_encoding='bgr8',
            )
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding='passthrough',
            )

            if color_image.shape[:2] != depth_image.shape[:2]:
                raise ValueError(
                    'Aligned depth size does not match color size: '
                    f'color={color_image.shape[:2]}, '
                    f'depth={depth_image.shape[:2]}'
                )

            result = self.model.predict(
                source=color_image,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device='cpu',
                save=False,
                verbose=False,
            )[0]

            raw_detection = self.select_nearest_manure(
                result,
                depth_image,
                color_image.shape,
            )
            self.history.append(raw_detection)
            stable_detection = self.calculate_stable_detection()

            self.publish_result(raw_detection, stable_detection)

            if self.publish_debug_image:
                self.publish_debug(
                    color_image,
                    raw_detection,
                    stable_detection,
                    color_message,
                )

            self.log_result(raw_detection, stable_detection, result.speed)

        except Exception as error:
            self.get_logger().error(f'Manure processing failed: {error}')
            self.history.append(None)
            self.publish_result(None, None)

    def select_nearest_manure(
        self,
        result,
        depth_image: np.ndarray,
        color_shape,
    ) -> Optional[ManureDetection]:
        image_height, image_width = color_shape[:2]
        candidates = []
        self.last_yolo_box_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id]).lower()
            if class_name != 'manure':
                continue

            self.last_yolo_box_count += 1
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = max(0.0, min(x1, image_width - 1.0))
            x2 = max(0.0, min(x2, image_width - 1.0))
            y1 = max(0.0, min(y1, image_height - 1.0))
            y2 = max(0.0, min(y2, image_height - 1.0))

            measurement = self.estimate_ground_position(
                depth_image,
                x1,
                y1,
                x2,
                y2,
            )
            if measurement is None:
                continue

            (
                forward_x_m,
                lateral_y_m,
                vertical_z_m,
                depth_m,
                ground_fraction,
            ) = measurement

            candidates.append(
                ManureDetection(
                    confidence=confidence,
                    forward_x_m=forward_x_m,
                    lateral_y_m=lateral_y_m,
                    vertical_z_m=vertical_z_m,
                    depth_m=depth_m,
                    ground_fraction=ground_fraction,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        self.last_geometry_pass_count = len(candidates)
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: (item.forward_x_m, -item.confidence),
        )

    def estimate_ground_position(
        self,
        depth_image: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 2.0 or box_height <= 2.0:
            return None

        inset = self.depth_roi_inset_ratio
        roi_x1 = max(0, int(math.floor(x1 + inset * box_width)))
        roi_x2 = min(
            depth_image.shape[1],
            int(math.ceil(x2 - inset * box_width)),
        )
        roi_y1 = max(0, int(math.floor(y1 + inset * box_height)))
        roi_y2 = min(
            depth_image.shape[0],
            int(math.ceil(y2 - inset * box_height)),
        )
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None

        step = self.depth_sample_step
        raw_depth = depth_image[
            roi_y1:roi_y2:step,
            roi_x1:roi_x2:step,
        ]
        if raw_depth.size == 0:
            return None

        if np.issubdtype(raw_depth.dtype, np.integer):
            depth_m = raw_depth.astype(np.float32) * self.depth_scale
        else:
            depth_m = raw_depth.astype(np.float32)

        pixel_u = np.arange(roi_x1, roi_x2, step, dtype=np.float32)
        pixel_v = np.arange(roi_y1, roi_y2, step, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(pixel_u, pixel_v)

        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= self.minimum_depth_m)
            & (depth_m <= self.maximum_depth_m)
        )
        valid_count = int(np.count_nonzero(valid_depth))
        if valid_count < self.minimum_depth_pixels:
            return None

        depths = depth_m[valid_depth]
        u_values = grid_u[valid_depth]
        v_values = grid_v[valid_depth]

        camera_right = (u_values - self.cx) * depths / self.fx
        camera_down = (v_values - self.cy) * depths / self.fy

        pitch = self.camera_pitch_rad
        forward = (
            self.camera_x_m
            - math.sin(pitch) * camera_down
            + math.cos(pitch) * depths
        )
        lateral = self.camera_y_m - camera_right
        vertical = (
            self.camera_z_m
            - math.cos(pitch) * camera_down
            - math.sin(pitch) * depths
        )

        ground_mask = (
            np.isfinite(forward)
            & np.isfinite(lateral)
            & np.isfinite(vertical)
            & (forward >= self.minimum_forward_m)
            & (forward <= self.maximum_forward_m)
            & (np.abs(lateral) <= self.maximum_abs_lateral_m)
            & (vertical >= self.minimum_ground_height_m)
            & (vertical <= self.maximum_ground_height_m)
        )

        ground_count = int(np.count_nonzero(ground_mask))
        ground_fraction = ground_count / float(valid_count)
        if ground_count < self.minimum_ground_pixels:
            return None
        if ground_fraction < self.minimum_ground_fraction:
            return None

        return (
            float(np.median(forward[ground_mask])),
            float(np.median(lateral[ground_mask])),
            float(np.median(vertical[ground_mask])),
            float(np.median(depths[ground_mask])),
            float(ground_fraction),
        )

    def calculate_stable_detection(self) -> Optional[ManureDetection]:
        detections = [item for item in self.history if item is not None]
        if len(detections) < self.required_hits:
            return None

        median_forward = float(
            np.median([item.forward_x_m for item in detections])
        )
        median_lateral = float(
            np.median([item.lateral_y_m for item in detections])
        )

        spatially_consistent = [
            item
            for item in detections
            if math.hypot(
                item.forward_x_m - median_forward,
                item.lateral_y_m - median_lateral,
            ) <= self.maximum_history_spread_m
        ]
        if len(spatially_consistent) < self.required_hits:
            return None

        sides = [
            self.side_from_lateral(item.lateral_y_m)
            for item in spatially_consistent
        ]
        non_center_sides = [side for side in sides if side != 'CENTER']
        if non_center_sides:
            left_count = non_center_sides.count('LEFT')
            right_count = non_center_sides.count('RIGHT')
            if max(left_count, right_count) < self.required_hits:
                return None

        items = spatially_consistent
        return ManureDetection(
            confidence=float(np.median([item.confidence for item in items])),
            forward_x_m=float(np.median([item.forward_x_m for item in items])),
            lateral_y_m=float(np.median([item.lateral_y_m for item in items])),
            vertical_z_m=float(np.median([item.vertical_z_m for item in items])),
            depth_m=float(np.median([item.depth_m for item in items])),
            ground_fraction=float(
                np.median([item.ground_fraction for item in items])
            ),
            x1=float(np.median([item.x1 for item in items])),
            y1=float(np.median([item.y1 for item in items])),
            x2=float(np.median([item.x2 for item in items])),
            y2=float(np.median([item.y2 for item in items])),
        )

    def side_from_lateral(self, lateral_y_m: float) -> str:
        if lateral_y_m > self.side_deadband_m:
            return 'LEFT'
        if lateral_y_m < -self.side_deadband_m:
            return 'RIGHT'
        return 'CENTER'

    @staticmethod
    def side_code(side: str) -> float:
        if side == 'LEFT':
            return 1.0
        if side == 'RIGHT':
            return -1.0
        return 0.0

    def publish_result(
        self,
        raw_detection: Optional[ManureDetection],
        stable_detection: Optional[ManureDetection],
    ) -> None:
        detected_message = Bool()
        detected_message.data = stable_detection is not None
        self.detected_publisher.publish(detected_message)

        side_message = String()
        if stable_detection is None:
            side_message.data = 'NONE'
        else:
            side_message.data = self.side_from_lateral(
                stable_detection.lateral_y_m
            )
        self.side_publisher.publish(side_message)

        output_detection = (
            stable_detection
            if stable_detection is not None
            else raw_detection
        )
        message = Float32MultiArray()

        if output_detection is None:
            message.data = [
                0.0,       # 0 stable_valid
                0.0,       # 1 raw_detected
                0.0,       # 2 confidence
                math.nan,  # 3 forward_x_m
                math.nan,  # 4 lateral_y_m
                math.nan,  # 5 vertical_z_m
                0.0,       # 6 side_code
                math.nan,  # 7 optical_depth_m
                math.nan,  # 8 bbox_x1
                math.nan,  # 9 bbox_y1
                math.nan,  # 10 bbox_x2
                math.nan,  # 11 bbox_y2
                0.0,       # 12 ground_fraction
            ]
        else:
            side = self.side_from_lateral(output_detection.lateral_y_m)
            message.data = [
                1.0 if stable_detection is not None else 0.0,
                1.0 if raw_detection is not None else 0.0,
                output_detection.confidence,
                output_detection.forward_x_m,
                output_detection.lateral_y_m,
                output_detection.vertical_z_m,
                self.side_code(side),
                output_detection.depth_m,
                output_detection.x1,
                output_detection.y1,
                output_detection.x2,
                output_detection.y2,
                output_detection.ground_fraction,
            ]

        self.detection_publisher.publish(message)

    def publish_debug(
        self,
        image: np.ndarray,
        raw_detection: Optional[ManureDetection],
        stable_detection: Optional[ManureDetection],
        color_message: Image,
    ) -> None:
        debug_image = image.copy()
        detection = stable_detection or raw_detection

        if detection is not None:
            color = (0, 255, 0) if stable_detection is not None else (0, 165, 255)
            cv2.rectangle(
                debug_image,
                (int(detection.x1), int(detection.y1)),
                (int(detection.x2), int(detection.y2)),
                color,
                3,
            )
            side = self.side_from_lateral(detection.lateral_y_m)
            text = (
                f'manure {detection.confidence:.2f} '
                f'x={detection.forward_x_m:.2f}m '
                f'y={detection.lateral_y_m:.2f}m '
                f'z={detection.vertical_z_m:.2f}m '
                f'g={detection.ground_fraction:.2f} {side}'
            )
            cv2.putText(
                debug_image,
                text,
                (int(detection.x1), max(30, int(detection.y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
        else:
            cv2.putText(
                debug_image,
                'NO VALID MANURE',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        debug_message = self.bridge.cv2_to_imgmsg(
            debug_image,
            encoding='bgr8',
        )
        debug_message.header = color_message.header
        self.debug_publisher.publish(debug_message)

    def log_result(
        self,
        raw_detection: Optional[ManureDetection],
        stable_detection: Optional[ManureDetection],
        speed,
    ) -> None:
        now = time.monotonic()
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now

        inference_ms = float(speed.get('inference', 0.0))
        if stable_detection is not None:
            side = self.side_from_lateral(stable_detection.lateral_y_m)
            self.get_logger().info(
                'STABLE manure: '
                f'conf={stable_detection.confidence:.3f}, '
                f'x={stable_detection.forward_x_m:.3f} m, '
                f'y={stable_detection.lateral_y_m:.3f} m, '
                f'z={stable_detection.vertical_z_m:.3f} m, '
                f'ground={stable_detection.ground_fraction:.2f}, '
                f'side={side}, inference={inference_ms:.1f} ms'
            )
        elif raw_detection is not None:
            self.get_logger().info(
                'RAW manure, waiting for multi-frame confirmation: '
                f'conf={raw_detection.confidence:.3f}, '
                f'x={raw_detection.forward_x_m:.3f} m, '
                f'y={raw_detection.lateral_y_m:.3f} m, '
                f'z={raw_detection.vertical_z_m:.3f} m, '
                f'ground={raw_detection.ground_fraction:.2f}, '
                f'inference={inference_ms:.1f} ms'
            )
        else:
            self.get_logger().info(
                'No valid manure: '
                f'yolo_boxes={self.last_yolo_box_count}, '
                f'geometry_pass={self.last_geometry_pass_count}, '
                f'inference={inference_ms:.1f} ms'
            )

    def timeout_callback(self) -> None:
        if self.last_sensor_time <= 0.0:
            return

        elapsed = time.monotonic() - self.last_sensor_time
        if elapsed <= self.sensor_timeout_sec:
            return
        if self.sensor_timeout_published:
            return

        self.sensor_timeout_published = True
        self.history.clear()
        self.publish_result(None, None)
        self.get_logger().error(
            'D435 synchronized images timed out; '
            'manure_detected forced to False'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManureGrooveDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
