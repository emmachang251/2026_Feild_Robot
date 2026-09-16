#!/usr/bin/env python3

import math
import time

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

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
class PigDetection:
    confidence: float
    forward_x_m: float
    lateral_y_m: float
    vertical_z_m: float
    depth_m: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def distance_valid(self) -> bool:
        return math.isfinite(self.forward_x_m) and self.forward_x_m > 0.0


class PigDetector(Node):
    """Detect pigs and publish both the legacy nearest result and every box."""

    ALL_DETECTIONS_FORMAT_VERSION = 1.0
    ALL_DETECTIONS_RECORD_SIZE = 10

    def __init__(self) -> None:
        super().__init__('pig_detector')

        # Model and inference settings. These defaults are unchanged.
        self.declare_parameter(
            'model_path',
            '/home/bme1234/rover_ws/models/best_ncnn_model',
        )
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.50)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_hz', 2.0)

        # D435 aligned image topics. These defaults are unchanged.
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

        # Camera pose in the vehicle frame: x forward, y left, z up.
        self.declare_parameter('camera_x_m', 0.14425)
        self.declare_parameter('camera_y_m', 0.033)
        self.declare_parameter('camera_z_m', 0.410)
        self.declare_parameter('camera_pitch_down_deg', 39.11)
        self.declare_parameter('camera_yaw_deg', 0.0)
        self.declare_parameter('camera_roll_deg', 0.0)

        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('minimum_depth_m', 0.20)
        self.declare_parameter('maximum_depth_m', 2.50)
        self.declare_parameter('minimum_depth_pixels', 30)
        self.declare_parameter('minimum_forward_m', 0.10)
        self.declare_parameter('maximum_forward_m', 2.00)
        self.declare_parameter('minimum_height_m', -0.15)
        self.declare_parameter('maximum_height_m', 0.35)
        self.declare_parameter('side_deadband_m', 0.05)

        # Legacy nearest-pig multi-frame confirmation.
        self.declare_parameter('history_size', 3)
        self.declare_parameter('required_hits', 2)
        self.declare_parameter('sensor_timeout_sec', 3.0)
        self.declare_parameter('publish_debug_image', False)

        self.model_path = str(self.get_parameter('model_path').value)
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.image_size = int(self.get_parameter('image_size').value)
        self.inference_hz = float(self.get_parameter('inference_hz').value)

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
        self.minimum_forward_m = float(
            self.get_parameter('minimum_forward_m').value
        )
        self.maximum_forward_m = float(
            self.get_parameter('maximum_forward_m').value
        )
        self.minimum_height_m = float(
            self.get_parameter('minimum_height_m').value
        )
        self.maximum_height_m = float(
            self.get_parameter('maximum_height_m').value
        )
        self.side_deadband_m = float(
            self.get_parameter('side_deadband_m').value
        )

        history_size = int(self.get_parameter('history_size').value)
        self.required_hits = int(self.get_parameter('required_hits').value)
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
        self.last_pig_box_count = 0
        self.last_valid_distance_count = 0

        # Existing outputs remain unchanged.
        self.detected_publisher = self.create_publisher(
            Bool,
            '/level_a/pig_detected',
            10,
        )
        self.side_publisher = self.create_publisher(
            String,
            '/level_a/pig_side',
            10,
        )
        self.detection_publisher = self.create_publisher(
            Float32MultiArray,
            '/level_a/pig_detection',
            10,
        )

        # New output. It contains all YOLO pig boxes from the current frame.
        self.all_detections_publisher = self.create_publisher(
            Float32MultiArray,
            '/level_a/pig_detections',
            10,
        )

        self.debug_publisher = self.create_publisher(
            Image,
            '/level_a/pig_debug_image',
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

        self.get_logger().info(f'載入NCNN模型：{self.model_path}')
        self.model = YOLO(self.model_path)

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

        self.get_logger().info('NCNN模型暖機完成')
        self.get_logger().info(f'推論頻率：{self.inference_hz:.2f} Hz')
        self.get_logger().info(
            '多豬輸出已啟用：/level_a/pig_detections'
        )

    def validate_parameters(self, history_size: int) -> None:
        if self.inference_hz <= 0.0:
            raise ValueError('inference_hz必須大於0')
        if history_size < 1:
            raise ValueError('history_size必須至少為1')
        if not 1 <= self.required_hits <= history_size:
            raise ValueError('required_hits必須介於1與history_size之間')
        if self.minimum_depth_pixels < 1:
            raise ValueError('minimum_depth_pixels必須至少為1')

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
            self.get_logger().warning('尚未收到D435 CameraInfo')
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
                    '對齊深度與彩色影像尺寸不同：'
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

            detections = self.extract_pig_detections(
                result,
                depth_image,
                color_image.shape,
            )
            raw_detection = self.select_nearest_valid_pig(detections)

            self.history.append(raw_detection)
            stable_detection = self.calculate_stable_detection()

            # The old topics keep exactly the old one-pig behavior.
            self.publish_legacy_result(raw_detection, stable_detection)

            # The new topic does not discard boxes whose depth is invalid.
            self.publish_all_detections(detections)

            if self.publish_debug_image:
                self.publish_debug(
                    color_image,
                    detections,
                    stable_detection,
                    color_message,
                )

            self.log_result(
                detections,
                raw_detection,
                stable_detection,
                result.speed,
            )

        except Exception as error:
            self.get_logger().error(f'豬隻辨識處理失敗：{error}')
            self.history.append(None)
            self.publish_legacy_result(None, None)
            self.publish_all_detections([])

    def extract_pig_detections(
        self,
        result,
        depth_image: np.ndarray,
        color_shape,
    ) -> List[PigDetection]:
        image_height, image_width = color_shape[:2]
        detections: List[PigDetection] = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id]).lower()
            if class_name != 'pig':
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = max(0.0, min(x1, image_width - 1.0))
            x2 = max(0.0, min(x2, image_width - 1.0))
            y1 = max(0.0, min(y1, image_height - 1.0))
            y2 = max(0.0, min(y2, image_height - 1.0))

            if x2 - x1 <= 2.0 or y2 - y1 <= 2.0:
                continue

            forward_x_m = math.nan
            lateral_y_m = math.nan
            vertical_z_m = math.nan
            depth_value_m = math.nan

            estimated_depth = self.estimate_pig_depth(
                depth_image,
                x1,
                y1,
                x2,
                y2,
            )

            if estimated_depth is not None:
                pixel_u = (x1 + x2) / 2.0
                pixel_v = y1 + 0.45 * (y2 - y1)
                position = self.pixel_to_vehicle(
                    pixel_u,
                    pixel_v,
                    estimated_depth,
                )

                candidate_forward, candidate_lateral, candidate_vertical = (
                    position
                )
                geometry_valid = (
                    self.minimum_forward_m
                    <= candidate_forward
                    <= self.maximum_forward_m
                    and self.minimum_height_m
                    <= candidate_vertical
                    <= self.maximum_height_m
                )

                if geometry_valid:
                    forward_x_m = candidate_forward
                    lateral_y_m = candidate_lateral
                    vertical_z_m = candidate_vertical
                    depth_value_m = estimated_depth

            detections.append(
                PigDetection(
                    confidence=confidence,
                    forward_x_m=forward_x_m,
                    lateral_y_m=lateral_y_m,
                    vertical_z_m=vertical_z_m,
                    depth_m=depth_value_m,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        self.last_pig_box_count = len(detections)
        self.last_valid_distance_count = sum(
            1 for item in detections if item.distance_valid
        )
        return detections

    @staticmethod
    def select_nearest_valid_pig(
        detections: List[PigDetection],
    ) -> Optional[PigDetection]:
        valid_detections = [
            item for item in detections if item.distance_valid
        ]
        if not valid_detections:
            return None

        return min(
            valid_detections,
            key=lambda item: (item.forward_x_m, -item.confidence),
        )

    def estimate_pig_depth(
        self,
        depth_image: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[float]:
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 2.0 or box_height <= 2.0:
            return None

        roi_x1 = max(0, int(x1 + 0.20 * box_width))
        roi_x2 = min(depth_image.shape[1], int(x1 + 0.80 * box_width))
        roi_y1 = max(0, int(y1 + 0.20 * box_height))
        roi_y2 = min(depth_image.shape[0], int(y1 + 0.70 * box_height))

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None

        roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2].astype(
            np.float32
        )
        depth_values = roi.reshape(-1) * self.depth_scale
        valid_values = depth_values[
            np.isfinite(depth_values)
            & (depth_values >= self.minimum_depth_m)
            & (depth_values <= self.maximum_depth_m)
        ]

        if valid_values.size < self.minimum_depth_pixels:
            return None

        return float(np.percentile(valid_values, 35.0))

    def pixel_to_vehicle(
        self,
        pixel_u: float,
        pixel_v: float,
        depth_m: float,
    ):
        camera_x = (pixel_u - self.cx) * depth_m / self.fx
        camera_y = (pixel_v - self.cy) * depth_m / self.fy
        camera_z = depth_m
        pitch = self.camera_pitch_rad

        forward_x = (
            self.camera_x_m
            - math.sin(pitch) * camera_y
            + math.cos(pitch) * camera_z
        )
        lateral_y = self.camera_y_m - camera_x
        vertical_z = (
            self.camera_z_m
            - math.cos(pitch) * camera_y
            - math.sin(pitch) * camera_z
        )

        return float(forward_x), float(lateral_y), float(vertical_z)

    def calculate_stable_detection(self) -> Optional[PigDetection]:
        detections = [item for item in self.history if item is not None]
        if len(detections) < self.required_hits:
            return None

        lateral_values = [item.lateral_y_m for item in detections]
        median_lateral = float(np.median(lateral_values))
        sides = [self.side_from_lateral(value) for value in lateral_values]
        non_center_sides = [side for side in sides if side != 'CENTER']

        if non_center_sides:
            left_count = non_center_sides.count('LEFT')
            right_count = non_center_sides.count('RIGHT')
            if max(left_count, right_count) < self.required_hits:
                return None

        return PigDetection(
            confidence=float(np.median([item.confidence for item in detections])),
            forward_x_m=float(
                np.median([item.forward_x_m for item in detections])
            ),
            lateral_y_m=median_lateral,
            vertical_z_m=float(
                np.median([item.vertical_z_m for item in detections])
            ),
            depth_m=float(np.median([item.depth_m for item in detections])),
            x1=float(np.median([item.x1 for item in detections])),
            y1=float(np.median([item.y1 for item in detections])),
            x2=float(np.median([item.x2 for item in detections])),
            y2=float(np.median([item.y2 for item in detections])),
        )

    def side_from_lateral(self, lateral_y_m: float) -> str:
        if not math.isfinite(lateral_y_m):
            return 'CENTER'
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

    def publish_legacy_result(
        self,
        raw_detection: Optional[PigDetection],
        stable_detection: Optional[PigDetection],
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

        output_detection = stable_detection or raw_detection
        message = Float32MultiArray()

        if output_detection is None:
            message.data = [
                0.0,
                0.0,
                0.0,
                math.nan,
                math.nan,
                math.nan,
                0.0,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
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
            ]

        self.detection_publisher.publish(message)

    def publish_all_detections(
        self,
        detections: List[PigDetection],
    ) -> None:
        """
        Publish a flat Float32MultiArray.

        Header:
            [0] format_version (=1)
            [1] record_size (=10)
            [2] detection_count

        Repeated record:
            [0] confidence
            [1] forward_x_m (NaN when per-pig depth is invalid)
            [2] lateral_y_m
            [3] vertical_z_m
            [4] side_code (+1 LEFT, -1 RIGHT, 0 CENTER/unknown)
            [5] optical_depth_m
            [6] bbox_x1
            [7] bbox_y1
            [8] bbox_x2
            [9] bbox_y2
        """
        data = [
            self.ALL_DETECTIONS_FORMAT_VERSION,
            float(self.ALL_DETECTIONS_RECORD_SIZE),
            float(len(detections)),
        ]

        for detection in detections:
            side = self.side_from_lateral(detection.lateral_y_m)
            data.extend(
                [
                    detection.confidence,
                    detection.forward_x_m,
                    detection.lateral_y_m,
                    detection.vertical_z_m,
                    self.side_code(side),
                    detection.depth_m,
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                ]
            )

        message = Float32MultiArray()
        message.data = data
        self.all_detections_publisher.publish(message)

    def publish_debug(
        self,
        image: np.ndarray,
        detections: List[PigDetection],
        stable_detection: Optional[PigDetection],
        color_message: Image,
    ) -> None:
        debug_image = image.copy()

        for detection in detections:
            distance_color = (
                (0, 255, 0) if detection.distance_valid else (0, 165, 255)
            )
            cv2.rectangle(
                debug_image,
                (int(detection.x1), int(detection.y1)),
                (int(detection.x2), int(detection.y2)),
                distance_color,
                2,
            )
            distance_text = (
                f'{detection.forward_x_m:.2f}m'
                if detection.distance_valid
                else 'depth invalid'
            )
            text = f'pig {detection.confidence:.2f} {distance_text}'
            cv2.putText(
                debug_image,
                text,
                (int(detection.x1), max(24, int(detection.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                distance_color,
                2,
            )

        if stable_detection is not None:
            cv2.rectangle(
                debug_image,
                (int(stable_detection.x1), int(stable_detection.y1)),
                (int(stable_detection.x2), int(stable_detection.y2)),
                (255, 255, 0),
                3,
            )

        if not detections:
            cv2.putText(
                debug_image,
                'NO PIG',
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
        detections: List[PigDetection],
        raw_detection: Optional[PigDetection],
        stable_detection: Optional[PigDetection],
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
                f'STABLE pig: boxes={len(detections)}, '
                f'valid_depth={self.last_valid_distance_count}, '
                f'conf={stable_detection.confidence:.3f}, '
                f'x={stable_detection.forward_x_m:.3f} m, '
                f'y={stable_detection.lateral_y_m:.3f} m, '
                f'side={side}, inference={inference_ms:.1f} ms'
            )
        elif raw_detection is not None:
            self.get_logger().info(
                f'RAW pig: boxes={len(detections)}, '
                f'valid_depth={self.last_valid_distance_count}, '
                f'conf={raw_detection.confidence:.3f}, '
                f'x={raw_detection.forward_x_m:.3f} m, '
                f'inference={inference_ms:.1f} ms'
            )
        elif detections:
            self.get_logger().info(
                f'Pig boxes={len(detections)}, but all per-pig distances '
                f'are invalid; inference={inference_ms:.1f} ms'
            )
        else:
            self.get_logger().info(
                f'No pig, inference={inference_ms:.1f} ms'
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
        self.publish_legacy_result(None, None)
        self.publish_all_detections([])
        self.get_logger().error(
            'D435同步影像逾時，pig_detected已強制設為False'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PigDetector()

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
