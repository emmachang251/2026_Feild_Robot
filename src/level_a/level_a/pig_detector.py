#!/usr/bin/env python3

import math
import time

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32MultiArray, String
from ultralytics import YOLO
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

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


class PigDetector(Node):

    def __init__(self) -> None:
        super().__init__('pig_detector')

        # ----------------------------------------------------------
        # 模型與推論參數
        # ----------------------------------------------------------
        self.declare_parameter(
            'model_path',
            '/home/bme1234/rover_ws/models/best_ncnn_model',
        )
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.50)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_hz', 2.0)

        # ----------------------------------------------------------
        # D435 Topics
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # D435相對車體中心的位置
        #
        # 車體座標：
        # x：向車頭為正
        # y：向車體左側為正
        # z：向上為正
        # ----------------------------------------------------------
        self.declare_parameter('camera_x_m', 0.14425)
        self.declare_parameter('camera_y_m', 0.033)
        self.declare_parameter('camera_z_m', 0.410)

        self.declare_parameter(
            'camera_pitch_down_deg',
            39.11,
        )

        # 目前實測 yaw、roll 皆視為0度。
        self.declare_parameter('camera_yaw_deg', 0.0)
        self.declare_parameter('camera_roll_deg', 0.0)

        # 16UC1深度影像：1單位 = 1 mm。
        self.declare_parameter('depth_scale', 0.001)

        # ----------------------------------------------------------
        # 深度取樣及物理合理性限制
        # ----------------------------------------------------------
        self.declare_parameter('minimum_depth_m', 0.20)
        self.declare_parameter('maximum_depth_m', 2.50)
        self.declare_parameter('minimum_depth_pixels', 30)

        self.declare_parameter('minimum_forward_m', 0.10)
        self.declare_parameter('maximum_forward_m', 2.00)

        # 初期設定較寬鬆，之後依實測的豬中心高度縮小。
        self.declare_parameter('minimum_height_m', -0.15)
        self.declare_parameter('maximum_height_m', 0.35)

        # y絕對值小於5 cm時暫時判為CENTER。
        self.declare_parameter('side_deadband_m', 0.05)

        # ----------------------------------------------------------
        # 多幀確認
        #
        # 最近3次推論至少2次看到豬，才發布stable_detected。
        # ----------------------------------------------------------
        self.declare_parameter('history_size', 3)
        self.declare_parameter('required_hits', 2)
        self.declare_parameter('sensor_timeout_sec', 3.0)

        self.declare_parameter(
            'publish_debug_image',
            False,
        )

        self.model_path = str(
            self.get_parameter('model_path').value
        )
        self.confidence_threshold = float(
            self.get_parameter(
                'confidence_threshold'
            ).value
        )
        self.iou_threshold = float(
            self.get_parameter('iou_threshold').value
        )
        self.image_size = int(
            self.get_parameter('image_size').value
        )
        self.inference_hz = float(
            self.get_parameter('inference_hz').value
        )

        self.color_topic = str(
            self.get_parameter('color_topic').value
        )
        self.depth_topic = str(
            self.get_parameter('depth_topic').value
        )
        self.camera_info_topic = str(
            self.get_parameter(
                'camera_info_topic'
            ).value
        )

        self.camera_x_m = float(
            self.get_parameter('camera_x_m').value
        )
        self.camera_y_m = float(
            self.get_parameter('camera_y_m').value
        )
        self.camera_z_m = float(
            self.get_parameter('camera_z_m').value
        )

        self.camera_pitch_rad = math.radians(
            float(
                self.get_parameter(
                    'camera_pitch_down_deg'
                ).value
            )
        )

        self.camera_yaw_rad = math.radians(
            float(
                self.get_parameter(
                    'camera_yaw_deg'
                ).value
            )
        )

        self.camera_roll_rad = math.radians(
            float(
                self.get_parameter(
                    'camera_roll_deg'
                ).value
            )
        )

        self.depth_scale = float(
            self.get_parameter('depth_scale').value
        )
        self.minimum_depth_m = float(
            self.get_parameter('minimum_depth_m').value
        )
        self.maximum_depth_m = float(
            self.get_parameter('maximum_depth_m').value
        )
        self.minimum_depth_pixels = int(
            self.get_parameter(
                'minimum_depth_pixels'
            ).value
        )

        self.minimum_forward_m = float(
            self.get_parameter(
                'minimum_forward_m'
            ).value
        )
        self.maximum_forward_m = float(
            self.get_parameter(
                'maximum_forward_m'
            ).value
        )
        self.minimum_height_m = float(
            self.get_parameter(
                'minimum_height_m'
            ).value
        )
        self.maximum_height_m = float(
            self.get_parameter(
                'maximum_height_m'
            ).value
        )
        self.side_deadband_m = float(
            self.get_parameter(
                'side_deadband_m'
            ).value
        )

        history_size = int(
            self.get_parameter('history_size').value
        )
        self.required_hits = int(
            self.get_parameter('required_hits').value
        )
        self.sensor_timeout_sec = float(
            self.get_parameter(
                'sensor_timeout_sec'
            ).value
        )
        self.publish_debug_image = bool(
            self.get_parameter(
                'publish_debug_image'
            ).value
        )

        if self.inference_hz <= 0.0:
            raise ValueError(
                'inference_hz必須大於0'
            )

        if self.required_hits > history_size:
            raise ValueError(
                'required_hits不能大於history_size'
            )

        self.inference_period_sec = (
            1.0 / self.inference_hz
        )

        self.bridge = CvBridge()
        self.history = deque(
            maxlen=history_size
        )

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.last_inference_time = 0.0
        self.last_sensor_time = 0.0
        self.last_log_time = 0.0
        self.sensor_timeout_published = False

        # ----------------------------------------------------------
        # ROS Publishers
        # ----------------------------------------------------------
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

        self.camera_info_subscription = (
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                sensor_qos,
            )
        )

        self.color_subscriber = (
            message_filters.Subscriber(
                self,
                Image,
                self.color_topic,
                qos_profile=sensor_qos,
            )
        )

        self.depth_subscriber = (
            message_filters.Subscriber(
                self,
                Image,
                self.depth_topic,
                qos_profile=sensor_qos,
            )
        )

        self.synchronizer = (
             message_filters.ApproximateTimeSynchronizer(
             [
                self.color_subscriber,
                self.depth_subscriber,
            ],
            queue_size=30,
            slop=0.20,
            )
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.timeout_timer = self.create_timer(
            0.2,
            self.timeout_callback,
        )

        # ----------------------------------------------------------
        # 載入並暖機模型
        # ----------------------------------------------------------
        self.get_logger().info(
            f'載入NCNN模型：{self.model_path}'
        )

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

        self.get_logger().info(
            'NCNN模型暖機完成'
        )
        self.get_logger().info(
            f'推論頻率：{self.inference_hz:.2f} Hz'
        )

    def camera_info_callback(
        self,
        message: CameraInfo,
    ) -> None:
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

        if (
            now - self.last_inference_time
            < self.inference_period_sec
        ):
            return

        self.last_inference_time = now

        if None in (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        ):
            self.get_logger().warning(
                '尚未收到D435 CameraInfo'
            )
            return

        try:
            color_image = (
                self.bridge.imgmsg_to_cv2(
                    color_message,
                    desired_encoding='bgr8',
                )
            )

            depth_image = (
                self.bridge.imgmsg_to_cv2(
                    depth_message,
                    desired_encoding='passthrough',
                )
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

            raw_detection = (
                self.select_nearest_pig(
                    result,
                    depth_image,
                    color_image.shape,
                )
            )

            self.history.append(raw_detection)

            stable_detection = (
                self.calculate_stable_detection()
            )

            self.publish_result(
                raw_detection,
                stable_detection,
            )

            if self.publish_debug_image:
                self.publish_debug(
                    color_image,
                    raw_detection,
                    stable_detection,
                )

            self.log_result(
                raw_detection,
                stable_detection,
                result.speed,
            )

        except Exception as error:
            self.get_logger().error(
                f'豬隻辨識處理失敗：{error}'
            )

            self.history.append(None)
            self.publish_result(None, None)

    def select_nearest_pig(
        self,
        result,
        depth_image: np.ndarray,
        color_shape,
    ) -> Optional[PigDetection]:
        image_height, image_width = (
            color_shape[:2]
        )

        candidates = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            if class_name != 'pig':
                continue

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            x1 = max(
                0.0,
                min(x1, image_width - 1.0),
            )
            x2 = max(
                0.0,
                min(x2, image_width - 1.0),
            )
            y1 = max(
                0.0,
                min(y1, image_height - 1.0),
            )
            y2 = max(
                0.0,
                min(y2, image_height - 1.0),
            )

            depth_m = self.estimate_pig_depth(
                depth_image,
                x1,
                y1,
                x2,
                y2,
            )

            if depth_m is None:
                continue

            pixel_u = (x1 + x2) / 2.0

            # 使用方框上方45%的位置代表豬身體中心，
            # 避免方框底部地面影響定位。
            pixel_v = y1 + 0.45 * (y2 - y1)

            forward_x_m, lateral_y_m, vertical_z_m = (
                self.pixel_to_vehicle(
                    pixel_u,
                    pixel_v,
                    depth_m,
                )
            )

            if not (
                self.minimum_forward_m
                <= forward_x_m
                <= self.maximum_forward_m
            ):
                continue

            if not (
                self.minimum_height_m
                <= vertical_z_m
                <= self.maximum_height_m
            ):
                continue

            candidates.append(
                PigDetection(
                    confidence=confidence,
                    forward_x_m=forward_x_m,
                    lateral_y_m=lateral_y_m,
                    vertical_z_m=vertical_z_m,
                    depth_m=depth_m,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        if not candidates:
            return None

        # 如果同時看到多個框，選車體前方距離最近者。
        return min(
            candidates,
            key=lambda item: item.forward_x_m,
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

        # 只取豬身體中央範圍，避開背景與地面。
        roi_x1 = int(x1 + 0.20 * box_width)
        roi_x2 = int(x1 + 0.80 * box_width)
        roi_y1 = int(y1 + 0.20 * box_height)
        roi_y2 = int(y1 + 0.70 * box_height)

        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(
            depth_image.shape[1],
            roi_x2,
        )
        roi_y2 = min(
            depth_image.shape[0],
            roi_y2,
        )

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None

        roi = depth_image[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ].astype(np.float32)

        depth_values = (
            roi.reshape(-1) * self.depth_scale
        )

        valid_values = depth_values[
            np.isfinite(depth_values)
            & (
                depth_values
                >= self.minimum_depth_m
            )
            & (
                depth_values
                <= self.maximum_depth_m
            )
        ]

        if (
            valid_values.size
            < self.minimum_depth_pixels
        ):
            return None

        # 取35百分位，降低方框背景深度的影響。
        return float(
            np.percentile(valid_values, 35.0)
        )

    def pixel_to_vehicle(
        self,
        pixel_u: float,
        pixel_v: float,
        depth_m: float,
    ):
        camera_x = (
            (pixel_u - self.cx)
            * depth_m
            / self.fx
        )
        camera_y = (
            (pixel_v - self.cy)
            * depth_m
            / self.fy
        )
        camera_z = depth_m

        pitch = self.camera_pitch_rad

        # D435 optical frame：
        # camera_x：畫面向右
        # camera_y：畫面向下
        # camera_z：鏡頭向前
        #
        # 轉換到車體座標：
        forward_x = (
            self.camera_x_m
            - math.sin(pitch) * camera_y
            + math.cos(pitch) * camera_z
        )

        lateral_y = (
            self.camera_y_m
            - camera_x
        )

        vertical_z = (
            self.camera_z_m
            - math.cos(pitch) * camera_y
            - math.sin(pitch) * camera_z
        )

        return (
            float(forward_x),
            float(lateral_y),
            float(vertical_z),
        )

    def calculate_stable_detection(
        self,
    ) -> Optional[PigDetection]:
        detections = [
            item
            for item in self.history
            if item is not None
        ]

        if len(detections) < self.required_hits:
            return None

        lateral_values = [
            item.lateral_y_m
            for item in detections
        ]

        median_lateral = float(
            np.median(lateral_values)
        )

        sides = [
            self.side_from_lateral(value)
            for value in lateral_values
        ]

        non_center_sides = [
            side
            for side in sides
            if side != 'CENTER'
        ]

        if non_center_sides:
            left_count = non_center_sides.count(
                'LEFT'
            )
            right_count = non_center_sides.count(
                'RIGHT'
            )

            if (
                max(left_count, right_count)
                < self.required_hits
            ):
                return None

        return PigDetection(
            confidence=float(
                np.median(
                    [
                        item.confidence
                        for item in detections
                    ]
                )
            ),
            forward_x_m=float(
                np.median(
                    [
                        item.forward_x_m
                        for item in detections
                    ]
                )
            ),
            lateral_y_m=median_lateral,
            vertical_z_m=float(
                np.median(
                    [
                        item.vertical_z_m
                        for item in detections
                    ]
                )
            ),
            depth_m=float(
                np.median(
                    [
                        item.depth_m
                        for item in detections
                    ]
                )
            ),
            x1=float(
                np.median(
                    [item.x1 for item in detections]
                )
            ),
            y1=float(
                np.median(
                    [item.y1 for item in detections]
                )
            ),
            x2=float(
                np.median(
                    [item.x2 for item in detections]
                )
            ),
            y2=float(
                np.median(
                    [item.y2 for item in detections]
                )
            ),
        )

    def side_from_lateral(
        self,
        lateral_y_m: float,
    ) -> str:
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
        raw_detection: Optional[PigDetection],
        stable_detection: Optional[PigDetection],
    ) -> None:
        detected_message = Bool()
        detected_message.data = ( stable_detection is not None)
        self.detected_publisher.publish(detected_message )

        side_message = String()

        if stable_detection is None:
            side_message.data = 'NONE'
        else:
            side_message.data = (
                self.side_from_lateral(
                    stable_detection.lateral_y_m
                )
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
            ]
        else:
            side = self.side_from_lateral(
                output_detection.lateral_y_m
            )

            message.data = [
                1.0
                if stable_detection is not None
                else 0.0,
                1.0
                if raw_detection is not None
                else 0.0,
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

    def publish_debug(
        self,
        image: np.ndarray,
        raw_detection: Optional[PigDetection],
        stable_detection: Optional[PigDetection],
    ) -> None:
        debug_image = image.copy()

        detection = (
            stable_detection
            if stable_detection is not None
            else raw_detection
        )

        if detection is not None:
            color = (
                (0, 255, 0)
                if stable_detection is not None
                else (0, 165, 255)
            )

            cv2.rectangle(
                debug_image,
                (
                    int(detection.x1),
                    int(detection.y1),
                ),
                (
                    int(detection.x2),
                    int(detection.y2),
                ),
                color,
                3,
            )

            side = self.side_from_lateral(
                detection.lateral_y_m
            )

            text = (
                f'pig {detection.confidence:.2f} '
                f'x={detection.forward_x_m:.2f}m '
                f'y={detection.lateral_y_m:.2f}m '
                f'{side}'
            )

            cv2.putText(
                debug_image,
                text,
                (
                    int(detection.x1),
                    max(30, int(detection.y1) - 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            cv2.putText(
                debug_image,
                'NO PIG',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        debug_message = (
            self.bridge.cv2_to_imgmsg(
                debug_image,
                encoding='bgr8',
            )
        )

        debug_message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        debug_message.header.frame_id = (
            'camera_color_optical_frame'
        )

        self.debug_publisher.publish(
            debug_message
        )

    def log_result(
        self,
        raw_detection: Optional[PigDetection],
        stable_detection: Optional[PigDetection],
        speed,
    ) -> None:
        now = time.monotonic()

        if now - self.last_log_time < 1.0:
            return

        self.last_log_time = now

        inference_ms = float(
            speed.get('inference', 0.0)
        )

        if stable_detection is not None:
            side = self.side_from_lateral(
                stable_detection.lateral_y_m
            )

            self.get_logger().info(
                'STABLE pig: '
                f'conf={stable_detection.confidence:.3f}, '
                f'x={stable_detection.forward_x_m:.3f} m, '
                f'y={stable_detection.lateral_y_m:.3f} m, '
                f'z={stable_detection.vertical_z_m:.3f} m, '
                f'side={side}, '
                f'inference={inference_ms:.1f} ms'
            )
        elif raw_detection is not None:
            self.get_logger().info(
                'RAW pig，等待多幀確認：'
                f'conf={raw_detection.confidence:.3f}, '
                f'x={raw_detection.forward_x_m:.3f} m, '
                f'y={raw_detection.lateral_y_m:.3f} m, '
                f'inference={inference_ms:.1f} ms'
            )
        else:
            self.get_logger().info(
                'No pig，'
                f'inference={inference_ms:.1f} ms'
            )

    def timeout_callback(self) -> None:
        if self.last_sensor_time <= 0.0:
            return

        elapsed = (
            time.monotonic()
            - self.last_sensor_time
        )

        if elapsed <= self.sensor_timeout_sec:
            return

        if self.sensor_timeout_published:
            return

        self.sensor_timeout_published = True
        self.history.clear()
        self.publish_result(None, None)

        self.get_logger().error(
            'D435同步影像逾時，'
            'pig_detected已強制設為False'
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