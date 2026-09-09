#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


@dataclass
class WallFit:
    """單側隔板直線擬合結果。"""

    valid: bool = False
    slope: float = math.nan
    intercept: float = math.nan
    distance: float = math.nan
    heading: float = math.nan
    rmse: float = math.nan
    inlier_count: int = 0
    x_span: float = 0.0


class LaneWallDetector(Node):
    """
    使用2D LiDAR擬合走道左右兩側隔板。

    車體座標定義：
        +x：車頭方向
        +y：車體左側
        yaw正值：逆時針

    目前LiDAR相對車體中心：
        x = +0.1375 m
        y = 0.0 m
        yaw = 0.0 rad

    訂閱：
        /scan

    發布：
        /level_a/lane_geometry

    Float32MultiArray欄位：
        [0]  geometry_valid
        [1]  left_wall_valid
        [2]  right_wall_valid
        [3]  left_wall_distance_m
        [4]  right_wall_distance_m
        [5]  measured_lane_width_m
        [6]  lane_heading_error_rad
        [7]  lane_width_valid
        [8]  left_wall_rmse_m
        [9]  right_wall_rmse_m
        [10] left_wall_inlier_count
        [11] right_wall_inlier_count
    """

    def __init__(self):
        super().__init__('lane_wall_detector')

        # ROS Topic
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter(
            'output_topic',
            '/level_a/lane_geometry'
        )

        # 競賽走道寬度
        self.declare_parameter(
            'expected_lane_width_m',
            0.75
        )
        self.declare_parameter(
            'lane_width_tolerance_m',
            0.15
        )

        # LiDAR雷射中心相對車體中心base_link的位置
        self.declare_parameter(
            'lidar_x_offset_m',
            0.1375
        )
        self.declare_parameter(
            'lidar_y_offset_m',
            0.0
        )
        self.declare_parameter(
            'lidar_yaw_offset_rad',
            0.0
        )

        # 隔板點雲搜索範圍
        self.declare_parameter(
            'roi_x_min_m',
            0.02
        )
        self.declare_parameter(
            'roi_x_max_m',
            2.00
        )
        self.declare_parameter(
            'roi_min_abs_y_m',
            0.12
        )
        self.declare_parameter(
            'roi_max_abs_y_m',
            0.80
        )
        self.declare_parameter(
            'max_scan_range_m',
            4.0
        )

        # 牆面擬合參數
        self.declare_parameter(
            'minimum_wall_points',
            8
        )
        self.declare_parameter(
            'minimum_wall_x_span_m',
            0.25
        )
        self.declare_parameter(
            'line_residual_threshold_m',
            0.035
        )
        self.declare_parameter(
            'maximum_wall_rmse_m',
            0.050
        )
        self.declare_parameter(
            'maximum_heading_deg',
            35.0
        )

        # 合理牆面距離
        self.declare_parameter(
            'minimum_wall_distance_m',
            0.10
        )
        self.declare_parameter(
            'maximum_wall_distance_m',
            0.80
        )

        # 輸出低通濾波
        self.declare_parameter(
            'ema_alpha',
            0.25
        )

        self.scan_topic = str(
            self.get_parameter('scan_topic').value
        )
        self.output_topic = str(
            self.get_parameter('output_topic').value
        )

        self.expected_lane_width = float(
            self.get_parameter(
                'expected_lane_width_m'
            ).value
        )
        self.lane_width_tolerance = float(
            self.get_parameter(
                'lane_width_tolerance_m'
            ).value
        )

        self.lidar_x_offset = float(
            self.get_parameter(
                'lidar_x_offset_m'
            ).value
        )
        self.lidar_y_offset = float(
            self.get_parameter(
                'lidar_y_offset_m'
            ).value
        )
        self.lidar_yaw_offset = float(
            self.get_parameter(
                'lidar_yaw_offset_rad'
            ).value
        )

        self.roi_x_min = float(
            self.get_parameter('roi_x_min_m').value
        )
        self.roi_x_max = float(
            self.get_parameter('roi_x_max_m').value
        )
        self.roi_min_abs_y = float(
            self.get_parameter(
                'roi_min_abs_y_m'
            ).value
        )
        self.roi_max_abs_y = float(
            self.get_parameter(
                'roi_max_abs_y_m'
            ).value
        )
        self.max_scan_range = float(
            self.get_parameter(
                'max_scan_range_m'
            ).value
        )

        self.minimum_wall_points = int(
            self.get_parameter(
                'minimum_wall_points'
            ).value
        )
        self.minimum_wall_x_span = float(
            self.get_parameter(
                'minimum_wall_x_span_m'
            ).value
        )
        self.line_residual_threshold = float(
            self.get_parameter(
                'line_residual_threshold_m'
            ).value
        )
        self.maximum_wall_rmse = float(
            self.get_parameter(
                'maximum_wall_rmse_m'
            ).value
        )
        self.maximum_heading = math.radians(
            float(
                self.get_parameter(
                    'maximum_heading_deg'
                ).value
            )
        )

        self.minimum_wall_distance = float(
            self.get_parameter(
                'minimum_wall_distance_m'
            ).value
        )
        self.maximum_wall_distance = float(
            self.get_parameter(
                'maximum_wall_distance_m'
            ).value
        )
        self.ema_alpha = float(
            self.get_parameter('ema_alpha').value
        )

        # 濾波後數值
        self.left_distance_filtered: Optional[float] = None
        self.right_distance_filtered: Optional[float] = None
        self.heading_filtered: Optional[float] = None

        self.last_log_time_ns = 0

        self.geometry_pub = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            'LaneWallDetector started. '
            f'Input={self.scan_topic}, '
            f'output={self.output_topic}, '
            f'lidar offset='
            f'({self.lidar_x_offset:.4f}, '
            f'{self.lidar_y_offset:.4f}, '
            f'{self.lidar_yaw_offset:.4f})'
        )

    def ema(
        self,
        previous: Optional[float],
        current: float
    ) -> float:
        """指數移動平均，降低LiDAR輸出抖動。"""

        if previous is None:
            return current

        if not math.isfinite(previous):
            return current

        return (
            self.ema_alpha * current
            + (1.0 - self.ema_alpha) * previous
        )

    def fit_wall(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray,
        expected_side: str
    ) -> WallFit:
        """
        將單側隔板擬合為：

            y = slope * x + intercept

        使用多次殘差剔除減少離群點影響。
        """

        result = WallFit()

        if x_points.size < self.minimum_wall_points:
            return result

        keep = np.ones(
            x_points.shape,
            dtype=bool
        )

        slope = math.nan
        intercept = math.nan

        for _ in range(4):
            if (
                np.count_nonzero(keep)
                < self.minimum_wall_points
            ):
                return result

            try:
                slope, intercept = np.polyfit(
                    x_points[keep],
                    y_points[keep],
                    1
                )
            except (
                TypeError,
                ValueError,
                np.linalg.LinAlgError
            ):
                return result

            predicted_y = slope * x_points + intercept
            residual = np.abs(
                y_points - predicted_y
            )

            new_keep = (
                residual
                <= self.line_residual_threshold
            )

            if (
                np.count_nonzero(new_keep)
                < self.minimum_wall_points
            ):
                return result

            if np.array_equal(new_keep, keep):
                keep = new_keep
                break

            keep = new_keep

        inlier_x = x_points[keep]
        inlier_y = y_points[keep]

        if (
            inlier_x.size
            < self.minimum_wall_points
        ):
            return result

        try:
            slope, intercept = np.polyfit(
                inlier_x,
                inlier_y,
                1
            )
        except (
            TypeError,
            ValueError,
            np.linalg.LinAlgError
        ):
            return result

        predicted_y = slope * inlier_x + intercept

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        inlier_y - predicted_y
                    )
                )
            )
        )

        x_span = float(np.ptp(inlier_x))
        heading = math.atan(float(slope))

        # 車體中心原點到隔板直線的垂直距離
        perpendicular_distance = (
            abs(float(intercept))
            / math.sqrt(
                1.0 + float(slope) ** 2
            )
        )

        if expected_side == 'left':
            side_is_correct = intercept > 0.0
        else:
            side_is_correct = intercept < 0.0

        valid = (
            side_is_correct
            and x_span >= self.minimum_wall_x_span
            and rmse <= self.maximum_wall_rmse
            and abs(heading) <= self.maximum_heading
            and (
                self.minimum_wall_distance
                <= perpendicular_distance
                <= self.maximum_wall_distance
            )
        )

        return WallFit(
            valid=valid,
            slope=float(slope),
            intercept=float(intercept),
            distance=perpendicular_distance,
            heading=heading,
            rmse=rmse,
            inlier_count=int(inlier_x.size),
            x_span=x_span
        )

    def scan_callback(self, msg: LaserScan):
        """處理每一筆LaserScan。"""

        ranges = np.asarray(
            msg.ranges,
            dtype=np.float64
        )

        if ranges.size == 0:
            self.publish_invalid()
            return

        indices = np.arange(
            ranges.size,
            dtype=np.float64
        )

        angles = (
            float(msg.angle_min)
            + indices * float(msg.angle_increment)
        )

        valid_range = (
            np.isfinite(ranges)
            & (ranges >= float(msg.range_min))
            & (ranges <= float(msg.range_max))
            & (ranges <= self.max_scan_range)
        )

        if (
            np.count_nonzero(valid_range)
            < 2 * self.minimum_wall_points
        ):
            self.publish_invalid()
            return

        ranges = ranges[valid_range]
        angles = angles[valid_range]

        # LiDAR自身座標
        x_lidar = ranges * np.cos(angles)
        y_lidar = ranges * np.sin(angles)

        # 將LiDAR點轉換到車體中心base_link
        cos_yaw = math.cos(
            self.lidar_yaw_offset
        )
        sin_yaw = math.sin(
            self.lidar_yaw_offset
        )

        x_base = (
            cos_yaw * x_lidar
            - sin_yaw * y_lidar
            + self.lidar_x_offset
        )

        y_base = (
            sin_yaw * x_lidar
            + cos_yaw * y_lidar
            + self.lidar_y_offset
        )

        # 只使用車體前方、可能屬於隔板的點
        common_roi = (
            (x_base >= self.roi_x_min)
            & (x_base <= self.roi_x_max)
            & (
                np.abs(y_base)
                >= self.roi_min_abs_y
            )
            & (
                np.abs(y_base)
                <= self.roi_max_abs_y
            )
        )

        left_mask = (
            common_roi
            & (y_base > 0.0)
        )

        right_mask = (
            common_roi
            & (y_base < 0.0)
        )

        left_fit = self.fit_wall(
            x_base[left_mask],
            y_base[left_mask],
            'left'
        )

        right_fit = self.fit_wall(
            x_base[right_mask],
            y_base[right_mask],
            'right'
        )

        left_distance = math.nan
        right_distance = math.nan
        heading = math.nan

        if left_fit.valid:
            self.left_distance_filtered = self.ema(
                self.left_distance_filtered,
                left_fit.distance
            )
            left_distance = (
                self.left_distance_filtered
            )

        if right_fit.valid:
            self.right_distance_filtered = self.ema(
                self.right_distance_filtered,
                right_fit.distance
            )
            right_distance = (
                self.right_distance_filtered
            )

        valid_headings = []

        if left_fit.valid:
            valid_headings.append(
                left_fit.heading
            )

        if right_fit.valid:
            valid_headings.append(
                right_fit.heading
            )

        if valid_headings:
            raw_heading = float(
                np.mean(valid_headings)
            )

            self.heading_filtered = self.ema(
                self.heading_filtered,
                raw_heading
            )

            heading = self.heading_filtered

        lane_width = math.nan
        lane_width_valid = False

        if left_fit.valid and right_fit.valid:
            lane_width = (
                left_distance
                + right_distance
            )

            lane_width_valid = (
                abs(
                    lane_width
                    - self.expected_lane_width
                )
                <= self.lane_width_tolerance
            )

        # 一側有效時仍可提供牆面跟隨資訊；
        # 兩側都有時，走道寬度還必須合理。
        geometry_valid = (
            left_fit.valid
            or right_fit.valid
        )

        if left_fit.valid and right_fit.valid:
            geometry_valid = (
                geometry_valid
                and lane_width_valid
            )

        output = Float32MultiArray()

        output.data = [
            1.0 if geometry_valid else 0.0,
            1.0 if left_fit.valid else 0.0,
            1.0 if right_fit.valid else 0.0,
            float(left_distance),
            float(right_distance),
            float(lane_width),
            float(heading),
            1.0 if lane_width_valid else 0.0,
            float(left_fit.rmse),
            float(right_fit.rmse),
            float(left_fit.inlier_count),
            float(right_fit.inlier_count)
        ]

        self.geometry_pub.publish(output)

        self.log_status(
            geometry_valid=geometry_valid,
            left_fit=left_fit,
            right_fit=right_fit,
            left_distance=left_distance,
            right_distance=right_distance,
            lane_width=lane_width,
            heading=heading
        )

    def publish_invalid(self):
        """LiDAR資料不足時發布無效結果。"""

        output = Float32MultiArray()

        output.data = [
            0.0,
            0.0,
            0.0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0.0,
            math.nan,
            math.nan,
            0.0,
            0.0
        ]

        self.geometry_pub.publish(output)

    def log_status(
        self,
        geometry_valid: bool,
        left_fit: WallFit,
        right_fit: WallFit,
        left_distance: float,
        right_distance: float,
        lane_width: float,
        heading: float
    ):
        """每兩秒顯示一次目前擬合結果。"""

        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        if (
            now_ns - self.last_log_time_ns
            < 2_000_000_000
        ):
            return

        self.last_log_time_ns = now_ns

        left_text = (
            f'{left_distance:.3f} m'
            if math.isfinite(left_distance)
            else 'invalid'
        )

        right_text = (
            f'{right_distance:.3f} m'
            if math.isfinite(right_distance)
            else 'invalid'
        )

        width_text = (
            f'{lane_width:.3f} m'
            if math.isfinite(lane_width)
            else 'invalid'
        )

        heading_text = (
            f'{math.degrees(heading):.2f} deg'
            if math.isfinite(heading)
            else 'invalid'
        )

        self.get_logger().info(
            f'valid={geometry_valid}, '
            f'left={left_text} '
            f'({left_fit.inlier_count} pts), '
            f'right={right_text} '
            f'({right_fit.inlier_count} pts), '
            f'width={width_text}, '
            f'heading={heading_text}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = LaneWallDetector()

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