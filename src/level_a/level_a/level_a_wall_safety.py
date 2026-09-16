#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class LevelAWallSafety(Node):
    """
    Level A mode-aware safety supervisor.

    订阅：
        /level_a/cmd_vel_raw
        /level_a/lane_geometry
        /level_a/motion_mode
        /scan

    发布：
        /cmd_vel
        /level_a/emergency_stop
        /level_a/safety_status

    LANE模式使用擬合後的走道牆面淨空；U_TURN模式不要求兩側
    走道同時存在，改用原始LaserScan計算車體矩形外圍淨空。
    """

    def __init__(self):
        super().__init__('level_a_wall_safety')

        # Topic
        self.declare_parameter(
            'raw_cmd_topic',
            '/level_a/cmd_vel_raw'
        )
        self.declare_parameter(
            'safe_cmd_topic',
            '/cmd_vel'
        )
        self.declare_parameter(
            'lane_geometry_topic',
            '/level_a/lane_geometry'
        )
        self.declare_parameter(
            'motion_mode_topic',
            '/level_a/motion_mode'
        )
        self.declare_parameter(
            'scan_topic',
            '/scan'
        )

        # 车体尺寸
        self.declare_parameter(
            'vehicle_width_m',
            0.40
        )
        self.declare_parameter(
            'vehicle_total_length_m',
            0.43
        )

        # 墙面净空门槛
        self.declare_parameter(
            'hard_clearance_m',
            0.010
        )
        self.declare_parameter(
            'warning_clearance_m',
            0.050
        )
        # 低於20 mm時，才禁止繼續朝牆面轉向
        self.declare_parameter(
            'turn_block_clearance_m',
            0.020
        )

        # U彎時的原始LiDAR安全包絡。scan點會先轉換到base_link，
        # 再計算它與車體矩形外框之間的最短距離。
        self.declare_parameter(
            'u_turn_hard_clearance_m',
            0.030
        )
        self.declare_parameter(
            'u_turn_warning_clearance_m',
            0.100
        )
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
        self.declare_parameter(
            'self_filter_inset_m',
            0.015
        )
        self.declare_parameter(
            'maximum_safety_scan_range_m',
            2.0
        )
        self.declare_parameter(
            'minimum_scan_points',
            8
        )

        # 與lane_change_controller使用相同的航向零點補償
        self.declare_parameter(
            'heading_bias_rad',
            0.050
        )

        # 速度上限
        self.declare_parameter(
            'maximum_forward_speed_mps',
            0.20
        )
        self.declare_parameter(
            'maximum_reverse_speed_mps',
            0.0
        )
        self.declare_parameter(
            'maximum_angular_speed_radps',
            0.80
        )

        # 资料逾时
        self.declare_parameter(
            'lane_timeout_s',
            0.50
        )
        self.declare_parameter(
            'command_timeout_s',
            0.30
        )
        self.declare_parameter(
            'mode_timeout_s',
            0.75
        )
        self.declare_parameter(
            'scan_timeout_s',
            0.50
        )

        # 发布频率
        self.declare_parameter(
            'control_rate_hz',
            20.0
        )

        self.raw_cmd_topic = str(
            self.get_parameter('raw_cmd_topic').value
        )
        self.safe_cmd_topic = str(
            self.get_parameter('safe_cmd_topic').value
        )
        self.lane_geometry_topic = str(
            self.get_parameter(
                'lane_geometry_topic'
            ).value
        )
        self.motion_mode_topic = str(
            self.get_parameter('motion_mode_topic').value
        )
        self.scan_topic = str(
            self.get_parameter('scan_topic').value
        )

        self.vehicle_width = float(
            self.get_parameter('vehicle_width_m').value
        )
        self.vehicle_total_length = float(
            self.get_parameter(
                'vehicle_total_length_m'
            ).value
        )

        self.hard_clearance = float(
            self.get_parameter('hard_clearance_m').value
        )
        self.warning_clearance = float(
            self.get_parameter(
                'warning_clearance_m'
            ).value
        )

        self.turn_block_clearance = float(
            self.get_parameter(
             'turn_block_clearance_m'
            ).value
        )

        self.u_turn_hard_clearance = float(
            self.get_parameter('u_turn_hard_clearance_m').value
        )
        self.u_turn_warning_clearance = float(
            self.get_parameter('u_turn_warning_clearance_m').value
        )
        self.lidar_x_offset = float(
            self.get_parameter('lidar_x_offset_m').value
        )
        self.lidar_y_offset = float(
            self.get_parameter('lidar_y_offset_m').value
        )
        self.lidar_yaw_offset = float(
            self.get_parameter('lidar_yaw_offset_rad').value
        )
        self.self_filter_inset = float(
            self.get_parameter('self_filter_inset_m').value
        )
        self.maximum_safety_scan_range = float(
            self.get_parameter('maximum_safety_scan_range_m').value
        )
        self.minimum_scan_points = int(
            self.get_parameter('minimum_scan_points').value
        )

        self.heading_bias = float(
            self.get_parameter(
                'heading_bias_rad'
         ).value
        )

        self.maximum_forward_speed = float(
            self.get_parameter(
                'maximum_forward_speed_mps'
            ).value
        )
        self.maximum_reverse_speed = float(
            self.get_parameter(
                'maximum_reverse_speed_mps'
            ).value
        )
        self.maximum_angular_speed = float(
            self.get_parameter(
                'maximum_angular_speed_radps'
            ).value
        )

        self.lane_timeout = float(
            self.get_parameter('lane_timeout_s').value
        )
        self.command_timeout = float(
            self.get_parameter(
                'command_timeout_s'
            ).value
        )
        self.mode_timeout = float(
            self.get_parameter('mode_timeout_s').value
        )
        self.scan_timeout = float(
            self.get_parameter('scan_timeout_s').value
        )

        control_rate = float(
            self.get_parameter('control_rate_hz').value
        )

        self.validate_parameters(control_rate)

        # 最近一次原始速度命令
        self.raw_linear_x = 0.0
        self.raw_angular_z = 0.0
        self.raw_command_valid = False
        self.last_command_time = None

        # motion_mode必須由任務管理器持續發布heartbeat。
        self.motion_mode = 'STOP'
        self.last_mode_time = None

        # 最近一次墙面资料
        self.geometry_valid = False
        self.left_wall_valid = False
        self.right_wall_valid = False

        self.left_wall_distance = math.nan
        self.right_wall_distance = math.nan
        self.heading_error = math.nan

        self.last_lane_time = None

        # U彎期間直接由LaserScan計算車體外框淨空。
        self.minimum_scan_clearance = math.inf
        self.valid_scan_points = 0
        self.scan_valid = False
        self.last_scan_time = None

        self.last_status: Optional[str] = None

        self.safe_cmd_pub = self.create_publisher(
            Twist,
            self.safe_cmd_topic,
            10
        )

        self.emergency_pub = self.create_publisher(
            Bool,
            '/level_a/emergency_stop',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/level_a/safety_status',
            10
        )

        self.raw_cmd_sub = self.create_subscription(
            Twist,
            self.raw_cmd_topic,
            self.raw_command_callback,
            10
        )

        self.lane_sub = self.create_subscription(
            Float32MultiArray,
            self.lane_geometry_topic,
            self.lane_geometry_callback,
            10
        )

        self.mode_sub = self.create_subscription(
            String,
            self.motion_mode_topic,
            self.motion_mode_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        timer_period = 1.0 / max(
            control_rate,
            1.0
        )

        self.control_timer = self.create_timer(
            timer_period,
            self.control_callback
        )

        self.get_logger().info(
            'Level A wall safety started. '
            f'Input command={self.raw_cmd_topic}, '
            f'output command={self.safe_cmd_topic}, '
            f'lane={self.lane_geometry_topic}, '
            f'scan={self.scan_topic}, '
            f'mode={self.motion_mode_topic}'
        )

    def validate_parameters(self, control_rate: float) -> None:
        if not (
            math.isfinite(self.hard_clearance)
            and math.isfinite(self.turn_block_clearance)
            and math.isfinite(self.warning_clearance)
            and 0.0 <= self.hard_clearance
            < self.turn_block_clearance
            < self.warning_clearance
        ):
            raise ValueError(
                'lane clearances must satisfy 0 <= hard < turn_block < warning'
            )
        if not (
            math.isfinite(self.u_turn_hard_clearance)
            and math.isfinite(self.u_turn_warning_clearance)
            and 0.0 <= self.u_turn_hard_clearance
            < self.u_turn_warning_clearance
        ):
            raise ValueError(
                'U-turn clearances must satisfy 0 <= hard < warning'
            )

        positive_values = {
            'vehicle_width_m': self.vehicle_width,
            'vehicle_total_length_m': self.vehicle_total_length,
            'maximum_forward_speed_mps': self.maximum_forward_speed,
            'maximum_angular_speed_radps': self.maximum_angular_speed,
            'lane_timeout_s': self.lane_timeout,
            'command_timeout_s': self.command_timeout,
            'mode_timeout_s': self.mode_timeout,
            'scan_timeout_s': self.scan_timeout,
            'maximum_safety_scan_range_m': self.maximum_safety_scan_range,
            'control_rate_hz': control_rate,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and greater than zero')
        if (
            not math.isfinite(self.maximum_reverse_speed)
            or self.maximum_reverse_speed < 0.0
        ):
            raise ValueError(
                'maximum_reverse_speed_mps must be finite and non-negative'
            )
        if not math.isfinite(self.self_filter_inset) or self.self_filter_inset < 0.0:
            raise ValueError('self_filter_inset_m must be finite and non-negative')
        if self.self_filter_inset >= min(
            self.vehicle_width,
            self.vehicle_total_length,
        ) / 2.0:
            raise ValueError('self_filter_inset_m is too large for the vehicle')
        if self.minimum_scan_points < 1:
            raise ValueError('minimum_scan_points must be at least one')

    def raw_command_callback(self, msg: Twist):
        """储存上游控制器要求的速度。"""

        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        self.raw_command_valid = all(
            math.isfinite(value)
            for value in (
                linear_x,
                float(msg.linear.y),
                float(msg.linear.z),
                float(msg.angular.x),
                float(msg.angular.y),
                angular_z,
            )
        )

        if self.raw_command_valid:
            self.raw_linear_x = linear_x
            self.raw_angular_z = angular_z
        else:
            self.raw_linear_x = 0.0
            self.raw_angular_z = 0.0

        self.last_command_time = (
            self.get_clock().now()
        )

    def motion_mode_callback(self, msg: String):
        """Store a validated mode heartbeat from the mission supervisor."""

        requested_mode = msg.data.strip().upper()
        if requested_mode == 'UTURN':
            requested_mode = 'U_TURN'
        if requested_mode not in {'STOP', 'LANE', 'U_TURN'}:
            self.get_logger().warning(
                f'Ignoring invalid motion mode: {requested_mode}'
            )
            return

        self.motion_mode = requested_mode
        self.last_mode_time = self.get_clock().now()

    def scan_callback(self, msg: LaserScan):
        """Measure minimum clearance from scan points to the robot rectangle."""

        half_length = self.vehicle_total_length / 2.0
        half_width = self.vehicle_width / 2.0
        # Deliberately shrink, rather than expand, the ignored self-return
        # box.  Expanding it would hide a real obstacle just outside the
        # vehicle and defeat the hard-clearance threshold.
        filter_half_length = max(0.0, half_length - self.self_filter_inset)
        filter_half_width = max(0.0, half_width - self.self_filter_inset)

        cos_offset = math.cos(self.lidar_yaw_offset)
        sin_offset = math.sin(self.lidar_yaw_offset)
        minimum_clearance = math.inf
        valid_points = 0

        angle = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        range_min = float(msg.range_min)
        range_max = min(
            float(msg.range_max),
            self.maximum_safety_scan_range,
        )

        for raw_range in msg.ranges:
            measured_range = float(raw_range)
            if (
                math.isfinite(measured_range)
                and range_min <= measured_range <= range_max
            ):
                x_lidar = measured_range * math.cos(angle)
                y_lidar = measured_range * math.sin(angle)
                x_base = (
                    cos_offset * x_lidar
                    - sin_offset * y_lidar
                    + self.lidar_x_offset
                )
                y_base = (
                    sin_offset * x_lidar
                    + cos_offset * y_lidar
                    + self.lidar_y_offset
                )

                # Ignore returns from the vehicle itself.  Obstacles outside
                # this box are measured from the true vehicle footprint.
                if not (
                    abs(x_base) <= filter_half_length
                    and abs(y_base) <= filter_half_width
                ):
                    dx = max(abs(x_base) - half_length, 0.0)
                    dy = max(abs(y_base) - half_width, 0.0)
                    clearance = math.hypot(dx, dy)
                    minimum_clearance = min(
                        minimum_clearance,
                        clearance,
                    )
                    valid_points += 1

            angle += angle_increment

        self.minimum_scan_clearance = minimum_clearance
        self.valid_scan_points = valid_points
        self.scan_valid = (
            valid_points >= self.minimum_scan_points
            and math.isfinite(minimum_clearance)
        )
        self.last_scan_time = self.get_clock().now()

    def lane_geometry_callback(
        self,
        msg: Float32MultiArray
    ):
        """
        lane_wall_detector输出：

        [0] geometry_valid
        [1] left_wall_valid
        [2] right_wall_valid
        [3] left_wall_distance
        [4] right_wall_distance
        [5] lane_width
        [6] heading_error
        """

        if len(msg.data) < 12:
            self.geometry_valid = False
            self.last_lane_time = (
                self.get_clock().now()
            )
            return

        self.geometry_valid = (
            msg.data[0] > 0.5
        )
        self.left_wall_valid = (
            msg.data[1] > 0.5
        )
        self.right_wall_valid = (
            msg.data[2] > 0.5
        )

        self.left_wall_distance = float(
            msg.data[3]
        )
        self.right_wall_distance = float(
            msg.data[4]
        )
        self.heading_error = float(
            msg.data[6]
        )

        self.last_lane_time = (
            self.get_clock().now()
        )

    def elapsed_seconds(self, previous_time) -> float:
        """计算ROS时间差，单位秒。"""

        if previous_time is None:
            return math.inf

        duration = (
            self.get_clock().now()
            - previous_time
        )

        return (
            duration.nanoseconds
            / 1_000_000_000.0
        )

    @staticmethod
    def clamp(
        value: float,
        minimum: float,
        maximum: float
    ) -> float:
        return max(
            minimum,
            min(maximum, value)
        )

    def publish_stop(
        self,
        reason: str,
        emergency: bool = True
    ):
        """发布零速度及安全状态。"""

        stop_command = Twist()
        self.safe_cmd_pub.publish(stop_command)

        emergency_msg = Bool()
        emergency_msg.data = emergency
        self.emergency_pub.publish(emergency_msg)

        self.publish_status(reason)

    def publish_status(self, status: str):
        """状态改变时才显示日志，避免终端机洗版。"""

        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

        if status != self.last_status:
            self.last_status = status
            self.get_logger().info(status)

    def calculate_lateral_extent(
        self,
        heading_error: float
    ) -> float:
        """
        计算车体旋转后在横向所占的半宽。

        车体不是一个点，转斜后车头与车尾角落
        会比单纯vehicle_width / 2更靠近隔板。
        """

        half_width = (
            self.vehicle_width / 2.0
        )
        half_length = (
            self.vehicle_total_length / 2.0
        )

        return (
            half_width
            * abs(math.cos(heading_error))
            + half_length
            * abs(math.sin(heading_error))
        )

    def control_callback(self):
        """Select the safety policy for the active motion mode."""

        if self.last_mode_time is None:
            self.publish_stop(
                'STOP: waiting for motion mode'
            )
            return

        if self.elapsed_seconds(self.last_mode_time) > self.mode_timeout:
            self.publish_stop(
                'STOP: motion mode timeout'
            )
            return

        if self.motion_mode == 'STOP':
            self.publish_stop(
                'STOP: motion mode is STOP',
                emergency=False
            )
            return

        if self.last_command_time is None:
            self.publish_stop(
                'STOP: waiting for raw command',
                emergency=False
            )
            return

        if self.elapsed_seconds(self.last_command_time) > self.command_timeout:
            self.publish_stop(
                'STOP: raw command timeout',
                emergency=False
            )
            return

        if not self.raw_command_valid:
            self.publish_stop(
                'STOP: raw command invalid'
            )
            return

        if self.motion_mode == 'LANE':
            self.control_lane_mode()
            return

        if self.motion_mode == 'U_TURN':
            self.control_u_turn_mode()
            return

        self.publish_stop('STOP: unsupported motion mode')

    def control_lane_mode(self):
        """Apply the existing fitted-wall safety policy in an aisle."""

        # 没有收到墙面资料，保持停止
        if self.last_lane_time is None:
            self.publish_stop(
                'STOP: waiting for lane geometry'
            )
            return

        # 墙面资料逾时
        lane_age = self.elapsed_seconds(
            self.last_lane_time
        )

        if lane_age > self.lane_timeout:
            self.publish_stop(
                'STOP: lane geometry timeout'
            )
            return

        # 墙面几何无效
        if not self.geometry_valid:
            self.publish_stop(
                'STOP: lane geometry invalid'
            )
            return

        # 左右墙都无效
        if (
            not self.left_wall_valid
            and not self.right_wall_valid
        ):
            self.publish_stop(
                'STOP: both walls invalid'
            )
            return

        # 角度无效
        if not math.isfinite(
            self.heading_error
        ):
            self.publish_stop(
                'STOP: heading invalid'
            )
            return

        # 没有收到上游速度命令
        if self.last_command_time is None:
            self.publish_stop(
                'STOP: waiting for raw command',
                emergency=False
            )
            return

        # 上游速度命令逾时
        command_age = self.elapsed_seconds(
            self.last_command_time
        )

        if command_age > self.command_timeout:
            self.publish_stop(
                'STOP: raw command timeout',
                emergency=False
            )
            return

        corrected_heading_error = (
            self.heading_error
            - self.heading_bias
        )

        lateral_extent = (
            self.calculate_lateral_extent(
                corrected_heading_error
            )
        )

        left_clearance = math.inf
        right_clearance = math.inf

        if self.left_wall_valid:
            if not math.isfinite(
                self.left_wall_distance
            ):
                self.publish_stop(
                    'STOP: left wall distance invalid'
                )
                return

            left_clearance = (
                self.left_wall_distance
                - lateral_extent
            )

        if self.right_wall_valid:
            if not math.isfinite(
                self.right_wall_distance
            ):
                self.publish_stop(
                    'STOP: right wall distance invalid'
                )
                return

            right_clearance = (
                self.right_wall_distance
                - lateral_extent
            )

        minimum_clearance = min(
            left_clearance,
            right_clearance
        )

        # 车体外框已太靠近隔板
        if (
            minimum_clearance
            <= self.hard_clearance
        ):
            self.publish_stop(
                'STOP: hard wall clearance violation'
            )
            return

        linear_x = self.clamp(
            self.raw_linear_x,
            -self.maximum_reverse_speed,
            self.maximum_forward_speed
        )

        angular_z = self.clamp(
            self.raw_angular_z,
            -self.maximum_angular_speed,
            self.maximum_angular_speed
        )

        # 接近墙时降低前进速度
        speed_scale = 1.0

        if (
            minimum_clearance
            < self.warning_clearance
        ):
            denominator = (
                self.warning_clearance
                - self.hard_clearance
            )

            if denominator > 0.0:
                speed_scale = (
                    minimum_clearance
                    - self.hard_clearance
                ) / denominator
            else:
                speed_scale = 0.0

            speed_scale = self.clamp(
                speed_scale,
                0.0,
                1.0
            )

            linear_x *= speed_scale

        # 靠近左墙时，禁止继续向左转
        if (
            left_clearance
            < self.turn_block_clearance
            and angular_z > 0.0
        ):
            angular_z = 0.0

        # 靠近右墙时，禁止继续向右转
        if (
            right_clearance
            < self.turn_block_clearance
            and angular_z < 0.0
        ):
            angular_z = 0.0

        safe_command = Twist()
        safe_command.linear.x = linear_x
        safe_command.angular.z = angular_z

        self.safe_cmd_pub.publish(
            safe_command
        )

        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_pub.publish(
            emergency_msg
        )

        self.publish_status(
            'SAFE: mode=LANE '
            f'left_clearance={left_clearance:.3f} m, '
            f'right_clearance={right_clearance:.3f} m, '
            f'scale={speed_scale:.2f}'
        )

    def control_u_turn_mode(self):
        """Apply all-around LaserScan clearance protection during a U-turn."""

        if self.last_scan_time is None:
            self.publish_stop('STOP: waiting for safety scan')
            return

        if self.elapsed_seconds(self.last_scan_time) > self.scan_timeout:
            self.publish_stop('STOP: safety scan timeout')
            return

        if not self.scan_valid:
            self.publish_stop(
                'STOP: safety scan invalid '
                f'points={self.valid_scan_points}'
            )
            return

        clearance = self.minimum_scan_clearance
        if clearance <= self.u_turn_hard_clearance:
            self.publish_stop(
                'STOP: U-turn hard clearance violation '
                f'clearance={clearance:.3f} m'
            )
            return

        linear_x = self.clamp(
            self.raw_linear_x,
            -self.maximum_reverse_speed,
            self.maximum_forward_speed
        )
        angular_z = self.clamp(
            self.raw_angular_z,
            -self.maximum_angular_speed,
            self.maximum_angular_speed
        )

        speed_scale = 1.0
        if clearance < self.u_turn_warning_clearance:
            denominator = (
                self.u_turn_warning_clearance
                - self.u_turn_hard_clearance
            )
            speed_scale = self.clamp(
                (clearance - self.u_turn_hard_clearance) / denominator,
                0.0,
                1.0
            )
            linear_x *= speed_scale
            angular_z *= speed_scale

        safe_command = Twist()
        safe_command.linear.x = linear_x
        safe_command.angular.z = angular_z
        self.safe_cmd_pub.publish(safe_command)

        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_pub.publish(emergency_msg)

        self.publish_status(
            'SAFE: mode=U_TURN '
            f'clearance={clearance:.3f} m, '
            f'points={self.valid_scan_points}, '
            f'scale={speed_scale:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = LevelAWallSafety()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 结束前再发布一次停止命令
        node.publish_stop(
            'STOP: wall safety shutting down'
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
