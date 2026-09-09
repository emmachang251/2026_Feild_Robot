#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class LevelAWallSafety(Node):
    """
    第一关隔板安全监督节点。

    订阅：
        /level_a/cmd_vel_raw
        /level_a/lane_geometry

    发布：
        /cmd_vel
        /level_a/emergency_stop
        /level_a/safety_status

    注意：
        目前只检查LiDAR隔板资料。
        LiDAR看不到猪，因此之后必须加入D435猪只安全判断。
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

        self.heading_bias = float(
            self.get_parameter(
                'heading_bias_rad'
         ).value
        )

        if not (
            self.hard_clearance
            < self.turn_block_clearance
            < self.warning_clearance
        ):
            raise ValueError(
             '距离必须满足：'
             'hard_clearance '
             '< turn_block_clearance '
                '< warning_clearance'
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

        control_rate = float(
            self.get_parameter('control_rate_hz').value
        )

        # 最近一次原始速度命令
        self.raw_linear_x = 0.0
        self.raw_angular_z = 0.0
        self.last_command_time = None

        # 最近一次墙面资料
        self.geometry_valid = False
        self.left_wall_valid = False
        self.right_wall_valid = False

        self.left_wall_distance = math.nan
        self.right_wall_distance = math.nan
        self.heading_error = math.nan

        self.last_lane_time = None

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
            f'lane={self.lane_geometry_topic}'
        )

    def raw_command_callback(self, msg: Twist):
        """储存上游控制器要求的速度。"""

        self.raw_linear_x = float(msg.linear.x)
        self.raw_angular_z = float(msg.angular.z)

        self.last_command_time = (
            self.get_clock().now()
        )

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
        """安全监督主循环。"""

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
            'SAFE: '
            f'left_clearance={left_clearance:.3f} m, '
            f'right_clearance={right_clearance:.3f} m, '
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