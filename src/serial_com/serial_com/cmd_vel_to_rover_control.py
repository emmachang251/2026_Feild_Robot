#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from rover_interfaces.msg import RoverControl


class CmdVelToRoverControl(Node):
    """
    將 /cmd_vel 轉換成左右履帶的目標 RPM。

    RoverControl 中：
        direction:
            0 = 停止
            1 = 正轉
            2 = 反轉

        speed:
            以 RPM × 10 傳送

    例如：
        +21.4 RPM -> direction=1, speed=214
        -21.4 RPM -> direction=2, speed=214
    """

    def __init__(self) -> None:
        super().__init__('cmd_vel_to_rover_control')

        # 主動輪轉一圈，理論上拉動14 cm履帶
        self.declare_parameter(
            'left_track_advance_m',
            0.14
        )
        self.declare_parameter(
            'right_track_advance_m',
            0.14
        )

        # 左履帶中心線到右履帶中心線：22.4 cm
        self.declare_parameter(
            'track_width_m',
            0.355
        )
        # 第一輪只使用已驗證過的 ±30 RPM
        self.declare_parameter(
            'max_target_rpm',
            70.0
        )

        # 以10 Hz送往 uart_node
        self.declare_parameter(
            'publish_rate_hz',
            10.0
        )

        # 超過0.5秒沒有收到 /cmd_vel 就輸出停止
        self.declare_parameter(
            'cmd_vel_timeout_s',
            0.5
        )

        self.left_track_advance_m = float(
            self.get_parameter(
                'left_track_advance_m'
            ).value
        )

        self.right_track_advance_m = float(
            self.get_parameter(
                'right_track_advance_m'
            ).value
        )

        self.track_width_m = float(
            self.get_parameter(
                'track_width_m'
            ).value
        )

        self.max_target_rpm = float(
            self.get_parameter(
                'max_target_rpm'
            ).value
        )

        self.publish_rate_hz = float(
            self.get_parameter(
                'publish_rate_hz'
            ).value
        )

        self.cmd_vel_timeout_s = float(
            self.get_parameter(
                'cmd_vel_timeout_s'
            ).value
        )

        if self.left_track_advance_m <= 0.0:
            raise ValueError(
                'left_track_advance_m 必須大於0'
            )

        if self.right_track_advance_m <= 0.0:
            raise ValueError(
                'right_track_advance_m 必須大於0'
            )

        if self.track_width_m <= 0.0:
            raise ValueError(
                'track_width_m 必須大於0'
            )

        if not 0.0 < self.max_target_rpm <= 100.0:
            raise ValueError(
                'max_target_rpm 必須位於0～100'
            )

        if self.publish_rate_hz <= 0.0:
            raise ValueError(
                'publish_rate_hz 必須大於0'
            )

        if self.cmd_vel_timeout_s <= 0.0:
            raise ValueError(
                'cmd_vel_timeout_s 必須大於0'
            )

        self.control_pub = self.create_publisher(
            RoverControl,
            'chassis_control',
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # 保存最近一次計算出的左右目標RPM
        self.latest_left_rpm = 0.0
        self.latest_right_rpm = 0.0

        # 尚未收到任何 /cmd_vel
        self.last_cmd_vel_time = None

        # 固定以10 Hz發布 chassis_control
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_control
        )

        self.get_logger().info(
            'cmd_vel → 左右目標RPM節點已啟動；'
            f'track_width={self.track_width_m:.3f} m；'
            f'max_target_rpm={self.max_target_rpm:.1f}'
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        ROS 2：
            linear.x  = 車體線速度，m/s
            angular.z = 車體角速度，rad/s
        """

        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)

        # 履帶車逆運動學
        left_velocity = (
            linear_x
            - angular_z * self.track_width_m / 2.0
        )

        right_velocity = (
            linear_x
            + angular_z * self.track_width_m / 2.0
        )

        # 履帶線速度換算成主動輪RPM
        left_rpm = (
            60.0
            * left_velocity
            / self.left_track_advance_m
        )

        right_rpm = (
            60.0
            * right_velocity
            / self.right_track_advance_m
        )

        # 若任一側超過限制，左右等比例縮小，
        # 保留原本的轉彎比例
        largest_rpm = max(
            abs(left_rpm),
            abs(right_rpm)
        )

        if largest_rpm > self.max_target_rpm:
            scale = (
                self.max_target_rpm
                / largest_rpm
            )

            left_rpm *= scale
            right_rpm *= scale

        self.latest_left_rpm = left_rpm


        self.latest_right_rpm = right_rpm
        self.last_cmd_vel_time = self.get_clock().now()

    def rpm_to_direction(
        self,
        rpm: float
    ) -> tuple[int, int]:
        """
        將正負RPM轉成方向及 RPM × 10。
        """

        # 過小的命令視為停止
        if abs(rpm) < 0.05:
            return 0, 0

        if rpm > 0.0:
            direction = 1
        else:
            direction = 2

        rpm_x10 = int(
            round(abs(rpm) * 10.0)
        )

        return direction, rpm_x10

    def publish_control(self) -> None:
        """
        固定以10 Hz發布。

        若超過0.5秒沒有收到 /cmd_vel，
        就持續發布停止命令。
        """

        command_is_valid = False

        if self.last_cmd_vel_time is not None:
            elapsed_s = (
                self.get_clock().now()
                - self.last_cmd_vel_time
            ).nanoseconds / 1_000_000_000.0

            command_is_valid = (
                elapsed_s
                <= self.cmd_vel_timeout_s
            )

        if command_is_valid:
            left_rpm = self.latest_left_rpm
            right_rpm = self.latest_right_rpm
        else:
            left_rpm = 0.0
            right_rpm = 0.0

        left_dir, left_speed = (
            self.rpm_to_direction(left_rpm)
        )

        right_dir, right_speed = (
            self.rpm_to_direction(right_rpm)
        )

        output = RoverControl()

        output.left_dir = left_dir
        output.left_speed = left_speed
        output.right_dir = right_dir
        output.right_speed = right_speed

        output.servo_angles = [0, 0, 0, 0]
        output.led_states = [
            False,
            False,
            False,
            False
        ]

        self.control_pub.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = CmdVelToRoverControl()

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

