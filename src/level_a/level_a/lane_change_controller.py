#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class LaneChangeController(Node):
    """
    Level A 走道位置控制器。

    訂閱：
        /level_a/lane_geometry
        /level_a/target_side

    發布：
        /level_a/lane_cmd_vel
        /level_a/lane_change_status

    target_side 可使用：
        STOP
        CENTER
        LEFT
        RIGHT

    座標定義：
        y > 0：車體位於走道中心左側
        y < 0：車體位於走道中心右側
        angular.z > 0：車頭向左轉
    """

    VALID_TARGETS = {
        'STOP',
        'CENTER',
        'LEFT',
        'RIGHT',
    }

    def __init__(self) -> None:
        super().__init__('lane_change_controller')

        # Topic
        self.declare_parameter(
            'lane_geometry_topic',
            '/level_a/lane_geometry'
        )
        self.declare_parameter(
            'target_side_topic',
            '/level_a/target_side'
        )
        self.declare_parameter(
            'cmd_vel_raw_topic',
            '/level_a/lane_cmd_vel'
        )
        self.declare_parameter(
            'status_topic',
            '/level_a/lane_change_status'
        )

        # 控制頻率與資料逾時
        self.declare_parameter(
            'control_rate_hz',
            20.0
        )
        self.declare_parameter(
            'lane_timeout_s',
            0.5
        )

        # 初期落地測試使用低速
        self.declare_parameter(
            'center_speed_mps',
            0.060
        )
        self.declare_parameter(
            'side_speed_mps',
            0.055
        )
        self.declare_parameter(
            'minimum_speed_mps',
            0.025
        )

        # 車體中心到目標側牆面的距離。
        # 車寬0.40 m，設定0.25 m代表車體邊緣離牆約0.05 m。
        # 初期測試先用0.25，不直接使用競賽極限0.225。
        self.declare_parameter(
            'side_wall_distance_m',
            0.225
        )

        # 橫向位置與航向控制增益
        self.declare_parameter(
            'lateral_kp',
            2.8
        )
        self.declare_parameter(
            'heading_kp',
            1.4
        )
        self.declare_parameter(
            'max_angular_speed_rps',
            0.35
        )

        # 逐漸改變目標橫向位置，避免突然大轉向
        self.declare_parameter(
            'target_slew_rate_mps',
            0.060
        )

        # 根據先前平行擺放時約+0.05 rad的量測值補償
        self.declare_parameter(
            'heading_bias_rad',
            0.050
        )

        # 若實車 angular.z 正負方向符合ROS慣例，保持1.0
        self.declare_parameter(
            'angular_sign',
            1.0
        )

        # 到位條件
        self.declare_parameter(
            'lateral_tolerance_m',
            0.015
        )
        self.declare_parameter(
            'heading_tolerance_rad',
            math.radians(3.0)
        )
        self.declare_parameter(
            'settle_cycles',
            10
        )

        self.lane_geometry_topic = str(
            self.get_parameter(
                'lane_geometry_topic'
            ).value
        )
        self.target_side_topic = str(
            self.get_parameter(
                'target_side_topic'
            ).value
        )
        self.cmd_vel_raw_topic = str(
            self.get_parameter(
                'cmd_vel_raw_topic'
            ).value
        )
        self.status_topic = str(
            self.get_parameter(
                'status_topic'
            ).value
        )

        self.control_rate_hz = float(
            self.get_parameter(
                'control_rate_hz'
            ).value
        )
        self.lane_timeout_s = float(
            self.get_parameter(
                'lane_timeout_s'
            ).value
        )

        self.center_speed_mps = float(
            self.get_parameter(
                'center_speed_mps'
            ).value
        )
        self.side_speed_mps = float(
            self.get_parameter(
                'side_speed_mps'
            ).value
        )
        self.minimum_speed_mps = float(
            self.get_parameter(
                'minimum_speed_mps'
            ).value
        )

        self.side_wall_distance_m = float(
            self.get_parameter(
                'side_wall_distance_m'
            ).value
        )

        self.lateral_kp = float(
            self.get_parameter(
                'lateral_kp'
            ).value
        )
        self.heading_kp = float(
            self.get_parameter(
                'heading_kp'
            ).value
        )
        self.max_angular_speed_rps = float(
            self.get_parameter(
                'max_angular_speed_rps'
            ).value
        )
        self.target_slew_rate_mps = float(
            self.get_parameter(
                'target_slew_rate_mps'
            ).value
        )

        self.heading_bias_rad = float(
            self.get_parameter(
                'heading_bias_rad'
            ).value
        )
        self.angular_sign = float(
            self.get_parameter(
                'angular_sign'
            ).value
        )

        self.lateral_tolerance_m = float(
            self.get_parameter(
                'lateral_tolerance_m'
            ).value
        )
        self.heading_tolerance_rad = float(
            self.get_parameter(
                'heading_tolerance_rad'
            ).value
        )
        self.settle_cycles_required = int(
            self.get_parameter(
                'settle_cycles'
            ).value
        )

        if self.control_rate_hz <= 0.0:
            raise ValueError(
                'control_rate_hz 必須大於0'
            )

        if self.lane_timeout_s <= 0.0:
            raise ValueError(
                'lane_timeout_s 必須大於0'
            )

        if self.side_wall_distance_m <= 0.20:
            raise ValueError(
                '初期測試 side_wall_distance_m 必須大於0.20 m'
            )

        if self.max_angular_speed_rps <= 0.0:
            raise ValueError(
                'max_angular_speed_rps 必須大於0'
            )

        if self.target_slew_rate_mps <= 0.0:
            raise ValueError(
                'target_slew_rate_mps 必須大於0'
            )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_vel_raw_topic,
            10
        )
        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10
        )

        self.lane_sub = self.create_subscription(
            Float32MultiArray,
            self.lane_geometry_topic,
            self.lane_callback,
            10
        )
        self.target_sub = self.create_subscription(
            String,
            self.target_side_topic,
            self.target_callback,
            10
        )

        # 啟動時必須停止，收到明確目標才移動
        self.target_side = 'STOP'

        self.lane_valid = False
        self.left_distance = 0.0
        self.right_distance = 0.0
        self.lane_width = 0.0
        self.heading_raw = 0.0
        self.last_lane_time = None

        self.reference_y = None
        self.settle_count = 0

        self.last_control_time = (
            self.get_clock().now()
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_callback
        )

        self.get_logger().info(
            'lane_change_controller 已啟動；'
            '預設STOP，等待 /level_a/target_side'
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

    def lane_callback(
        self,
        msg: Float32MultiArray
    ) -> None:
        data = list(msg.data)

        if len(data) < 8:
            self.lane_valid = False
            return

        values_to_check = data[:8]

        if not all(
            math.isfinite(float(value))
            for value in values_to_check
        ):
            self.lane_valid = False
            return

        overall_valid = data[0] > 0.5
        left_valid = data[1] > 0.5
        right_valid = data[2] > 0.5
        width_valid = data[7] > 0.5

        self.left_distance = float(data[3])
        self.right_distance = float(data[4])
        self.lane_width = float(data[5])
        self.heading_raw = float(data[6])

        self.lane_valid = (
            overall_valid
            and left_valid
            and right_valid
            and width_valid
            and 0.55 < self.lane_width < 0.95
            and self.left_distance > 0.0
            and self.right_distance > 0.0
        )

        self.last_lane_time = (
            self.get_clock().now()
        )

    def target_callback(
        self,
        msg: String
    ) -> None:
        requested_target = (
            msg.data.strip().upper()
        )

        if requested_target not in self.VALID_TARGETS:
            self.get_logger().warning(
                '忽略未知target_side：'
                f'{requested_target}'
            )
            return

        # Mission會定期重發目前target，確保較晚啟動或重新連線的
        # controller也能取得最新命令。相同target不可重設settle_count，
        # 否則控制器將永遠無法累積到HOLDING。
        if requested_target == self.target_side:
            return

        self.get_logger().info(
            f'target_side：'
            f'{self.target_side} → '
            f'{requested_target}'
        )

        self.target_side = requested_target
        self.settle_count = 0

    def lane_data_is_fresh(self) -> bool:
        if self.last_lane_time is None:
            return False

        age_s = (
            self.get_clock().now()
            - self.last_lane_time
        ).nanoseconds / 1_000_000_000.0

        return (
            self.lane_valid
            and age_s <= self.lane_timeout_s
        )

    def calculate_vehicle_y(self) -> float:
        """
        車體在走道內的橫向位置。

        車體向左移動：
            left_distance 變小
            right_distance 變大
            vehicle_y 變成正值
        """
        return (
            self.right_distance
            - self.left_distance
        ) / 2.0

    def calculate_desired_y(self) -> float:
        available_offset = max(
            0.0,
            self.lane_width / 2.0
            - self.side_wall_distance_m
        )

        if self.target_side == 'LEFT':
            return available_offset

        if self.target_side == 'RIGHT':
            return -available_offset

        return 0.0

    def update_reference_y(
        self,
        desired_y: float,
        dt: float
    ) -> None:
        if self.reference_y is None:
            self.reference_y = (
                self.calculate_vehicle_y()
            )

        difference = (
            desired_y
            - self.reference_y
        )

        maximum_change = (
            self.target_slew_rate_mps
            * dt
        )

        change = self.clamp(
            difference,
            -maximum_change,
            maximum_change
        )

        self.reference_y += change

    def publish_stop(
        self,
        state: str
    ) -> None:
        self.cmd_pub.publish(Twist())

        status = String()
        status.data = (
            f'target={self.target_side} '
            f'state={state} '
            'cmd_v=0.000 cmd_w=0.000'
        )
        self.status_pub.publish(status)

    def control_callback(self) -> None:
        now = self.get_clock().now()

        dt = (
            now
            - self.last_control_time
        ).nanoseconds / 1_000_000_000.0

        self.last_control_time = now

        dt = self.clamp(
            dt,
            0.001,
            0.20
        )

        if not self.lane_data_is_fresh():
            self.reference_y = None
            self.settle_count = 0
            self.publish_stop(
                'NO_LANE_DATA'
            )
            return

        vehicle_y = (
            self.calculate_vehicle_y()
        )

        if self.target_side == 'STOP':
            self.reference_y = vehicle_y
            self.settle_count = 0
            self.publish_stop(
                'STOPPED'
            )
            return

        desired_y = (
            self.calculate_desired_y()
        )

        self.update_reference_y(
            desired_y,
            dt
        )

        lateral_error = (
            self.reference_y
            - vehicle_y
        )

        heading_error = (
            self.heading_raw
            - self.heading_bias_rad
        )

        angular_z = (
            self.lateral_kp
            * lateral_error
            + self.heading_kp
            * heading_error
        )

        angular_z *= self.angular_sign

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed_rps,
            self.max_angular_speed_rps
        )

        if self.target_side == 'CENTER':
            base_speed = self.center_speed_mps
        else:
            base_speed = self.side_speed_mps

        # 轉向越大，前進速度越低
        turn_ratio = (
            abs(angular_z)
            / self.max_angular_speed_rps
        )

        speed_scale = (
            1.0
            - 0.55 * turn_ratio
        )

        linear_x = max(
            self.minimum_speed_mps,
            base_speed * speed_scale
        )

        final_lateral_error = (
            desired_y
            - vehicle_y
        )

        reference_reached = (
            abs(desired_y - self.reference_y)
            <= self.lateral_tolerance_m
        )
        position_reached = (
            abs(final_lateral_error)
            <= self.lateral_tolerance_m
        )
        heading_reached = (
            abs(heading_error)
            <= self.heading_tolerance_rad
        )

        if (
            reference_reached
            and position_reached
            and heading_reached
        ):
            self.settle_count += 1
        else:
            self.settle_count = 0

        if (
            self.settle_count
            >= self.settle_cycles_required
        ):
            state = 'HOLDING'
        else:
            state = 'MOVING'

        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z

        self.cmd_pub.publish(command)

        status = String()
        status.data = (
            f'target={self.target_side} '
            f'state={state} '
            f'y={vehicle_y:.3f} '
            f'target_y={desired_y:.3f} '
            f'ref_y={self.reference_y:.3f} '
            f'heading={heading_error:.3f} '
            f'cmd_v={linear_x:.3f} '
            f'cmd_w={angular_z:.3f}'
        )
        self.status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LaneChangeController()

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
