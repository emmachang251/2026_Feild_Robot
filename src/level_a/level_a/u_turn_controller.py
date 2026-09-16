#!/usr/bin/env python3

import math
import time
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String


class UTurnState(Enum):
    IDLE = 'IDLE'
    WAITING_DATA = 'WAITING_DATA'
    APPROACH = 'APPROACH'
    CLEAR_DIVIDER = 'CLEAR_DIVIDER'
    ARC = 'ARC'
    SEARCH_AISLE = 'SEARCH_AISLE'
    ALIGN = 'ALIGN'
    COMPLETE = 'COMPLETE'
    ABORTED = 'ABORTED'


class UTurnController(Node):
    """Drive a guarded U-turn from the first aisle into the second aisle.

    The 2026 Level A field has two 0.75 m aisles separated by a divider.
    Starting in the upper aisle, the default manoeuvre is a left U-turn.

    ``u_turn_enable`` is a heartbeat, not a one-shot command.  The future
    course manager must publish True continuously while the manoeuvre owns
    the vehicle and publish False to reset it.  If that heartbeat, odometry,
    or required lane geometry disappears, this node publishes zero velocity.

    This node only publishes ``/level_a/u_turn_cmd_vel``.  It never publishes
    ``/cmd_vel`` or ``/level_a/cmd_vel_raw`` directly; the motion arbiter and
    wall-safety node remain downstream.
    """

    def __init__(self) -> None:
        super().__init__('u_turn_controller')

        # Topics
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter(
            'lane_geometry_topic',
            '/level_a/lane_geometry',
        )
        self.declare_parameter(
            'enable_topic',
            '/level_a/u_turn_enable',
        )
        self.declare_parameter(
            'command_topic',
            '/level_a/u_turn_cmd_vel',
        )
        self.declare_parameter(
            'status_topic',
            '/level_a/u_turn_status',
        )

        # Field orientation.  For the rulebook start position, the centre
        # divider is on the robot's LEFT and the required turn is LEFT.
        self.declare_parameter('turn_direction', 'LEFT')
        self.declare_parameter('inner_wall_side', 'LEFT')

        # Rates and watchdogs
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('enable_timeout_sec', 0.75)
        self.declare_parameter('odom_timeout_sec', 0.50)
        self.declare_parameter('lane_timeout_sec', 0.50)
        self.declare_parameter('input_wait_timeout_sec', 3.0)
        self.declare_parameter('overall_timeout_sec', 35.0)
        self.declare_parameter('approach_timeout_sec', 12.0)
        self.declare_parameter('align_timeout_sec', 8.0)

        # Opening detection and path geometry
        self.declare_parameter('opening_confirm_frames', 5)
        self.declare_parameter('minimum_approach_distance_m', 0.10)
        self.declare_parameter('maximum_approach_distance_m', 1.50)
        # The wall detector uses points up to the divider tip and its LiDAR is
        # forward of base_link.  This distance is intentionally configurable
        # and must be calibrated on the real field before a powered run.
        self.declare_parameter('divider_clear_distance_m', 0.38)
        self.declare_parameter('arc_radius_m', 0.375)
        self.declare_parameter('arc_search_start_angle_rad', 2.70)
        self.declare_parameter('maximum_turn_angle_rad', 3.60)
        self.declare_parameter('maximum_odom_yaw_step_rad', 0.50)

        # Speeds
        self.declare_parameter('approach_speed_mps', 0.045)
        self.declare_parameter('clear_speed_mps', 0.040)
        self.declare_parameter('arc_speed_mps', 0.055)
        self.declare_parameter('search_speed_mps', 0.035)
        self.declare_parameter('align_speed_mps', 0.035)
        self.declare_parameter('maximum_angular_speed_radps', 0.35)

        # Final centre-line alignment
        self.declare_parameter('heading_bias_rad', 0.050)
        self.declare_parameter('align_lateral_kp', 2.8)
        self.declare_parameter('align_heading_kp', 1.4)
        self.declare_parameter('align_lateral_tolerance_m', 0.025)
        self.declare_parameter(
            'align_heading_tolerance_rad',
            math.radians(5.0),
        )
        self.declare_parameter('align_settle_cycles', 10)
        self.declare_parameter('expected_lane_width_m', 0.75)
        self.declare_parameter('lane_width_tolerance_m', 0.15)

        self.odom_topic = self.parameter_string('odom_topic')
        self.lane_geometry_topic = self.parameter_string(
            'lane_geometry_topic'
        )
        self.enable_topic = self.parameter_string('enable_topic')
        self.command_topic = self.parameter_string('command_topic')
        self.status_topic = self.parameter_string('status_topic')

        self.turn_direction = self.parameter_string(
            'turn_direction'
        ).upper()
        self.inner_wall_side = self.parameter_string(
            'inner_wall_side'
        ).upper()
        self.turn_sign = 1.0 if self.turn_direction == 'LEFT' else -1.0

        self.control_rate_hz = self.parameter_float('control_rate_hz')
        self.enable_timeout_sec = self.parameter_float('enable_timeout_sec')
        self.odom_timeout_sec = self.parameter_float('odom_timeout_sec')
        self.lane_timeout_sec = self.parameter_float('lane_timeout_sec')
        self.input_wait_timeout_sec = self.parameter_float(
            'input_wait_timeout_sec'
        )
        self.overall_timeout_sec = self.parameter_float('overall_timeout_sec')
        self.approach_timeout_sec = self.parameter_float(
            'approach_timeout_sec'
        )
        self.align_timeout_sec = self.parameter_float('align_timeout_sec')

        self.opening_confirm_frames = self.parameter_int(
            'opening_confirm_frames'
        )
        self.minimum_approach_distance = self.parameter_float(
            'minimum_approach_distance_m'
        )
        self.maximum_approach_distance = self.parameter_float(
            'maximum_approach_distance_m'
        )
        self.divider_clear_distance = self.parameter_float(
            'divider_clear_distance_m'
        )
        self.arc_radius = self.parameter_float('arc_radius_m')
        self.arc_search_start_angle = self.parameter_float(
            'arc_search_start_angle_rad'
        )
        self.maximum_turn_angle = self.parameter_float(
            'maximum_turn_angle_rad'
        )
        self.maximum_odom_yaw_step = self.parameter_float(
            'maximum_odom_yaw_step_rad'
        )

        self.approach_speed = self.parameter_float('approach_speed_mps')
        self.clear_speed = self.parameter_float('clear_speed_mps')
        self.arc_speed = self.parameter_float('arc_speed_mps')
        self.search_speed = self.parameter_float('search_speed_mps')
        self.align_speed = self.parameter_float('align_speed_mps')
        self.maximum_angular_speed = self.parameter_float(
            'maximum_angular_speed_radps'
        )

        self.heading_bias = self.parameter_float('heading_bias_rad')
        self.align_lateral_kp = self.parameter_float('align_lateral_kp')
        self.align_heading_kp = self.parameter_float('align_heading_kp')
        self.align_lateral_tolerance = self.parameter_float(
            'align_lateral_tolerance_m'
        )
        self.align_heading_tolerance = self.parameter_float(
            'align_heading_tolerance_rad'
        )
        self.align_settle_cycles = self.parameter_int(
            'align_settle_cycles'
        )
        self.expected_lane_width = self.parameter_float(
            'expected_lane_width_m'
        )
        self.lane_width_tolerance = self.parameter_float(
            'lane_width_tolerance_m'
        )

        self.validate_parameters()

        self.state = UTurnState.IDLE
        self.state_enter_time = time.monotonic()
        self.manoeuvre_start_time: Optional[float] = None
        self.state_start_x: Optional[float] = None
        self.state_start_y: Optional[float] = None

        self.enable = False
        self.last_enable_time: Optional[float] = None

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.turn_progress = 0.0
        self.odom_fault: Optional[str] = None

        self.geometry_valid = False
        self.left_wall_valid = False
        self.right_wall_valid = False
        self.lane_width_valid = False
        self.left_distance = math.nan
        self.right_distance = math.nan
        self.lane_width = math.nan
        self.heading = math.nan
        self.last_lane_time: Optional[float] = None
        self.opening_frames = 0
        self.align_settle_count = 0

        self.last_status_key: Optional[str] = None

        self.command_pub = self.create_publisher(
            Twist,
            self.command_topic,
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )
        self.lane_sub = self.create_subscription(
            Float32MultiArray,
            self.lane_geometry_topic,
            self.lane_callback,
            10,
        )
        self.enable_sub = self.create_subscription(
            Bool,
            self.enable_topic,
            self.enable_callback,
            10,
        )
        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_callback,
        )

        self.get_logger().info(
            'U-turn controller started in IDLE. '
            f'direction={self.turn_direction}, '
            f'inner_wall={self.inner_wall_side}, '
            f'command={self.command_topic}'
        )

    def parameter_string(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def parameter_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def parameter_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def validate_parameters(self) -> None:
        if self.turn_direction not in {'LEFT', 'RIGHT'}:
            raise ValueError('turn_direction must be LEFT or RIGHT')
        if self.inner_wall_side not in {'LEFT', 'RIGHT'}:
            raise ValueError('inner_wall_side must be LEFT or RIGHT')

        positive_values = {
            'control_rate_hz': self.control_rate_hz,
            'enable_timeout_sec': self.enable_timeout_sec,
            'odom_timeout_sec': self.odom_timeout_sec,
            'lane_timeout_sec': self.lane_timeout_sec,
            'input_wait_timeout_sec': self.input_wait_timeout_sec,
            'overall_timeout_sec': self.overall_timeout_sec,
            'approach_timeout_sec': self.approach_timeout_sec,
            'align_timeout_sec': self.align_timeout_sec,
            'maximum_approach_distance_m': self.maximum_approach_distance,
            'divider_clear_distance_m': self.divider_clear_distance,
            'arc_radius_m': self.arc_radius,
            'arc_search_start_angle_rad': self.arc_search_start_angle,
            'maximum_turn_angle_rad': self.maximum_turn_angle,
            'maximum_odom_yaw_step_rad': self.maximum_odom_yaw_step,
            'approach_speed_mps': self.approach_speed,
            'clear_speed_mps': self.clear_speed,
            'arc_speed_mps': self.arc_speed,
            'search_speed_mps': self.search_speed,
            'align_speed_mps': self.align_speed,
            'maximum_angular_speed_radps': self.maximum_angular_speed,
            'align_lateral_kp': self.align_lateral_kp,
            'align_heading_kp': self.align_heading_kp,
            'align_lateral_tolerance_m': self.align_lateral_tolerance,
            'align_heading_tolerance_rad': self.align_heading_tolerance,
            'expected_lane_width_m': self.expected_lane_width,
            'lane_width_tolerance_m': self.lane_width_tolerance,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and greater than zero')

        if (
            not math.isfinite(self.minimum_approach_distance)
            or self.minimum_approach_distance < 0.0
        ):
            raise ValueError(
                'minimum_approach_distance_m must be finite and non-negative'
            )
        if self.maximum_approach_distance <= self.minimum_approach_distance:
            raise ValueError(
                'maximum_approach_distance_m must exceed '
                'minimum_approach_distance_m'
            )
        if self.maximum_turn_angle <= self.arc_search_start_angle:
            raise ValueError(
                'maximum_turn_angle_rad must exceed '
                'arc_search_start_angle_rad'
            )
        if self.opening_confirm_frames < 1:
            raise ValueError('opening_confirm_frames must be at least one')
        if self.align_settle_cycles < 1:
            raise ValueError('align_settle_cycles must be at least one')

        arc_angular_speed = self.arc_speed / self.arc_radius
        if arc_angular_speed > self.maximum_angular_speed:
            raise ValueError(
                'arc_speed_mps / arc_radius_m exceeds '
                'maximum_angular_speed_radps'
            )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def quaternion_to_yaw(orientation) -> float:
        sin_yaw = 2.0 * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        )
        cos_yaw = 1.0 - 2.0 * (
            float(orientation.y) ** 2
            + float(orientation.z) ** 2
        )
        return math.atan2(sin_yaw, cos_yaw)

    @staticmethod
    def age_seconds(now: float, timestamp: Optional[float]) -> float:
        if timestamp is None:
            return math.inf
        return max(0.0, now - timestamp)

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def odom_callback(self, message: Odometry) -> None:
        now = time.monotonic()
        yaw = self.quaternion_to_yaw(message.pose.pose.orientation)
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)

        if not all(math.isfinite(value) for value in (x, y, yaw)):
            self.odom_fault = 'non_finite_odom'
            self.last_odom_time = None
            return

        if self.last_odom_yaw is not None:
            yaw_step = self.normalize_angle(yaw - self.last_odom_yaw)
            if (
                self.state in (UTurnState.ARC, UTurnState.SEARCH_AISLE)
                and abs(yaw_step) > self.maximum_odom_yaw_step
            ):
                self.odom_fault = 'odom_yaw_jump'
            elif self.state in (
                UTurnState.ARC,
                UTurnState.SEARCH_AISLE,
            ):
                self.turn_progress += self.turn_sign * yaw_step

        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = yaw
        self.last_odom_yaw = yaw
        self.last_odom_time = now

    def lane_callback(self, message: Float32MultiArray) -> None:
        now = time.monotonic()
        data = list(message.data)
        self.last_lane_time = now

        if len(data) < 8:
            self.clear_lane_geometry()
            return

        self.geometry_valid = data[0] > 0.5
        self.left_wall_valid = data[1] > 0.5
        self.right_wall_valid = data[2] > 0.5
        self.lane_width_valid = data[7] > 0.5
        self.left_distance = float(data[3])
        self.right_distance = float(data[4])
        self.lane_width = float(data[5])
        self.heading = float(data[6])

        if self.state == UTurnState.APPROACH:
            if self.inner_opening_is_visible():
                self.opening_frames += 1
            else:
                self.opening_frames = 0

    def clear_lane_geometry(self) -> None:
        self.geometry_valid = False
        self.left_wall_valid = False
        self.right_wall_valid = False
        self.lane_width_valid = False
        self.left_distance = math.nan
        self.right_distance = math.nan
        self.lane_width = math.nan
        self.heading = math.nan
        self.opening_frames = 0

    def inner_opening_is_visible(self) -> bool:
        if not self.geometry_valid:
            return False
        if self.inner_wall_side == 'LEFT':
            return not self.left_wall_valid and self.right_wall_valid
        return not self.right_wall_valid and self.left_wall_valid

    def full_lane_is_valid(self) -> bool:
        values_valid = all(
            math.isfinite(value)
            for value in (
                self.left_distance,
                self.right_distance,
                self.lane_width,
                self.heading,
            )
        )
        return (
            self.geometry_valid
            and self.left_wall_valid
            and self.right_wall_valid
            and self.lane_width_valid
            and values_valid
            and self.left_distance > 0.0
            and self.right_distance > 0.0
            and abs(self.lane_width - self.expected_lane_width)
            <= self.lane_width_tolerance
        )

    def enable_callback(self, message: Bool) -> None:
        now = time.monotonic()
        requested_enable = bool(message.data)
        self.last_enable_time = now

        if not requested_enable:
            self.enable = False
            if self.state != UTurnState.IDLE:
                self.transition(UTurnState.IDLE, now, 'enable_released')
            self.publish_stop('enable_false')
            return

        self.enable = True
        if self.state == UTurnState.IDLE:
            self.reset_manoeuvre(now)
            self.transition(UTurnState.WAITING_DATA, now, 'enable_asserted')

    def reset_manoeuvre(self, now: float) -> None:
        self.manoeuvre_start_time = now
        self.state_start_x = None
        self.state_start_y = None
        self.turn_progress = 0.0
        self.odom_fault = None
        self.opening_frames = 0
        self.align_settle_count = 0

    def transition(
        self,
        new_state: UTurnState,
        now: float,
        reason: str,
    ) -> None:
        previous_state = self.state
        self.state = new_state
        self.state_enter_time = now
        self.state_start_x = self.odom_x if self.last_odom_time is not None else None
        self.state_start_y = self.odom_y if self.last_odom_time is not None else None

        if new_state == UTurnState.ARC:
            self.turn_progress = 0.0
            self.last_odom_yaw = self.odom_yaw
        if new_state == UTurnState.ALIGN:
            self.align_settle_count = 0
        if new_state in (UTurnState.IDLE, UTurnState.ABORTED):
            self.opening_frames = 0

        self.get_logger().info(
            f'U-turn state: {previous_state.value} -> {new_state.value} '
            f'reason={reason}'
        )

    def distance_from_state_start(self) -> float:
        if self.state_start_x is None or self.state_start_y is None:
            return 0.0
        return math.hypot(
            self.odom_x - self.state_start_x,
            self.odom_y - self.state_start_y,
        )

    def abort(self, now: float, reason: str) -> None:
        if self.state != UTurnState.ABORTED:
            self.transition(UTurnState.ABORTED, now, reason)
        self.publish_stop(reason)

    def control_callback(self) -> None:
        now = time.monotonic()
        try:
            self.control_once(now)
        except Exception as error:
            self.get_logger().error(
                f'U-turn controller exception; forcing stop: {error}'
            )
            self.abort(now, 'internal_exception')

    def control_once(self, now: float) -> None:
        if self.state == UTurnState.IDLE:
            self.publish_stop('idle')
            return

        if not self.enable:
            self.transition(UTurnState.IDLE, now, 'enable_not_active')
            self.publish_stop('enable_not_active')
            return

        # Terminal states are latched until an explicit False resets the
        # controller.  They always continue publishing zero velocity.
        if self.state == UTurnState.COMPLETE:
            self.publish_stop('complete')
            return
        if self.state == UTurnState.ABORTED:
            self.publish_stop('aborted')
            return

        if self.age_seconds(now, self.last_enable_time) > self.enable_timeout_sec:
            self.abort(now, 'enable_timeout')
            return

        if self.odom_fault is not None:
            self.abort(now, self.odom_fault)
            return

        if (
            self.manoeuvre_start_time is not None
            and now - self.manoeuvre_start_time > self.overall_timeout_sec
        ):
            self.abort(now, 'overall_timeout')
            return

        if self.state == UTurnState.WAITING_DATA:
            if now - self.state_enter_time > self.input_wait_timeout_sec:
                self.abort(now, 'input_wait_timeout')
                return
            if self.age_seconds(now, self.last_odom_time) > self.odom_timeout_sec:
                self.publish_stop('waiting_for_odometry')
                return
            if self.age_seconds(now, self.last_lane_time) > self.lane_timeout_sec:
                self.publish_stop('waiting_for_lane_geometry')
                return
            if not self.full_lane_is_valid():
                self.publish_stop('waiting_for_valid_first_aisle')
                return
            self.transition(UTurnState.APPROACH, now, 'inputs_ready')
            self.publish_stop('approach_guard')
            return

        if self.age_seconds(now, self.last_odom_time) > self.odom_timeout_sec:
            self.abort(now, 'odom_timeout')
            return

        if self.state == UTurnState.APPROACH:
            self.control_approach(now)
            return

        if self.state == UTurnState.CLEAR_DIVIDER:
            self.control_clear_divider(now)
            return

        if self.state == UTurnState.ARC:
            self.control_arc(now)
            return

        if self.state == UTurnState.SEARCH_AISLE:
            self.control_search_aisle(now)
            return

        if self.state == UTurnState.ALIGN:
            self.control_align(now)
            return

        if self.state == UTurnState.COMPLETE:
            self.publish_stop('complete')
            return

        self.publish_stop('aborted')

    def lane_is_fresh(self, now: float) -> bool:
        return self.age_seconds(now, self.last_lane_time) <= self.lane_timeout_sec

    def control_approach(self, now: float) -> None:
        if not self.lane_is_fresh(now):
            self.abort(now, 'lane_timeout_during_approach')
            return
        if now - self.state_enter_time > self.approach_timeout_sec:
            self.abort(now, 'opening_not_found_timeout')
            return

        distance = self.distance_from_state_start()
        if distance > self.maximum_approach_distance:
            self.abort(now, 'opening_not_found_distance')
            return

        if (
            distance >= self.minimum_approach_distance
            and self.opening_frames >= self.opening_confirm_frames
        ):
            self.transition(UTurnState.CLEAR_DIVIDER, now, 'opening_confirmed')
            self.publish_stop('opening_transition_guard')
            return

        angular_z = 0.0
        if self.full_lane_is_valid():
            angular_z = self.center_alignment_angular_z()
        self.publish_command(
            self.approach_speed,
            angular_z,
            'approaching_divider_end',
        )

    def control_clear_divider(self, now: float) -> None:
        if self.distance_from_state_start() >= self.divider_clear_distance:
            self.transition(UTurnState.ARC, now, 'divider_clear_distance_reached')
            self.publish_stop('arc_transition_guard')
            return
        self.publish_command(
            self.clear_speed,
            0.0,
            'clearing_divider_tip',
        )

    def arc_angular_speed(self, linear_speed: float) -> float:
        magnitude = min(
            linear_speed / self.arc_radius,
            self.maximum_angular_speed,
        )
        return self.turn_sign * magnitude

    def control_arc(self, now: float) -> None:
        if self.turn_progress >= self.arc_search_start_angle:
            self.transition(
                UTurnState.SEARCH_AISLE,
                now,
                'search_angle_reached',
            )
            self.publish_stop('search_transition_guard')
            return
        self.publish_command(
            self.arc_speed,
            self.arc_angular_speed(self.arc_speed),
            'turning_arc',
        )

    def control_search_aisle(self, now: float) -> None:
        if self.turn_progress > self.maximum_turn_angle:
            self.abort(now, 'maximum_turn_angle_exceeded')
            return
        if self.lane_is_fresh(now) and self.full_lane_is_valid():
            self.transition(UTurnState.ALIGN, now, 'second_aisle_acquired')
            self.publish_stop('align_transition_guard')
            return
        self.publish_command(
            self.search_speed,
            self.arc_angular_speed(self.search_speed),
            'searching_second_aisle',
        )

    def control_align(self, now: float) -> None:
        if now - self.state_enter_time > self.align_timeout_sec:
            self.abort(now, 'align_timeout')
            return
        if not self.lane_is_fresh(now) or not self.full_lane_is_valid():
            self.align_settle_count = 0
            self.publish_stop('waiting_for_stable_second_aisle')
            return

        vehicle_y = (self.right_distance - self.left_distance) / 2.0
        lateral_error = -vehicle_y
        heading_error = self.heading - self.heading_bias
        angular_z = self.center_alignment_angular_z()

        if (
            abs(lateral_error) <= self.align_lateral_tolerance
            and abs(heading_error) <= self.align_heading_tolerance
        ):
            self.align_settle_count += 1
        else:
            self.align_settle_count = 0

        if self.align_settle_count >= self.align_settle_cycles:
            self.transition(UTurnState.COMPLETE, now, 'alignment_confirmed')
            self.publish_stop('complete')
            return

        self.publish_command(
            self.align_speed,
            angular_z,
            'aligning_second_aisle',
        )

    def center_alignment_angular_z(self) -> float:
        vehicle_y = (self.right_distance - self.left_distance) / 2.0
        lateral_error = -vehicle_y
        heading_error = self.heading - self.heading_bias
        angular_z = (
            self.align_lateral_kp * lateral_error
            + self.align_heading_kp * heading_error
        )
        return self.clamp(
            angular_z,
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )

    def publish_command(
        self,
        linear_x: float,
        angular_z: float,
        reason: str,
    ) -> None:
        if not all(math.isfinite(value) for value in (linear_x, angular_z)):
            self.publish_stop('non_finite_generated_command')
            return

        command = Twist()
        command.linear.x = max(0.0, float(linear_x))
        command.angular.z = self.clamp(
            float(angular_z),
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )
        self.command_pub.publish(command)
        self.publish_status(reason, command)

    def publish_stop(self, reason: str) -> None:
        command = Twist()
        self.command_pub.publish(command)
        self.publish_status(reason, command)

    def publish_status(self, reason: str, command: Twist) -> None:
        status = String()
        status.data = (
            f'state={self.state.value} '
            f'reason={reason} '
            f'enable={self.enable} '
            f'turn_progress={self.turn_progress:.3f} '
            f'opening_frames={self.opening_frames}/'
            f'{self.opening_confirm_frames} '
            f'cmd_v={command.linear.x:.3f} '
            f'cmd_w={command.angular.z:.3f}'
        )
        self.status_pub.publish(status)

        status_key = f'{self.state.value}|{reason}'
        if status_key != self.last_status_key:
            self.last_status_key = status_key
            self.get_logger().info(status.data)

    def publish_shutdown_stop(self) -> None:
        self.enable = False
        self.state = UTurnState.IDLE
        self.publish_stop('controller_shutting_down')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UTurnController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_shutdown_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
