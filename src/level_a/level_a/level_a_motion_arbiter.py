#!/usr/bin/env python3

import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String


class LevelAMotionArbiter(Node):
    """Select exactly one Level A velocity source and fail safely to zero.

    Control chain:

        lane_change_controller -> /level_a/lane_cmd_vel --+
                                                          |
        u_turn_controller     -> /level_a/u_turn_cmd_vel --+-> arbiter
                                                               |
                                                    /level_a/cmd_vel_raw

    The downstream wall-safety node remains the only publisher of /cmd_vel.
    A mode heartbeat is required so that a crashed mission supervisor cannot
    leave the last controller active indefinitely.
    """

    VALID_MODES = {'STOP', 'LANE', 'U_TURN'}

    def __init__(self) -> None:
        super().__init__('level_a_motion_arbiter')

        # Topics
        self.declare_parameter('lane_cmd_topic', '/level_a/lane_cmd_vel')
        self.declare_parameter('u_turn_cmd_topic', '/level_a/u_turn_cmd_vel')
        self.declare_parameter('motion_mode_topic', '/level_a/motion_mode')
        self.declare_parameter(
            'emergency_stop_topic',
            '/level_a/emergency_stop',
        )
        self.declare_parameter('output_cmd_topic', '/level_a/cmd_vel_raw')
        self.declare_parameter(
            'status_topic',
            '/level_a/arbiter_status',
        )

        # Watchdogs and output rate
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('command_timeout_sec', 0.30)
        self.declare_parameter('mode_timeout_sec', 0.75)
        self.declare_parameter('emergency_timeout_sec', 0.75)
        self.declare_parameter('switch_stop_duration_sec', 0.20)

        # Conservative limits. The downstream wall-safety node applies its
        # own limits again before publishing /cmd_vel.
        self.declare_parameter('maximum_forward_speed_mps', 0.20)
        self.declare_parameter('maximum_reverse_speed_mps', 0.0)
        self.declare_parameter('maximum_angular_speed_radps', 0.80)

        self.lane_cmd_topic = str(
            self.get_parameter('lane_cmd_topic').value
        )
        self.u_turn_cmd_topic = str(
            self.get_parameter('u_turn_cmd_topic').value
        )
        self.motion_mode_topic = str(
            self.get_parameter('motion_mode_topic').value
        )
        self.emergency_stop_topic = str(
            self.get_parameter('emergency_stop_topic').value
        )
        self.output_cmd_topic = str(
            self.get_parameter('output_cmd_topic').value
        )
        self.status_topic = str(
            self.get_parameter('status_topic').value
        )

        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').value
        )
        self.command_timeout_sec = float(
            self.get_parameter('command_timeout_sec').value
        )
        self.mode_timeout_sec = float(
            self.get_parameter('mode_timeout_sec').value
        )
        self.emergency_timeout_sec = float(
            self.get_parameter('emergency_timeout_sec').value
        )
        self.switch_stop_duration_sec = float(
            self.get_parameter('switch_stop_duration_sec').value
        )
        self.maximum_forward_speed = float(
            self.get_parameter('maximum_forward_speed_mps').value
        )
        self.maximum_reverse_speed = float(
            self.get_parameter('maximum_reverse_speed_mps').value
        )
        self.maximum_angular_speed = float(
            self.get_parameter('maximum_angular_speed_radps').value
        )

        self.validate_parameters()
        self.validate_topic_configuration()

        self.mode = 'STOP'
        self.last_mode_time: Optional[float] = None
        self.mode_changed_time: Optional[float] = None
        self.switch_stop_until = math.inf

        self.lane_command = Twist()
        self.u_turn_command = Twist()
        self.last_lane_command_time: Optional[float] = None
        self.last_u_turn_command_time: Optional[float] = None

        self.emergency_stop = True
        self.last_emergency_time: Optional[float] = None

        self.last_status: Optional[str] = None

        self.output_pub = self.create_publisher(
            Twist,
            self.output_cmd_topic,
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )

        self.lane_cmd_sub = self.create_subscription(
            Twist,
            self.lane_cmd_topic,
            self.lane_command_callback,
            10,
        )
        self.u_turn_cmd_sub = self.create_subscription(
            Twist,
            self.u_turn_cmd_topic,
            self.u_turn_command_callback,
            10,
        )
        self.mode_sub = self.create_subscription(
            String,
            self.motion_mode_topic,
            self.mode_callback,
            10,
        )
        self.emergency_sub = self.create_subscription(
            Bool,
            self.emergency_stop_topic,
            self.emergency_callback,
            10,
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_callback,
        )

        self.get_logger().info(
            'Level A motion arbiter started in fail-safe STOP mode. '
            f'lane={self.lane_cmd_topic}, '
            f'u_turn={self.u_turn_cmd_topic}, '
            f'output={self.output_cmd_topic}'
        )

    def validate_parameters(self) -> None:
        if not math.isfinite(self.control_rate_hz) or self.control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        if (
            not math.isfinite(self.command_timeout_sec)
            or self.command_timeout_sec <= 0.0
        ):
            raise ValueError('command_timeout_sec must be greater than zero')
        if (
            not math.isfinite(self.mode_timeout_sec)
            or self.mode_timeout_sec <= 0.0
        ):
            raise ValueError('mode_timeout_sec must be greater than zero')
        if (
            not math.isfinite(self.emergency_timeout_sec)
            or self.emergency_timeout_sec <= 0.0
        ):
            raise ValueError('emergency_timeout_sec must be greater than zero')
        if (
            not math.isfinite(self.switch_stop_duration_sec)
            or self.switch_stop_duration_sec < 0.0
        ):
            raise ValueError(
                'switch_stop_duration_sec cannot be negative'
            )
        if (
            not math.isfinite(self.maximum_forward_speed)
            or self.maximum_forward_speed < 0.0
        ):
            raise ValueError(
                'maximum_forward_speed_mps cannot be negative'
            )
        if (
            not math.isfinite(self.maximum_reverse_speed)
            or self.maximum_reverse_speed < 0.0
        ):
            raise ValueError(
                'maximum_reverse_speed_mps cannot be negative'
            )
        if (
            not math.isfinite(self.maximum_angular_speed)
            or self.maximum_angular_speed <= 0.0
        ):
            raise ValueError(
                'maximum_angular_speed_radps must be greater than zero'
            )

    def validate_topic_configuration(self) -> None:
        command_inputs = {
            self.lane_cmd_topic,
            self.u_turn_cmd_topic,
        }
        if len(command_inputs) != 2:
            raise ValueError(
                'lane_cmd_topic and u_turn_cmd_topic must be different'
            )
        if self.output_cmd_topic in command_inputs:
            raise ValueError(
                'output_cmd_topic must differ from controller input topics'
            )

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def command_is_finite(message: Twist) -> bool:
        values = (
            message.linear.x,
            message.linear.y,
            message.linear.z,
            message.angular.x,
            message.angular.y,
            message.angular.z,
        )
        return all(math.isfinite(float(value)) for value in values)

    @staticmethod
    def copy_planar_command(message: Twist) -> Twist:
        copied = Twist()
        copied.linear.x = float(message.linear.x)
        copied.angular.z = float(message.angular.z)
        return copied

    def lane_command_callback(self, message: Twist) -> None:
        if not self.command_is_finite(message):
            self.last_lane_command_time = None
            self.get_logger().error(
                'Rejected non-finite command from lane controller'
            )
            if self.mode == 'LANE':
                self.publish_output(
                    Twist(),
                    'STOPPED',
                    'lane',
                    'invalid_lane_command',
                )
            return
        self.lane_command = self.copy_planar_command(message)
        self.last_lane_command_time = time.monotonic()

    def u_turn_command_callback(self, message: Twist) -> None:
        if not self.command_is_finite(message):
            self.last_u_turn_command_time = None
            self.get_logger().error(
                'Rejected non-finite command from U-turn controller'
            )
            if self.mode == 'U_TURN':
                self.publish_output(
                    Twist(),
                    'STOPPED',
                    'u_turn',
                    'invalid_u_turn_command',
                )
            return
        self.u_turn_command = self.copy_planar_command(message)
        self.last_u_turn_command_time = time.monotonic()

    def mode_callback(self, message: String) -> None:
        requested_mode = message.data.strip().upper()
        if requested_mode == 'UTURN':
            requested_mode = 'U_TURN'

        if requested_mode not in self.VALID_MODES:
            self.get_logger().warning(
                f'Rejected unknown motion mode: {requested_mode}'
            )
            return

        now = time.monotonic()
        previous_mode = self.mode
        self.last_mode_time = now

        if requested_mode != previous_mode:
            self.mode = requested_mode
            self.mode_changed_time = now
            if requested_mode == 'STOP':
                self.switch_stop_until = now
            else:
                self.switch_stop_until = (
                    now + self.switch_stop_duration_sec
                )
            self.get_logger().info(
                f'Motion mode: {previous_mode} -> {requested_mode}'
            )

            # Do not wait for the next timer tick when entering STOP.  Also
            # publish zero during a controller handover so the downstream
            # safety layer immediately sees the switching guard.
            if requested_mode == 'STOP':
                self.publish_output(
                    Twist(),
                    'STOPPED',
                    'none',
                    'mode_stop',
                )
            else:
                self.publish_output(
                    Twist(),
                    'STOPPED',
                    'none',
                    'controller_switch_guard',
                )

    def emergency_callback(self, message: Bool) -> None:
        was_stopped = self.emergency_stop
        self.emergency_stop = bool(message.data)
        self.last_emergency_time = time.monotonic()

        if self.emergency_stop and not was_stopped:
            self.get_logger().warning('Emergency stop asserted')
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'emergency_stop_active',
            )
        elif was_stopped and not self.emergency_stop:
            self.get_logger().info('Emergency stop released')

    @staticmethod
    def age_seconds(now: float, timestamp: Optional[float]) -> float:
        if timestamp is None:
            return math.inf
        return max(0.0, now - timestamp)

    def selected_command(self):
        if self.mode == 'LANE':
            return (
                'lane',
                self.lane_command,
                self.last_lane_command_time,
            )
        if self.mode == 'U_TURN':
            return (
                'u_turn',
                self.u_turn_command,
                self.last_u_turn_command_time,
            )
        return ('none', Twist(), None)

    def bounded_command(self, command: Twist) -> Twist:
        bounded = Twist()
        bounded.linear.x = self.clamp(
            float(command.linear.x),
            -self.maximum_reverse_speed,
            self.maximum_forward_speed,
        )
        bounded.angular.z = self.clamp(
            float(command.angular.z),
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )
        return bounded

    def control_callback(self) -> None:
        try:
            self.control_once(time.monotonic())
        except Exception as error:  # Final fail-safe boundary.
            self.get_logger().error(
                f'Arbiter exception; forcing STOP: {error}'
            )
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'internal_exception',
            )

    def control_once(self, now: float) -> None:
        emergency_age = self.age_seconds(now, self.last_emergency_time)
        mode_age = self.age_seconds(now, self.last_mode_time)

        if emergency_age > self.emergency_timeout_sec:
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'emergency_status_timeout',
            )
            return

        if self.emergency_stop:
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'emergency_stop_active',
            )
            return

        if mode_age > self.mode_timeout_sec:
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'motion_mode_timeout',
            )
            return

        if self.mode == 'STOP':
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'mode_stop',
            )
            return

        if now < self.switch_stop_until:
            self.publish_output(
                Twist(),
                'STOPPED',
                'none',
                'controller_switch_guard',
            )
            return

        source, command, command_time = self.selected_command()
        command_age = self.age_seconds(now, command_time)
        if (
            command_time is None
            or self.mode_changed_time is None
            or command_time < self.mode_changed_time
        ):
            self.publish_output(
                Twist(),
                'STOPPED',
                source,
                f'waiting_for_new_{source}_command',
            )
            return

        if command_age > self.command_timeout_sec:
            self.publish_output(
                Twist(),
                'STOPPED',
                source,
                f'{source}_command_timeout',
            )
            return

        if not self.command_is_finite(command):
            self.publish_output(
                Twist(),
                'STOPPED',
                source,
                f'invalid_{source}_command',
            )
            return

        bounded = self.bounded_command(command)
        self.publish_output(bounded, 'ACTIVE', source, 'ok')

    def publish_output(
        self,
        command: Twist,
        state: str,
        source: str,
        reason: str,
    ) -> None:
        self.output_pub.publish(command)

        status = String()
        status.data = (
            f'mode={self.mode} '
            f'source={source} '
            f'state={state} '
            f'reason={reason} '
            f'cmd_v={command.linear.x:.3f} '
            f'cmd_w={command.angular.z:.3f}'
        )
        self.status_pub.publish(status)

        log_key = f'{self.mode}|{source}|{state}|{reason}'
        if log_key != self.last_status:
            self.last_status = log_key
            if state == 'ACTIVE':
                self.get_logger().info(status.data)
            else:
                self.get_logger().warning(status.data)

    def publish_shutdown_stop(self) -> None:
        self.mode = 'STOP'
        self.publish_output(
            Twist(),
            'STOPPED',
            'none',
            'arbiter_shutting_down',
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LevelAMotionArbiter()

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
