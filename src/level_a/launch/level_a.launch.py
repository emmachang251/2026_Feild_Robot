#!/usr/bin/env python3

"""Bring up the non-manure Level A stack in a fail-safe state."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import EmitEvent
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    level_a_share = get_package_share_directory('level_a')
    default_config = os.path.join(
        level_a_share,
        'config',
        'level_a.yaml',
    )

    parameter_file = LaunchConfiguration('parameter_file')
    camera_serial = LaunchConfiguration('camera_serial')
    lidar_serial_port = LaunchConfiguration('lidar_serial_port')
    ai_python = LaunchConfiguration('ai_python')

    start_realsense = LaunchConfiguration('start_realsense')
    start_lidar = LaunchConfiguration('start_lidar')
    start_chassis = LaunchConfiguration('start_chassis')
    start_pig_detector = LaunchConfiguration('start_pig_detector')

    launch_arguments = [
        DeclareLaunchArgument(
            'parameter_file',
            default_value=default_config,
            description='Shared Level A ROS parameter YAML.',
        ),
        DeclareLaunchArgument(
            'camera_serial',
            # Keep the quotes in the launch value. The RealSense wrapper
            # otherwise converts an all-digit serial number to an integer,
            # while the camera node requires a string parameter.
            default_value="'944622072730'",
            description='Front RealSense D435 serial number.',
        ),
        DeclareLaunchArgument(
            'lidar_serial_port',
            default_value='/dev/ttyUSB0',
            description='SLLIDAR A1 serial device.',
        ),
        DeclareLaunchArgument(
            'ai_python',
            default_value='/home/bme1234/level_a_ai_venv/bin/python3',
            description='Python interpreter containing ultralytics.',
        ),
        DeclareLaunchArgument(
            'start_realsense',
            default_value='true',
            description='Start the front RealSense camera.',
        ),
        DeclareLaunchArgument(
            'start_lidar',
            default_value='true',
            description='Start the SLLIDAR A1 driver.',
        ),
        DeclareLaunchArgument(
            'start_chassis',
            default_value='false',
            description=(
                'Start UART, odometry, and motor command conversion. '
                'Keep false for no-motion testing.'
            ),
        ),
        DeclareLaunchArgument(
            'start_pig_detector',
            default_value='true',
            description='Start the pig AI detector.',
        ),
    ]

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py',
            ])
        ),
        condition=IfCondition(start_realsense),
        launch_arguments={
            'serial_no': camera_serial,
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.color_profile': '1280x720x15',
            'depth_module.depth_profile': '1280x720x15',
        }.items(),
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('sllidar_ros2'),
                'launch',
                'sllidar_a1_launch.py',
            ])
        ),
        condition=IfCondition(start_lidar),
        launch_arguments={
            'serial_port': lidar_serial_port,
            'serial_baudrate': '115200',
            'frame_id': 'laser',
            'inverted': 'false',
            'angle_compensate': 'true',
        }.items(),
    )

    chassis_nodes = [
        Node(
            package='serial_com',
            executable='uart_node',
            name='uart_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(start_chassis),
        ),
        Node(
            package='serial_com',
            executable='odom_node',
            name='odom_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(start_chassis),
        ),
        Node(
            package='serial_com',
            executable='cmd_vel_to_rover_control',
            name='cmd_vel_to_rover_control',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(start_chassis),
        ),
    ]

    lane_wall_detector = Node(
        package='level_a',
        executable='lane_wall_detector',
        name='lane_wall_detector',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    pig_detector = Node(
        package='level_a',
        executable='pig_detector',
        name='pig_detector',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
        prefix=ai_python,
        condition=IfCondition(start_pig_detector),
    )

    level_a_mission = Node(
        package='level_a',
        executable='level_a_mission',
        name='level_a_mission',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    lane_change_controller = Node(
        package='level_a',
        executable='lane_change_controller',
        name='lane_change_controller',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    u_turn_controller = Node(
        package='level_a',
        executable='u_turn_controller',
        name='u_turn_controller',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    motion_arbiter = Node(
        package='level_a',
        executable='level_a_motion_arbiter',
        name='level_a_motion_arbiter',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    wall_safety = Node(
        package='level_a',
        executable='level_a_wall_safety',
        name='level_a_wall_safety',
        output='screen',
        emulate_tty=True,
        parameters=[parameter_file],
    )

    critical_exit_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=motion_arbiter,
                on_exit=[
                    LogInfo(
                        msg='Motion arbiter exited; shutting down Level A.'
                    ),
                    EmitEvent(
                        event=Shutdown(reason='motion arbiter exited')
                    ),
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=wall_safety,
                on_exit=[
                    LogInfo(
                        msg='Wall safety exited; shutting down Level A.'
                    ),
                    EmitEvent(
                        event=Shutdown(reason='wall safety exited')
                    ),
                ],
            )
        ),
    ]

    startup_notice = LogInfo(
        msg=(
            'Level A non-manure stack started fail-safe. No motion is '
            'allowed until a mission manager publishes fresh '
            '/level_a/motion_mode heartbeats.'
        )
    )

    return LaunchDescription(
        launch_arguments
        + [
            startup_notice,
            realsense_launch,
            lidar_launch,
            *chassis_nodes,
            *critical_exit_handlers,
            lane_wall_detector,
            pig_detector,
            level_a_mission,
            lane_change_controller,
            u_turn_controller,
            motion_arbiter,
            wall_safety,
        ]
    )
