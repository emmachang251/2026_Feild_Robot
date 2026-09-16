#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    level_a_share = get_package_share_directory('level_a')
    realsense_share = get_package_share_directory('realsense2_camera')

    level_a_config = os.path.join(
        level_a_share,
        'config',
        'level_a.yaml',
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                realsense_share,
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '1280x720x15',
            'depth_module.depth_profile': '1280x720x15',
        }.items(),
    )

    manure_detector = Node(
        package='level_a',
        executable='manure_groove_detector',
        name='manure_groove_detector',
        output='screen',
        emulate_tty=True,
        parameters=[level_a_config],
    )

    return LaunchDescription([
        realsense_launch,
        manure_detector,
    ])
