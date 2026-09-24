import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # =========================================================
    # 1. UART
    #    ROS2 <-> STM32
    # =========================================================

    uart_node = Node(
        package='serial_com',
        executable='uart_node',
        name='uart_node',
        output='screen'
    )

    # =========================================================
    # 2. Odometry
    # =========================================================

    odom_node = Node(
        package='serial_com',
        executable='odom_node',
        name='odom_node',
        output='screen'
    )

    # =========================================================
    # 3. /cmd_vel -> /chassis_control
    #
    #    Level C 會發布：
    #
    #        /cmd_vel
    #
    #    這個 node 會轉成：
    #
    #        /chassis_control
    # =========================================================

    cmd_vel_to_rover_control = Node(
        package='serial_com',
        executable='cmd_vel_to_rover_control',
        name='cmd_vel_to_rover_control',
        output='screen'
    )

    # =========================================================
    # 4. Static TF
    #
    #    base_link -> laser
    # =========================================================

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0',
            '0',
            '0.1',
            '0',
            '0',
            '0',
            'base_link',
            'laser'
        ],
        name='static_tf_pub'
    )

    # =========================================================
    # 5. LiDAR
    # =========================================================

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('sllidar_ros2'),
                'launch',
                'sllidar_a1_launch.py'
            )
        ]),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'laser'
        }.items()
    )

    # =========================================================
    # 6. RealSense D435
    #
    #    Level C 指定使用：
    #
    #    Serial:
    #        146322073872
    #
    #    Color:
    #        640x480 @ 15 FPS
    #
    #    Depth:
    #        640x480 @ 15 FPS
    #
    #    並開啟 depth -> color alignment
    # =========================================================

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'serial_no': "'146322073872'",

            'enable_color': 'true',
            'enable_depth': 'true',

            'rgb_camera.profile': '640x480x15',
            'depth_module.profile': '640x480x15',

            'align_depth.enable': 'true',

        }.items()
    )

    # =========================================================
    # 7. Level C fruit detector
    # =========================================================

    fruit_detector = Node(
        package='level_c',
        executable='fruit_detector',
        name='fruit_detector',
        output='screen'
    )

    # =========================================================
    # 8. Level C mission
    #
    #    綠色 -> SKIP
    #    非綠色 -> GRAB
    #              ↓
    #            STOP
    # =========================================================

    level_c_mission = Node(
        package='level_c',
        executable='level_c_mission',
        name='level_c_mission',
        output='screen'
    )

    # =========================================================
    # 啟動所有 Level C 所需節點
    # =========================================================

    return LaunchDescription([

        uart_node,

        odom_node,

        cmd_vel_to_rover_control,

        static_tf_node,

        lidar_launch,

        realsense_launch,

        fruit_detector,

        level_c_mission,

    ])