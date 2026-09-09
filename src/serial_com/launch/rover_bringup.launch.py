import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # 1. 啟動 UART 節點 (與 STM32 通訊)
    uart_node = Node(
        package='serial_com', 
        executable='uart_node',
        name='uart_node',
        output='screen'
    )

    # 2. 啟動 Odometry 里程計節點
    odom_node = Node(
        package='serial_com', 
        executable='odom_node',
        name='odom_node',
        output='screen'
    )

    # 3. 啟動靜態座標轉換 (Static TF: base_link -> laser)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'],
        name='static_tf_pub'
    )

    # 4. 啟動光達 (原廠 Launch 檔)，注意這裡把它的輸出重新命名為 /scan_raw
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_a1_launch.py')
        ]),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'laser'
        }.items(),
        # remappings=[
        #     ('/scan', '/scan_raw')  # ★ 把光達原始資料攔截下來改名
        # ]
    )

    # 4-1. 🌟 新增雷射過濾器：過濾掉後方車體遮擋，只保留 -90度 到 +90度
    # laser_filter_node = Node(
    #     package='laser_filters',
    #     executable='scan_to_scan_filter_chain',
    #     name='laser_filter',
    #     parameters=['/home/bme1234/rover_ws/src/serial_com/config/laser_filter.yaml'],
    #     remappings=[
    #         ('/scan_in', '/scan_raw'),  # 接收有雜訊的原始光達資料
    #         ('/scan', '/scan')          # 吐出乾淨、只看前方的資料給 SLAM
    #     ]
    # )

    # 5. 啟動 RealSense D435i 相機
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ]),
        launch_arguments={
            'depth_module.profile': '424x240x15', 
            'rgb_camera.profile': '424x240x15', 
        }.items()
    )

    # 把所有任務打包回傳
    return LaunchDescription([
        uart_node,
        odom_node,
        static_tf_node,
        lidar_launch,
        #laser_filter_node,  # ★ 把過濾器加入啟動清單
        realsense_launch
    ])