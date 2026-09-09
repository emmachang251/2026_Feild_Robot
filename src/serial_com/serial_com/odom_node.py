#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

from rover_interfaces.msg import RoverSensor
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomNode(Node):
    def __init__(self):
        super().__init__('odom_node')

        # 參數設置 (必須與你的 cmd_vel 節點一致)
        self.track_advance_m = 0.14  # 轉一圈前進 14 cm
        self.track_width_m = 0.36    # 左右輪距 36 cm

        # 訂閱來自 UART Node 的感測器資料
        self.sensor_sub = self.create_subscription(
            RoverSensor,
            'sensor_data',
            self.sensor_callback,
            10
        )

        # 發布 Odometry 與 TF
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 里程計狀態變數
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()

        self.get_logger().info("里程計 (Odometry) 節點已啟動")

    def sensor_callback(self, msg: RoverSensor):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0:
            return

        # 取得左右輪 RPM (需確認 STM32 傳上來的單位是否需轉換，這裡假設為真實 RPM)
        rpm_left = float(msg.encoder_speed_left) / 10.0
        rpm_right = float(msg.encoder_speed_right) / 10.0

        # 1. 計算左右輪真實線速度 (m/s)
        v_left = (rpm_left / 60.0) * self.track_advance_m
        v_right = (rpm_right / 60.0) * self.track_advance_m

        # 2. 計算車體中心線速度與角速度
        v_x = (v_right + v_left) / 2.0
        v_theta = (v_right - v_left) / self.track_width_m

        # 3. 運動學積分更新座標 (Euler Integration)
        # delta_x = v_x * cos(theta) * dt
        # delta_y = v_x * sin(theta) * dt
        # delta_theta = v_theta * dt
        self.x += v_x * math.cos(self.theta) * dt
        self.y += v_x * math.sin(self.theta) * dt
        self.theta += v_theta * dt

        # 4. 發布 Odometry 訊息
        self.publish_odom(current_time, v_x, v_theta)

        # 5. 發布 TF 轉換 (最重要的一步，讓 Foxglove 知道車子移動了)
        self.publish_tf(current_time)

    def publish_odom(self, current_time, v_x, v_theta):
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # 設置位置
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        # 將 theta (Yaw) 轉成 Quaternion (四元數)
        odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        # 設置速度
        odom.twist.twist.linear.x = v_x
        odom.twist.twist.angular.z = v_theta

        self.odom_pub.publish(odom)

    def publish_tf(self, current_time):
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        t.transform.rotation.z = math.sin(self.theta / 2.0)
        t.transform.rotation.w = math.cos(self.theta / 2.0)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()