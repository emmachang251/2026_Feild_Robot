#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from enum import Enum


# =========================
# Level C Mission 狀態
# =========================

class MissionState(Enum):

    # 車子正在搜尋水果
    SEARCHING = 'SEARCHING'

    # 已經發現要摘的水果，車子已經停止
    STOPPED = 'STOPPED'


# =========================
# Level C Mission
# =========================

class LevelCMission(Node):

    def __init__(self):
        super().__init__('level_c_mission')

        # =========================
        # 訂閱 fruit_detector
        # =========================

        self.detection_subscription = self.create_subscription(
            Float32MultiArray,
            '/level_c/fruit_detections',
            self.detection_callback,
            10
        )

        # =========================
        # 發布底盤速度
        #
        # Level C 不使用 Level A 的控制節點
        # 直接使用整台車共用的 /cmd_vel
        # =========================

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # =========================
        # 初始狀態
        # =========================

        self.state = MissionState.SEARCHING

        self.get_logger().info(
            'Level C mission started.'
        )

        self.get_logger().info(
            'State: SEARCHING'
        )

    # =========================
    # 水果偵測 callback
    # =========================

    def detection_callback(self, msg):

        data = msg.data

        # =========================
        # 如果已經停車
        # =========================

        if self.state == MissionState.STOPPED:

            # 現階段先不做其他事情
            #
            # 後面加入手臂抓取之後，
            # 這裡才會繼續進行抓取流程。

            return

        # =========================
        # SEARCHING 狀態
        # =========================

        if self.state == MissionState.SEARCHING:

            # -------------------------
            # 沒有偵測到物體
            # -------------------------

            if len(data) == 0:

                return

            # -------------------------
            # 確認資料格式
            #
            # 每一個物體有 5 個數值：
            #
            # X
            # Y
            # Z
            # green_ratio
            # is_green
            # -------------------------

            if len(data) % 5 != 0:

                self.get_logger().warn(
                    f'Invalid detection data length: {len(data)}'
                )

                return

            # =========================
            # 計算偵測到幾個物體
            # =========================

            object_count = len(data) // 5

            self.get_logger().info(
                f'Found {object_count} object(s).'
            )

            # =========================
            # 檢查所有物體
            # =========================

            for i in range(object_count):

                index = i * 5

                X = data[index]
                Y = data[index + 1]
                Z = data[index + 2]

                green_ratio = data[index + 3]
                is_green = data[index + 4]

                # =========================
                # 綠色水果
                # =========================

                if is_green >= 0.5:

                    self.get_logger().info(
                        f'Object {i + 1}: '
                        f'GREEN -> SKIP'
                    )

                    continue

                # =========================
                # 非綠色水果
                #
                # 代表這顆水果需要摘取
                # =========================

                self.get_logger().info(
                    f'Object {i + 1}: '
                    f'NOT GREEN -> GRAB'
                )

                self.get_logger().info(
                    f'Target XYZ = '
                    f'({X:.3f}, {Y:.3f}, {Z:.3f}) m'
                )

                self.get_logger().info(
                    f'Green ratio = {green_ratio:.2f}'
                )

                # =========================
                # 發出停止命令
                # =========================

                self.stop_robot()

                # =========================
                # 修改 Mission 狀態
                # =========================

                self.state = MissionState.STOPPED

                self.get_logger().info(
                    '================================'
                )

                self.get_logger().info(
                    'TARGET FOUND!'
                )

                self.get_logger().info(
                    'Robot stopped.'
                )

                self.get_logger().info(
                    'State: STOPPED'
                )

                self.get_logger().info(
                    'Waiting for arm grabbing process.'
                )

                self.get_logger().info(
                    '================================'
                )

                # -------------------------
                # 找到第一顆要摘的水果後
                # 就不用繼續檢查其他物體
                # -------------------------

                break

    # =========================
    # 停止車子
    # =========================

    def stop_robot(self):

        stop_cmd = Twist()

        # =========================
        # 線速度
        # =========================

        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0

        # =========================
        # 角速度
        # =========================

        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        # =========================
        # 發布停止命令
        # =========================

        self.cmd_vel_publisher.publish(stop_cmd)

        self.get_logger().info(
            'STOP command published to /cmd_vel.'
        )


# =========================
# main
# =========================

def main(args=None):

    rclpy.init(args=args)

    node = LevelCMission()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


# =========================
# Python 程式進入點
# =========================

if __name__ == '__main__':
    main()

