
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Float32MultiArray

class LevelAMission(Node):

    # ============================================================
    # Mission states
    # ============================================================
    SEARCH_PIG = "SEARCH_PIG"
    AVOID_PIG = "AVOID_PIG"
    PASSING_PIG = "PASSING_PIG"
    STOP = "STOP"

    def __init__(self):
        super().__init__("level_a_mission")

        # ========================================================
        # Parameters
        # ========================================================

        # 豬進入這個距離內，才開始避讓
        self.declare_parameter("avoid_start_distance", 0.40)

        # 豬離開這個距離後，才允許重新搜尋下一隻豬
        #
        # 這裡目前是利用 pig_detector 的 forward_x：
        # forward_x < pig_passed_distance
        # 時視為豬已經不在車子的前方。
        self.declare_parameter("pig_passed_distance", 0.10)

        # 確認豬已經離開前方後，需要連續幾次確認
        # 才算真的通過，避免單幀誤判。
        self.declare_parameter("passed_confirm_count", 5)

        self.avoid_start_distance = self.get_parameter(
            "avoid_start_distance"
        ).value

        self.pig_passed_distance = self.get_parameter(
            "pig_passed_distance"
        ).value

        self.passed_confirm_count_required = self.get_parameter(
            "passed_confirm_count"
        ).value

        # ========================================================
        # Internal state
        # ========================================================

        self.state = self.SEARCH_PIG

        # 目前任務鎖定的避讓方向
        #
        # 例如：
        # pig = LEFT
        # target = RIGHT
        #
        # 一旦進入 AVOID_PIG / PASSING_PIG，
        # target_side 不會因為新的豬出現而改變。
        self.target_side = "CENTER"

        # 被鎖定的豬原本在哪一側
        self.locked_pig_side = "NONE"

        # 最新豬資料
        self.pig_detected = False
        self.pig_side = "NONE"
        self.pig_forward_x = None
        self.pig_confidence = 0.0

        # 通過豬的確認計數
        self.passed_confirm_count = 0

        # ========================================================
        # Subscribers
        # ========================================================

        # 穩定後的豬偵測結果
        self.pig_side_sub = self.create_subscription(
            String,
            "/level_a/pig_side",
            self.pig_side_callback,
            10
        )

        # 完整 pig detection 資訊
        #
        # [0] stable_valid
        # [1] raw_detected
        # [2] confidence
        # [3] forward_x_m
        # [4] lateral_y_m
        # [5] vertical_z_m
        # [6] side_code
        # [7] optical_depth
        # [8] bbox x1
        # [9] bbox y1
        # [10] bbox x2
        # [11] bbox y2
        self.pig_detection_sub = self.create_subscription(
            Float32MultiArray,
            "/level_a/pig_detection",
            self.pig_detection_callback,
            10
        )

        # lane_change_controller 回報：
        #
        # MOVING
        # HOLDING
        self.lane_status_sub = self.create_subscription(
            String,
            "/level_a/lane_change_status",
            self.lane_status_callback,
            10
        )

        self.lane_change_status = "UNKNOWN"

        # ========================================================
        # Publisher
        # ========================================================

        # 給 lane_change_controller
        #
        # STOP
        # CENTER
        # LEFT
        # RIGHT
        self.target_side_pub = self.create_publisher(
            String,
            "/level_a/target_side",
            10
        )

        # 額外提供 mission 狀態，方便 RViz / terminal 除錯
        self.mission_status_pub = self.create_publisher(
            String,
            "/level_a/mission_status",
            10
        )

        # ========================================================
        # Timer
        # ========================================================

        # Mission 主迴圈 20 Hz
        self.timer = self.create_timer(
            0.05,
            self.mission_loop
        )

        self.get_logger().info(
            "Level A Mission started."
        )

        self.get_logger().info(
            f"avoid_start_distance = "
            f"{self.avoid_start_distance:.2f} m"
        )

        self.get_logger().info(
            f"pig_passed_distance = "
            f"{self.pig_passed_distance:.2f} m"
        )

        # 啟動時先停止
        self.publish_target_side("STOP")

    # ============================================================
    # Pig detector callbacks
    # ============================================================

    def pig_side_callback(self, msg):
        """
        接收穩定後的豬位置。

        LEFT
        RIGHT
        CENTER
        NONE
        """

        self.pig_side = msg.data

    def pig_detection_callback(self, msg):
        """
        接收完整 pig detection 資訊。
        """

        data = msg.data

        # 防止資料長度不夠
        if len(data) < 8:
            self.get_logger().warning(
                "pig_detection data length < 8"
            )
            return

        # [0] stable_valid
        stable_valid = data[0] > 0.5

        # [1] raw_detected
        raw_detected = data[1] > 0.5

        # [2] confidence
        confidence = data[2]

        # [3] forward_x_m
        forward_x = data[3]

        self.pig_confidence = confidence

        # 我們主要使用 stable_valid
        self.pig_detected = stable_valid

        if stable_valid:
            self.pig_forward_x = forward_x
        else:
            self.pig_forward_x = None

    # ============================================================
    # Lane change status callback
    # ============================================================

    def lane_status_callback(self, msg):
        """
        接收 lane_change_controller 狀態：

        MOVING
        HOLDING
        """

        self.lane_change_status = msg.data

    # ============================================================
    # Main mission state machine
    # ============================================================

    def mission_loop(self):

        # ========================================================
        # SEARCH_PIG
        # ========================================================

        if self.state == self.SEARCH_PIG:

            # 正常搜尋狀態：
            # 保持在中央
            self.publish_target_side("CENTER")

            # 必須有穩定豬偵測
            if not self.pig_detected:
                self.publish_mission_status()
                return

            # 必須知道豬在哪一側
            if self.pig_side not in ["LEFT", "RIGHT"]:
                self.publish_mission_status()
                return

            # 必須知道豬距離
            if self.pig_forward_x is None:
                self.publish_mission_status()
                return

            # 豬還太遠
            if self.pig_forward_x > self.avoid_start_distance:
                self.publish_mission_status()
                return

            # ----------------------------------------------------
            # 開始避讓
            # ----------------------------------------------------

            self.locked_pig_side = self.pig_side

            if self.locked_pig_side == "LEFT":
                self.target_side = "RIGHT"

            elif self.locked_pig_side == "RIGHT":
                self.target_side = "LEFT"

            self.passed_confirm_count = 0

            self.get_logger().info(
                f"🐷 發現豬！"
                f" side={self.locked_pig_side}, "
                f"distance={self.pig_forward_x:.2f} m"
            )

            self.get_logger().info(
                f"➡️ 開始避讓，目標側 = {self.target_side}"
            )

            self.state = self.AVOID_PIG

        # ========================================================
        # AVOID_PIG
        # ========================================================

        elif self.state == self.AVOID_PIG:

            # ★ 最重要的地方
            #
            # 不重新看新的 pig_side 決定方向。
            #
            # target_side 已經鎖定。
            self.publish_target_side(
                self.target_side
            )

            # 等 lane_change_controller 到達目標側
            if self.lane_change_status == "HOLDING":

                self.get_logger().info(
                    f"✅ 已到達 {self.target_side} 側，"
                    f"開始通過豬。"
                )

                self.passed_confirm_count = 0

                self.state = self.PASSING_PIG

        # ========================================================
        # PASSING_PIG
        # ========================================================

        elif self.state == self.PASSING_PIG:

            # ★★★ 非常重要 ★★★
            #
            # 即使現在看到另一隻豬，
            # 也不能改變 target_side。
            #
            # 一直保持在原本避讓的那一側。
            self.publish_target_side(
                self.target_side
            )

            # ----------------------------------------------------
            # 判斷目前這隻豬是否已經通過
            # ----------------------------------------------------

            pig_is_still_in_front = False

            if (
                self.pig_detected
                and self.pig_forward_x is not None
            ):
                if self.pig_forward_x >= self.pig_passed_distance:
                    pig_is_still_in_front = True

            if pig_is_still_in_front:

                # 還沒通過
                self.passed_confirm_count = 0

            else:

                # 豬已經不在前方
                self.passed_confirm_count += 1

                if (
                    self.passed_confirm_count
                    >= self.passed_confirm_count_required
                ):

                    self.get_logger().info(
                        "✅ 已確認通過豬，"
                        "重新搜尋下一隻豬。"
                    )

                    self.locked_pig_side = "NONE"
                    self.target_side = "CENTER"

                    self.passed_confirm_count = 0

                    self.state = self.SEARCH_PIG

        # ========================================================
        # STOP
        # ========================================================

        elif self.state == self.STOP:

            self.publish_target_side("STOP")

        self.publish_mission_status()

    # ============================================================
    # Publish target side
    # ============================================================

    def publish_target_side(self, side):

        msg = String()
        msg.data = side

        self.target_side_pub.publish(msg)

    # ============================================================
    # Publish mission status
    # ============================================================

    def publish_mission_status(self):

        msg = String()

        if self.state == self.SEARCH_PIG:
            msg.data = "SEARCH_PIG"

        elif self.state == self.AVOID_PIG:
            msg.data = (
                f"AVOID_PIG:"
                f"{self.locked_pig_side}"
                f"->"
                f"{self.target_side}"
            )

        elif self.state == self.PASSING_PIG:
            msg.data = (
                f"PASSING_PIG:"
                f"target={self.target_side}"
            )

        elif self.state == self.STOP:
            msg.data = "STOP"

        else:
            msg.data = "UNKNOWN"

        self.mission_status_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = LevelAMission()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # 離開前要求停止
        node.publish_target_side("STOP")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

