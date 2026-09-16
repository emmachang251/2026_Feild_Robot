import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from fruit_color import is_green


class FruitDetector(Node):

    def __init__(self):
        super().__init__('fruit_detector')

        # ========================================
        # 1. ROS Image → OpenCV
        # ========================================

        self.bridge = CvBridge()

        # 訂閱 D435 RGB 影像
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # 計算處理了幾張影像
        self.frame_count = 0

        self.get_logger().info(
            'Fruit detector started.'
        )

    def image_callback(self, msg):

        # ========================================
        # 2. ROS Image → OpenCV BGR image
        # ========================================

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        self.frame_count += 1

        # ========================================
        # 3. BGR → HSV
        # ========================================

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # ========================================
        # 4. 建立彩色物體 Mask
        # ========================================

        # HSV:
        #
        # H = Hue        色相
        # S = Saturation 飽和度
        # V = Value      亮度
        #
        # 目前先只要求：
        # 「顏色夠鮮豔」

        lower = np.array([0, 80, 50])
        upper = np.array([179, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        # ========================================
        # 5. 去除雜訊
        # ========================================

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        # 去除小白點
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # 填補物體內的小黑洞
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ========================================
        # 6. 找輪廓
        # ========================================

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        object_count = 0

        # ========================================
        # 7. 分析每一個物體
        # ========================================

        for i, contour in enumerate(contours):

            area = cv2.contourArea(contour)

            # 太小的區域忽略
            if area < 500:
                continue

            object_count += 1

            # 找 bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # 中心點
            center_x = x + w // 2
            center_y = y + h // 2

            # ========================================
            # 8. 判斷是不是綠色
            # ========================================

            green = is_green(
                image,
                contour
            )

            if green:
                decision = "GREEN - SKIP"
            else:
                decision = "NOT GREEN - GRAB"

            # ========================================
            # 9. 印出結果
            # ========================================

            self.get_logger().info(
                f"Object {object_count}: "
                f"center=({center_x}, {center_y}), "
                f"area={area:.0f}, "
                f"{decision}"
            )

            # ========================================
            # 10. 畫 bounding box
            # ========================================

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                5
            )

            # ========================================
            # 11. 寫上 GRAB / SKIP
            # ========================================

            cv2.putText(
                image,
                decision,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                3
            )

            # ========================================
            # 12. 畫中心點
            # ========================================

            cv2.circle(
                image,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

        # ========================================
        # 13. 儲存結果
        # ========================================

        # 不需要每一幀都存
        # 每 30 幀存一次

        if self.frame_count % 30 == 0:

            result_path = (
                '/home/bme1234/rover_ws/src/level_c/'
                'test_images/d435_result.jpg'
            )

            mask_path = (
                '/home/bme1234/rover_ws/src/level_c/'
                'test_images/d435_mask.jpg'
            )

            cv2.imwrite(
                result_path,
                image
            )

            cv2.imwrite(
                mask_path,
                mask
            )


def main(args=None):

    rclpy.init(args=args)

    node = FruitDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()