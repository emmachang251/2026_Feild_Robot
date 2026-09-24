#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import cv2
import numpy as np

from .fruit_color import get_green_ratio


class FruitDetector(Node):

    def __init__(self):
        super().__init__('fruit_detector')

        # ROS Image ↔ OpenCV 影像轉換工具
        self.bridge = CvBridge()

        # =========================
        # RGB camera
        # =========================
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        # =========================
        # Aligned depth
        # =========================
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # =========================
        # CameraInfo
        # =========================
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # 最近收到的 depth image
        self.latest_depth = None

        # 相機內參
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # depth scale
        # D435 已經確認：
        # 1 depth unit ≈ 0.001 m
        self.depth_scale = 0.001

        self.frame_count = 0
        # ==========================================
        # Debug image publisher
        # ==========================================
        self.image_publisher = self.create_publisher(Image,'/level_c/debug_image',10)

        self.get_logger().info('Fruit detector started.')

        # ==========================================
        # Fruit detection result publisher
        # ==========================================
        self.detection_publisher = self.create_publisher(Float32MultiArray,'/level_c/fruit_detections',10)

    # ==========================================
    # RGB callback
    # ==========================================
    def color_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        self.frame_count += 1

        # 如果 depth 或 CameraInfo 還沒準備好
        if self.latest_depth is None:
            return

        if self.fx is None:
            return

        # ======================================
        # BGR → HSV
        # ======================================
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # ======================================
        # 找有顏色的物體
        # ======================================
        lower = np.array([0,80,50])

        upper = np.array([179,255,255])

        mask = cv2.inRange(hsv,lower,upper)

        # ======================================
        # 去除小雜訊
        # ======================================
        kernel = np.ones((5, 5),np.uint8)

        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

        # ======================================
        # 找輪廓
        # ======================================
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        object_count = 0

        # 儲存這一幀所有水果的結果
        detections = [] 

        for contour in contours:

            area = cv2.contourArea(contour)

            # 太小的物體直接忽略
            if area < 500:
                continue

            object_count += 1

            # ==================================
            # Bounding box
            # ==================================
            x, y, w, h = cv2.boundingRect(
                contour
            )

            # ==================================
            # 物體中心
            # ==================================
            center_x = x + w // 2
            center_y = y + h // 2

            # ==================================
            # 綠色比例
            # ==================================
            green_ratio = get_green_ratio(image,contour)

            # ==================================
            # 取得 XYZ
            # ==================================
            xyz = self.get_xyz(
                center_x,
                center_y
            )

            # ==================================
            # 綠色判斷
            # ==================================
            if green_ratio > 0.40:
                decision = 'GREEN - SKIP'
                is_green = 1.0
            else:
                decision = 'NOT GREEN - GRAB'
                is_green = 0.0

            # ==================================
            # 印出結果
            # ==================================
            if xyz is not None:

                X, Y, Z = xyz
                detections.extend([float(X),float(Y),float(Z),float(green_ratio),float(is_green)])

                self.get_logger().info(
                    f'Object {object_count}: '
                    f'center=({center_x}, {center_y}), '
                    f'XYZ=({X:.3f}, {Y:.3f}, {Z:.3f}) m, '
                    f'green_ratio={green_ratio:.2f}, '
                    f'{decision}'
                )

            else:

                self.get_logger().warn(
                    f'Object {object_count}: '
                    f'center=({center_x}, {center_y}), '
                    f'XYZ unavailable, '
                    f'green_ratio={green_ratio:.2f}, '
                    f'{decision}'
                )

            # ==================================
            # 畫框
            # ==================================
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                5
            )

            # ==================================
            # 顯示文字
            # ==================================
            cv2.putText(
                image,
                decision,
                (x, y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                3
            )

            if xyz is not None:

                X, Y, Z = xyz

                xyz_text = (
                    f'XYZ=({X:.2f}, '
                    f'{Y:.2f}, '
                    f'{Z:.2f})m'
                )

                cv2.putText(
                    image,
                    xyz_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

            # ==================================
            # 中心點
            # ==================================
            cv2.circle(image,(center_x, center_y),5,(0, 0, 255),-1)

        # ======================================
        # 發布這一幀的水果偵測結果
        # ======================================
        detection_msg = Float32MultiArray()
        detection_msg.data = detections
        self.detection_publisher.publish(detection_msg)


        # ======================================
        # 每 30 幀儲存一次
        # ======================================
        if self.frame_count % 30 == 0:

            result_path = ('/home/bme1234/rover_ws/src/level_c/''test_images/d435_result.jpg')

            mask_path = ('/home/bme1234/rover_ws/src/level_c/''test_images/d435_mask.jpg')

            cv2.imwrite(result_path,image)

            cv2.imwrite(mask_path,mask)
            # 發布處理後的影像，讓 Foxglove 可以看到
       
        
       
    # ==========================================
    # Depth callback
    # ==========================================
    def depth_callback(self, msg):

        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

    # ==========================================
    # CameraInfo callback
    # ==========================================
    def camera_info_callback(self, msg):

        # 只需要設定一次
        if self.fx is not None:
            return

        self.fx = msg.k[0]
        self.fy = msg.k[4]

        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.get_logger().info(
            f'Camera intrinsics received: '
            f'fx={self.fx:.3f}, '
            f'fy={self.fy:.3f}, '
            f'cx={self.cx:.3f}, '
            f'cy={self.cy:.3f}'
        )

    # ==========================================
    # 取得 XYZ
    # ==========================================
    def get_xyz(self, u, v):

        if self.latest_depth is None:
            return None

        if self.fx is None:
            return None

        height, width = self.latest_depth.shape

        # 防止座標超出影像範圍
        if u < 0 or u >= width:
            return None

        if v < 0 or v >= height:
            return None

        # ======================================
        # 取中心附近 5×5 depth
        # ======================================
        radius = 2

        x1 = max(0, u - radius)
        x2 = min(width, u + radius + 1)

        y1 = max(0, v - radius)
        y2 = min(height, v + radius + 1)

        depth_patch = self.latest_depth[
            y1:y2,
            x1:x2
        ]

        # ======================================
        # 只保留有效 depth
        # 0 = 沒有深度
        # ======================================
        valid_depth = depth_patch[
            depth_patch > 0
        ]

        if len(valid_depth) == 0:
            return None

        # ======================================
        # 使用 median
        # 避免單一 pixel 雜訊
        # ======================================
        depth_raw = np.median(
            valid_depth
        )

        # ======================================
        # depth unit → 公尺
        # ======================================
        Z = depth_raw * self.depth_scale

        # ======================================
        # Pixel → Camera XYZ
        # ======================================
        X = (
            (u - self.cx)
            * Z
            / self.fx
        )

        Y = (
            (v - self.cy)
            * Z
            / self.fy
        )

        return X, Y, Z


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