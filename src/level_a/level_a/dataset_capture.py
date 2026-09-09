#!/usr/bin/env python3

import re
from datetime import datetime
from pathlib import Path

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DatasetCapture(Node):
    """
    從D435 RGB Topic儲存YOLO訓練照片。

    發送場景名稱到：
        /level_a/capture_label

    每次命令預設儲存1張原始解析度JPEG。
    """

    def __init__(self) -> None:
        super().__init__('dataset_capture')

        self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw'
        )
        self.declare_parameter(
            'command_topic',
            '/level_a/capture_label'
        )
        self.declare_parameter(
            'status_topic',
            '/level_a/capture_status'
        )
        self.declare_parameter(
            'output_directory',
            '~/pig_dataset_raw'
        )
        self.declare_parameter(
            'images_per_command',
            1
        )
        self.declare_parameter(
            'capture_interval_s',
            0.40
        )
        self.declare_parameter(
            'jpeg_quality',
            95
        )

        self.image_topic = str(
            self.get_parameter(
                'image_topic'
            ).value
        )
        self.command_topic = str(
            self.get_parameter(
                'command_topic'
            ).value
        )
        self.status_topic = str(
            self.get_parameter(
                'status_topic'
            ).value
        )

        self.output_directory = Path(
            str(
                self.get_parameter(
                    'output_directory'
                ).value
            )
        ).expanduser()

        self.images_per_command = int(
            self.get_parameter(
                'images_per_command'
            ).value
        )
        self.capture_interval_s = float(
            self.get_parameter(
                'capture_interval_s'
            ).value
        )
        self.jpeg_quality = int(
            self.get_parameter(
                'jpeg_quality'
            ).value
        )

        if self.images_per_command <= 0:
            raise ValueError(
                'images_per_command 必須大於0'
            )

        if self.capture_interval_s <= 0.0:
            raise ValueError(
                'capture_interval_s 必須大於0'
            )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                'jpeg_quality 必須位於1～100'
            )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_stamp = None
        self.last_saved_stamp = None

        self.pending_count = 0
        self.capture_index = 0
        self.current_label = ''
        self.session_stamp = ''
        self.next_capture_time_ns = 0

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10
        )

        self.capture_timer = self.create_timer(
            0.05,
            self.capture_timer_callback
        )

        self.get_logger().info(
            'Dataset capture started. '
            f'image={self.image_topic}, '
            f'output={self.output_directory}'
        )

    def publish_status(
        self,
        text: str
    ) -> None:
        message = String()
        message.data = text
        self.status_pub.publish(message)
        self.get_logger().info(text)

    def image_callback(
        self,
        msg: Image
    ) -> None:
        self.latest_image = msg

        self.latest_image_stamp = (
            int(msg.header.stamp.sec)
            * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

    @staticmethod
    def sanitize_label(
        raw_label: str
    ) -> str:
        label = raw_label.strip().lower()

        label = re.sub(
            r'[^a-z0-9_-]+',
            '_',
            label
        )

        label = label.strip('_')

        return label

    def command_callback(
        self,
        msg: String
    ) -> None:
        label = self.sanitize_label(
            msg.data
        )

        if not label:
            self.publish_status(
                'ERROR: capture label is empty'
            )
            return

        if self.pending_count > 0:
            self.publish_status(
                'BUSY: previous capture is not finished'
            )
            return

        if self.latest_image is None:
            self.publish_status(
                'ERROR: no RGB image received'
            )
            return

        self.current_label = label

        self.session_stamp = (
            datetime.now().strftime(
                '%Y%m%d_%H%M%S_%f'
            )
        )

        self.pending_count = (
            self.images_per_command
        )
        self.capture_index = 0

        self.next_capture_time_ns = (
            self.get_clock().now().nanoseconds
        )

        self.publish_status(
            f'START: label={self.current_label}, '
            f'count={self.images_per_command}'
        )

    def capture_timer_callback(self) -> None:
        if self.pending_count <= 0:
            return

        now_ns = (
            self.get_clock().now().nanoseconds
        )

        if now_ns < self.next_capture_time_ns:
            return

        if self.latest_image is None:
            return

        # 避免同一個影像Frame被重複儲存
        if (
            self.latest_image_stamp
            == self.last_saved_stamp
        ):
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                self.latest_image,
                desired_encoding='bgr8'
            )
        except Exception as error:
            self.pending_count = 0
            self.publish_status(
                f'ERROR: CvBridge failed: {error}'
            )
            return

        self.capture_index += 1

        filename = (
            f'{self.current_label}_'
            f'{self.session_stamp}_'
            f'{self.capture_index:02d}.jpg'
        )

        output_path = (
            self.output_directory
            / filename
        )

        saved = cv2.imwrite(
            str(output_path),
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.jpeg_quality
            ]
        )

        if not saved:
            self.pending_count = 0
            self.publish_status(
                f'ERROR: failed to save {output_path}'
            )
            return

        self.last_saved_stamp = (
            self.latest_image_stamp
        )

        self.pending_count -= 1

        self.publish_status(
            f'SAVED: {output_path} '
            f'({frame.shape[1]}x{frame.shape[0]})'
        )

        if self.pending_count > 0:
            interval_ns = int(
                self.capture_interval_s
                * 1_000_000_000
            )

            self.next_capture_time_ns = (
                now_ns
                + interval_ns
            )
        else:
            self.publish_status(
                f'DONE: label={self.current_label}, '
                f'saved={self.images_per_command}'
            )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = DatasetCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()