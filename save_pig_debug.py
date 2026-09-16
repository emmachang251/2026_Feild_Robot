#!/usr/bin/env python3

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DebugImageSaver(Node):

    def __init__(self):
        super().__init__('debug_image_saver')

        self.bridge = CvBridge()
        self.saved = False

        self.subscription = self.create_subscription(
            Image,
            '/level_a/pig_debug_image',
            self.image_callback,
            10,
        )

        self.get_logger().info(
            'Waiting for /level_a/pig_debug_image ...'
        )

    def image_callback(self, message):
        if self.saved:
            return

        image = self.bridge.imgmsg_to_cv2(
            message,
            desired_encoding='bgr8',
        )

        output_path = '/home/bme1234/rover_ws/pig_debug.jpg'

        if not cv2.imwrite(output_path, image):
            self.get_logger().error('Failed to save image')
            return

        self.saved = True
        self.get_logger().info(f'Saved: {output_path}')


def main():
    rclpy.init()
    node = DebugImageSaver()

    while rclpy.ok() and not node.saved:
        rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()