import cv2
import numpy as np


def get_green_ratio(image, contour):
    """
    計算一個物體裡面有多少比例是綠色。

    image:
        原始 BGR 影像

    contour:
        fruit_detector 找到的物體輪廓

    return:
        green_ratio:
            0.0 ~ 1.0
    """

    # ========================================
    # 1. 建立物體 mask
    # ========================================

    object_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        object_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ========================================
    # 2. BGR → HSV
    # ========================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # ========================================
    # 3. 綠色 HSV 範圍
    # ========================================

    lower_green = np.array([
        35,
        50,
        40
    ])

    upper_green = np.array([
        85,
        255,
        255
    ])

    # ========================================
    # 4. 找綠色像素
    # ========================================

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # ========================================
    # 5. 只保留物體裡面的綠色
    # ========================================

    green_pixels = cv2.bitwise_and(
        green_mask,
        object_mask
    )

    # ========================================
    # 6. 計算比例
    # ========================================

    object_pixels = cv2.countNonZero(
        object_mask
    )

    green_pixel_count = cv2.countNonZero(
        green_pixels
    )

    if object_pixels == 0:
        return 0.0

    green_ratio = (
        green_pixel_count /
        object_pixels
    )

    return green_ratio