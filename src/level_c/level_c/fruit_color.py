import cv2
import numpy as np


def is_green(image, contour):
    """
    判斷一個物體是不是綠色。

    image:
        原始 BGR 影像

    contour:
        fruit_detector 找到的物體輪廓

    return:
        True  = 綠色
        False = 非綠色
    """

    # 建立一張和原圖一樣大的黑色 mask
    object_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 把目前這個物體的 contour 填滿成白色
    cv2.drawContours(
        object_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 綠色的 HSV 範圍
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([85, 255, 255])

    # 找出綠色像素
    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # 只保留「物體裡面的綠色」
    green_pixels = cv2.bitwise_and(
        green_mask,
        object_mask
    )

    # 計算物體總像素數
    object_pixels = cv2.countNonZero(object_mask)

    # 計算物體裡面綠色像素數
    green_pixel_count = cv2.countNonZero(green_pixels)

    # 避免除以 0
    if object_pixels == 0:
        return False

    # 綠色比例
    green_ratio = green_pixel_count / object_pixels

    print(f"Green ratio: {green_ratio:.2f}")

    # 如果超過 40% 是綠色，就判定為綠色
    return green_ratio > 0.40