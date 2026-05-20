# feature_extraction.py

import cv2
import numpy as np

def extract_hsv(image):
    hsv = cv2.cvtColor(
        (image * 255).astype(np.uint8),
        cv2.COLOR_BGR2HSV
    )

    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    return [h, s, v]