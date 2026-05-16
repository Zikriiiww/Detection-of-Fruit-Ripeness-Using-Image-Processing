# preprocessing.py

import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Resize sesuai VGG16
    img = cv2.resize(img, (224, 224))

    # Normalisasi
    img = img / 255.0

    return img