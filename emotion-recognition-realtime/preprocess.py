import cv2
import numpy as np

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped
