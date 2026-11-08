"""Tiền xử lý cho bộ phân loại mũ (giữ nguyên bản 48x48 + gamma + CLAHE + HOG 900).

Trước đây file này dùng chung với bộ phát hiện đầu; nay bộ phát hiện đầu chuyển sang
HOG 64x64 + LinearSVM (head_preprocessing.py) nên phần này tự chứa để không bị ảnh hưởng.
Crop RGB thô phải đi vào đây, không bao giờ đưa crop đã qua gamma/CLAHE.
"""
import cv2
import numpy as np
from skimage.feature import hog
from .head_preprocessing import to_gray

WINDOW_SIZE = (48, 48)
FEATURE_COUNT = 900
CLASS_NAMES = ["No_helmet", "helmet"]


def preprocess(image, color_order="RGB", return_stages=False):
    gray = to_gray(image, color_order)
    if gray.shape != (48, 48):
        gray = cv2.resize(gray, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
    corrected = gray.copy()
    m = gray[8:44, 8:40].mean()
    if not 90 <= m <= 160:
        m = np.clip(m / 255.0, 1e-3, 0.999)
        g = np.clip(np.log(128 / 255.0) / np.log(m), 0.6, 2.0)
        corrected = (255 * (gray / 255.0) ** g).clip(0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(corrected, (3, 3), sigmaX=0, sigmaY=0)
    final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(blurred)
    if return_stages:
        return {"gray": gray, "gamma": corrected, "gaussian": blurred, "clahe": final}
    return final


def extract_features(image, color_order="RGB"):
    return hog(preprocess(image, color_order), orientations=9,
               pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", transform_sqrt=False,
               feature_vector=True).astype(np.float32)
