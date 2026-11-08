"""Tiền xử lý + HOG cho bộ phát hiện đầu HOG + LinearSVM (khớp head-detect-clean.ipynb).

Khác bản 48x48 cũ: cửa sổ 64x64, ảnh xám thô, KHÔNG gamma / Gaussian / CLAHE.
Mọi hằng số dưới đây phải trùng với file .pkl (được kiểm tra khi nạp model).
"""
import cv2
import numpy as np
from skimage.feature import hog

WINDOW_SIZE = (64, 64)          # (rộng, cao)
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
BLOCK_NORM = "L2-Hys"
BLOCKS_X = WINDOW_SIZE[0] // PIXELS_PER_CELL[0] - CELLS_PER_BLOCK[0] + 1   # 7
BLOCKS_Y = WINDOW_SIZE[1] // PIXELS_PER_CELL[1] - CELLS_PER_BLOCK[1] + 1   # 7
FEATURE_COUNT = BLOCKS_X * BLOCKS_Y * CELLS_PER_BLOCK[0] * CELLS_PER_BLOCK[1] * ORIENTATIONS  # 1764
CLASS_NAMES = ["background", "head"]


def to_gray(image, color_order="RGB"):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Ảnh phải là uint8 và không rỗng.")
    if image.ndim == 2:
        return image.copy()
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order phải là RGB hoặc BGR.")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh xám, RGB/BGR hoặc RGBA/BGRA.")
    mode = {("RGB", 3): cv2.COLOR_RGB2GRAY,
            ("BGR", 3): cv2.COLOR_BGR2GRAY,
            ("RGB", 4): cv2.COLOR_RGBA2GRAY,
            ("BGR", 4): cv2.COLOR_BGRA2GRAY}[color_order, image.shape[2]]
    return cv2.cvtColor(image, mode)


def to_window(gray):
    """Resize về 64x64 giống `to_win` trong notebook."""
    if gray.shape[1] > WINDOW_SIZE[0]:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(gray, WINDOW_SIZE, interpolation=interp)


def hog_vector(gray_window):
    return hog(gray_window, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
               cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM,
               feature_vector=True).astype(np.float32)


def hog_blocks(gray):
    """Lưới block HOG của cả ảnh, shape (ny, nx, D) — tính một lần cho mỗi tầng pyramid."""
    b = hog(gray, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM, feature_vector=False)
    return b.reshape(b.shape[0], b.shape[1], -1)


def extract_features(image, color_order="RGB"):
    """HOG 1764 chiều của một ảnh crop bất kỳ (resize về 64x64)."""
    return hog_vector(to_window(to_gray(image, color_order)))
