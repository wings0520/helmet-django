"""Nạp file head_hog_svm.pkl (chỉ chứa số/mảng numpy + tham số, không chứa đối tượng sklearn).

Chỉ nạp file .pkl do chính bạn huấn luyện: pickle có thể chạy mã tuỳ ý nếu file không đáng tin.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from . import head_preprocessing as hp

REQUIRED_KEYS = ("w", "b", "win", "orientations", "pixels_per_cell", "cells_per_block", "block_norm",
                 "det_thresh", "min_neighbors", "use_refine", "scale_step", "min_win_px", "stride_cells",
                 "group_iou", "nms_iou", "contain_iom", "box_scale", "box_shift_y")


@dataclass(frozen=True)
class HeadModel:
    w: np.ndarray
    b: float
    params: dict
    reg_coef: np.ndarray | None
    reg_intercept: np.ndarray | None

    def refine_predict(self, X):
        return X @ self.reg_coef.T + self.reg_intercept


def load_head_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có model đầu: {path}")
    with open(path, "rb") as f:
        M = pickle.load(f)  # Only load your own trusted model files.
    if not isinstance(M, dict):
        raise ValueError("File .pkl của model đầu phải là dict tham số (xuất từ notebook head-detect-clean).")
    missing = [k for k in REQUIRED_KEYS if k not in M]
    if missing:
        raise ValueError(f"Thiếu khoá trong model đầu: {missing}")
    expected = {"win": hp.WINDOW_SIZE, "orientations": hp.ORIENTATIONS, "pixels_per_cell": hp.PIXELS_PER_CELL,
                "cells_per_block": hp.CELLS_PER_BLOCK, "block_norm": hp.BLOCK_NORM}
    for key, value in expected.items():
        got = tuple(M[key]) if isinstance(value, tuple) else M[key]
        if got != value:
            raise ValueError(f"Tham số HOG '{key}' trong model là {M[key]}, code cần {value}.")
    w = np.asarray(M["w"], dtype=np.float64).ravel()
    if w.size != hp.FEATURE_COUNT:
        raise ValueError(f"Model có {w.size} trọng số; HOG hiện tại tạo {hp.FEATURE_COUNT}.")
    coef = M.get("reg_coef"); intercept = M.get("reg_intercept")
    if coef is not None:
        coef = np.asarray(coef, dtype=np.float64); intercept = np.asarray(intercept, dtype=np.float64)
        if coef.shape != (4, hp.FEATURE_COUNT) or intercept.shape != (4,):
            raise ValueError("Hồi quy hiệu chỉnh khung có kích thước không đúng.")
    params = {k: M[k] for k in REQUIRED_KEYS if k not in ("w", "b")}
    return HeadModel(w=w, b=float(M["b"]), params=params, reg_coef=coef, reg_intercept=intercept)
