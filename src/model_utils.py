"""Load trusted local joblib models; reject incompatible pipelines."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


def load_model(path, expected_names):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có model: {path}")
    payload = joblib.load(path)  # Only load your own trusted model files.
    model = payload.get("model") if isinstance(payload, dict) else payload
    names = list(payload.get("class_names", expected_names)) if isinstance(payload, dict) else list(expected_names)
    if sorted(names) != sorted(expected_names):
        raise ValueError(f"Sai lớp model: {names}; cần {expected_names}")
    if isinstance(payload, dict):
        if tuple(payload.get("window_size", (48, 48))) != (48, 48):
            raise ValueError("Model dùng kích thước khác 48x48. Không dùng model 64x64 với code này.")
        # Rich metadata from the earlier 64x64 notebook is intentionally not
        # interpreted as the current central-ROI gamma preprocessing.
        if "preprocess_config" in payload:
            raise ValueError("Bundle có cấu hình tiền xử lý riêng. Hãy xuất pipeline đúng từ code Kaggle 48x48 hiện tại.")
    if not isinstance(model, Pipeline):
        raise ValueError("Cần lưu TOÀN BỘ pipeline StandardScaler + SVC, không chỉ SVC.")
    steps = [step for _, step in model.steps]
    if len(steps) != 2 or not isinstance(steps[0], StandardScaler) or not isinstance(steps[1], SVC):
        raise ValueError("Code này cần pipeline StandardScaler + SVC, HOG 900 đặc trưng, không PCA.")
    for step in steps:
        check_is_fitted(step)
    if int(model.n_features_in_) != 900:
        raise ValueError(f"Model cần {model.n_features_in_} đặc trưng; HOG hiện tại tạo 900.")
    classes = list(model.classes_)
    if len(classes) != 2:
        raise ValueError("Cần model hai lớp.")
    if all(isinstance(c, (str, np.str_)) for c in classes):
        mapping = {c: str(c) for c in classes}
        if set(mapping.values()) != set(expected_names):
            raise ValueError("Tên lớp trong model không đúng.")
    else:
        if set(classes) != {0, 1}:
            raise ValueError("Nhãn số cần là 0 và 1.")
        mapping = {c: names[int(c)] for c in classes}
    return model, mapping
