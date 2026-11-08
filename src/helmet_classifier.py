import numpy as np
from .helmet_preprocessing import extract_features, CLASS_NAMES
from .model_utils import load_model


class HelmetClassifier:
    def __init__(self, model_path):
        self.model, self.label_map = load_model(model_path, CLASS_NAMES)
        self.score_sign = 1 if self.label_map[self.model.classes_[1]] == "helmet" else -1

    def classify(self, crops, color_order="RGB"):
        if not crops:
            return []
        X = np.asarray([extract_features(crop, color_order) for crop in crops], dtype=np.float32)
        labels = self.model.predict(X)
        margins = np.asarray(self.model.decision_function(X)).reshape(-1) * self.score_sign
        return [{"label": self.label_map[label], "helmet_score": float(score)}
                for label, score in zip(labels, margins)]
