import time
import cv2
import numpy as np
from .head_detector import HeadDetector
from .helmet_classifier import HelmetClassifier


def expanded_box(box, width, height, padding=0.0):
    if not 0 <= padding <= 0.5:
        raise ValueError("Padding phải trong [0, 0.5].")
    x1, y1, x2, y2 = box
    dx = round((x2 - x1) * padding); dy = round((y2 - y1) * padding)
    return max(0, x1-dx), max(0, y1-dy), min(width, x2+dx), min(height, y2+dy)


class HelmetPipeline:
    def __init__(self, head_model_path, helmet_model_path):
        self.detector = HeadDetector(head_model_path)
        self.classifier = HelmetClassifier(helmet_model_path)

    def run(self, image, config=None, padding=0.0, color_order="RGB", progress=None):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.size == 0:
            raise ValueError("Ảnh đầu vào phải là uint8, không rỗng.")
        if color_order not in ("RGB", "BGR"):
            raise ValueError("color_order phải là RGB hoặc BGR.")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = image.copy() if color_order == "RGB" else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Cần ảnh RGB/BGR 3 kênh hoặc ảnh xám.")
        start = time.perf_counter()
        detections, stats = self.detector.detect(rgb, config, progress=progress)
        height, width = rgb.shape[:2]
        crop_boxes = [expanded_box(d["box"], width, height, padding) for d in detections]
        # Always crop raw RGB. Never crop gamma/CLAHE-processed windows.
        crops = [rgb[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in crop_boxes]
        classifications = self.classifier.classify(crops)
        annotated = rgb.copy()
        results = []
        for index, (detection, crop_box, classification) in enumerate(zip(detections, crop_boxes, classifications), 1):
            result = {"id": index, **detection, "crop_box": crop_box, **classification}
            results.append(result)
            x1, y1, x2, y2 = detection["box"]
            color = (30, 190, 90) if classification["label"] == "helmet" else (235, 65, 65)
            cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), color, 2)
            cv2.putText(annotated, f'{index}: {classification["label"]}',
                        (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        stats["seconds"] = time.perf_counter() - start
        return {"annotated_rgb": annotated, "detections": results, "crops_rgb": crops, "stats": stats}
