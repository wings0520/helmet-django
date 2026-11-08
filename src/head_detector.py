"""Phát hiện đầu: sliding window + image pyramid + HOG/LinearSVM (khớp head-detect-clean.ipynb).

Quy trình: quét pyramid (HOG tính một lần mỗi tầng) -> gộp cụm -> lọc min_neighbors ->
hiệu chỉnh khung (Ridge) -> NMS -> loại khung lồng nhau -> cắt theo ảnh.
Giá trị mặc định của DetectionConfig lấy từ head_hog_svm.pkl (xem tests kiểm tra khớp).
"""
from dataclasses import dataclass
import cv2
import numpy as np
from . import head_preprocessing as hp
from .head_model import load_head_model

WIN_W, WIN_H = hp.WINDOW_SIZE
PPC_X, PPC_Y = hp.PIXELS_PER_CELL


@dataclass(frozen=True)
class DetectionConfig:
    min_win_px: int = 40            # đầu nhỏ nhất cần tìm (px)
    scale_step: float = 1.1         # tỉ lệ giữa các tầng pyramid
    stride_cells: int = 1           # bước trượt (đơn vị cell, 1 cell = 8 px trên tầng pyramid)
    det_thresh: float = 0.8         # ngưỡng điểm SVM
    min_neighbors: int = 1          # cụm cần >= N cửa sổ mới được giữ
    nms_iou: float = 0.3
    # None = dùng giá trị lưu trong file .pkl
    group_iou: float | None = None
    contain_iom: float | None = None
    use_refine: bool | None = None
    box_scale: tuple | None = None
    box_shift_y: float | None = None
    # giới hạn an toàn (không có trong .pkl)
    max_image_side: int = 1280      # ảnh lớn hơn được thu nhỏ khi quét (như notebook), khung quy về ảnh gốc
    max_windows: int = 3_000_000
    max_candidates: int = 20_000

    def validate(self):
        if not WIN_W // 2 <= self.min_win_px <= 2048:
            raise ValueError(f"min_win_px phải trong [{WIN_W // 2}, 2048].")
        if not np.isfinite(self.scale_step) or self.scale_step <= 1:
            raise ValueError("scale_step phải lớn hơn 1.")
        if self.stride_cells < 1:
            raise ValueError("stride_cells phải >= 1.")
        if self.min_neighbors < 1:
            raise ValueError("min_neighbors phải >= 1.")
        for name in ("nms_iou", "group_iou", "contain_iom"):
            v = getattr(self, name)
            if v is not None and not (np.isfinite(v) and 0 <= v <= 1):
                raise ValueError(f"{name} phải trong [0, 1].")
        if not np.isfinite(self.det_thresh):
            raise ValueError("det_thresh không hợp lệ.")
        if self.max_image_side < WIN_W or self.max_windows < 1 or self.max_candidates < 1:
            raise ValueError("Giới hạn kích thước ảnh, cửa sổ hoặc ứng viên không hợp lệ.")


def iou_one(b, B):
    iw = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def nms(boxes, scores, iou_threshold=0.3):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable"); keep = []
    while order.size:
        i, rest = order[0], order[1:]; keep.append(int(i))
        order = rest[iou_one(boxes[i], boxes[rest]) <= iou_threshold]
    return keep


def group_boxes(dets, thr, group_iou):
    """Nx5 (x1,y1,x2,y2,score) -> Mx6 (..., score_max, số cửa sổ trong cụm); khung = trung bình có trọng số."""
    d = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
    d = d[d[:, 4] >= thr]
    if len(d) == 0:
        return np.zeros((0, 6))
    d = d[np.argsort(-d[:, 4], kind="stable")]
    alive = np.ones(len(d), bool); out = []
    for i in range(len(d)):
        if not alive[i]:
            continue
        mem = alive & (iou_one(d[i, :4], d[:, :4]) >= group_iou); mem[i] = True
        wts = d[mem, 4] - thr + 0.05
        out.append([*(d[mem, :4] * wts[:, None]).sum(0) / wts.sum(), d[i, 4], mem.sum()])
        alive &= ~mem
    return np.array(out)


def suppress_contained(boxes, scores, contain_iom):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]); keep = []
    for i in np.argsort(-scores, kind="stable"):
        ok = True
        for j in keep:
            iw = max(0.0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]))
            ih = max(0.0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
            if iw * ih / (min(area[i], area[j]) + 1e-9) >= contain_iom:
                ok = False; break
        if ok:
            keep.append(int(i))
    return keep


def crop_window(gray, box):
    """Cắt theo khung bất kỳ (có thể tràn ra ngoài ảnh -> đệm replicate), resize về 64x64."""
    H, W = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x2 = max(x2, x1 + 2); y2 = max(y2, y1 + 2)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return np.zeros((WIN_H, WIN_W), np.uint8)
    p = cv2.copyMakeBorder(gray[cy1:cy2, cx1:cx2], cy1 - y1, y2 - cy2, cx1 - x1, x2 - cx2, cv2.BORDER_REPLICATE)
    return hp.to_window(p)


class HeadDetector:
    def __init__(self, model_path):
        self.model = load_head_model(model_path)
        self.params = self.model.params
        self._w3 = self.model.w.reshape(hp.BLOCKS_Y, hp.BLOCKS_X, -1)

    # ---- quét ----
    def _score_map(self, B):
        oy, ox = B.shape[0] - hp.BLOCKS_Y + 1, B.shape[1] - hp.BLOCKS_X + 1
        if oy <= 0 or ox <= 0:
            return None
        S = np.full((oy, ox), self.model.b, dtype=np.float64)
        for i in range(hp.BLOCKS_Y):
            for j in range(hp.BLOCKS_X):
                S += B[i:i + oy, j:j + ox] @ self._w3[i, j]
        return S

    @staticmethod
    def _layers(W, H, min_win, step):
        """[(win_px, f, nw, nh)] các tầng pyramid có thể chứa ít nhất một cửa sổ."""
        max_win = min(W, H * WIN_W / WIN_H)
        layers, win_px = [], float(min_win)
        while win_px <= max_win:
            f = WIN_W / win_px
            nw, nh = int(round(W * f)), int(round(H * f))
            if nw >= WIN_W and nh >= WIN_H:
                layers.append((win_px, f, nw, nh))
            win_px *= step
        return layers

    def _refine(self, gray, boxes):
        t = self.model.refine_predict(np.array([hp.hog_vector(crop_window(gray, b)) for b in boxes]))
        w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 + t[:, 0] * w
        cy = (boxes[:, 1] + boxes[:, 3]) / 2 + t[:, 1] * h
        w2 = w * np.exp(np.clip(t[:, 2], -0.5, 0.5)); h2 = h * np.exp(np.clip(t[:, 3], -0.5, 0.5))
        return np.stack([cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2], 1)

    def detect(self, image, config=None, color_order="RGB", progress=None):
        cfg = config or DetectionConfig()
        cfg.validate()
        P = self.params
        pick = lambda value, key: P[key] if value is None else value
        group_iou, contain_iom = pick(cfg.group_iou, "group_iou"), pick(cfg.contain_iom, "contain_iom")
        use_refine = bool(pick(cfg.use_refine, "use_refine")) and self.model.reg_coef is not None
        box_scale, box_shift_y = pick(cfg.box_scale, "box_scale"), pick(cfg.box_shift_y, "box_shift_y")

        full_gray = hp.to_gray(image, color_order)
        H0, W0 = full_gray.shape
        k = min(1.0, cfg.max_image_side / max(H0, W0))
        gray = full_gray if k == 1 else cv2.resize(full_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        H, W = gray.shape
        stride = cfg.stride_cells

        layers = self._layers(W, H, cfg.min_win_px, cfg.scale_step)
        n_windows = 0
        for _, _, nw, nh in layers:
            oy = nh // PPC_Y - hp.CELLS_PER_BLOCK[1] - hp.BLOCKS_Y + 2
            ox = nw // PPC_X - hp.CELLS_PER_BLOCK[0] - hp.BLOCKS_X + 2
            if oy > 0 and ox > 0:
                n_windows += -(-oy // stride) * -(-ox // stride)
            if n_windows > cfg.max_windows:
                raise ValueError("Quá nhiều cửa sổ. Tăng stride_cells/min_win_px hoặc giảm độ phân giải ảnh.")

        dets = []; scanned = 0
        for done, (win_px, f, nw, nh) in enumerate(layers, 1):
            img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
            S = self._score_map(hp.hog_blocks(img))
            if S is not None:
                Ss = S[::stride, ::stride]
                scanned += Ss.size
                ys, xs = np.nonzero(Ss > cfg.det_thresh)
                for y, x in zip(ys, xs):
                    yy, xx = y * stride, x * stride
                    px, py = xx * PPC_X / f, yy * PPC_Y / f
                    dets.append((px, py, px + WIN_W / f, py + WIN_H / f, float(S[yy, xx])))
            if progress:
                progress(done, len(layers))
        stats = {"windows": scanned, "before_nms": len(dets), "after_nms": 0}
        if not dets:
            return [], stats
        if len(dets) > cfg.max_candidates:
            d = np.asarray(dets); dets = d[np.argsort(-d[:, 4], kind="stable")[:cfg.max_candidates]]

        grouped = group_boxes(np.asarray(dets), cfg.det_thresh, group_iou)
        g = grouped[grouped[:, 5] >= cfg.min_neighbors] if len(grouped) else grouped
        if len(g) == 0:
            return [], stats
        boxes, scores, neigh = g[:, :4], g[:, 4], g[:, 5]
        if use_refine:
            boxes = self._refine(gray, boxes)
        keep = nms(boxes, scores, cfg.nms_iou)
        boxes, scores, neigh = boxes[keep], scores[keep], neigh[keep]
        keep = suppress_contained(boxes, scores, contain_iom)
        boxes, scores, neigh = boxes[keep], scores[keep], neigh[keep]

        w_, h_ = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]          # mở rộng khung theo quy ước dataset
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2 + box_shift_y * h_
        w_, h_ = w_ * box_scale[0], h_ * box_scale[1]
        boxes = np.stack([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2], 1) / k   # về toạ độ ảnh gốc
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W0); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H0)
        boxes = np.round(boxes).astype(int)

        results = [{"box": tuple(int(v) for v in b), "head_score": float(s), "neighbors": int(n)}
                   for b, s, n in zip(boxes, scores, neigh) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        stats["after_nms"] = len(results)
        return results, stats
