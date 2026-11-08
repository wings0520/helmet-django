import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle
import cv2
import numpy as np
import pytest
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from src import head_preprocessing as hp
from src.head_preprocessing import extract_features, to_gray, hog_blocks
from src.head_detector import nms, DetectionConfig, HeadDetector, group_boxes
from src.head_model import load_head_model
from src.helmet_preprocessing import preprocess as helmet_pre, extract_features as helmet_feats
from src.pipeline import HelmetPipeline, expanded_box

ROOT = Path(__file__).resolve().parents[1]
HEAD_PKL = ROOT / "models" / "head_hog_svm.pkl"
HELMET = ROOT / "models" / "helmet_svm.joblib"


def test_hog_matches_notebook():
    img = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    expected = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    np.testing.assert_array_equal(extract_features(img), expected)
    assert expected.size == hp.FEATURE_COUNT == 1764


def test_color_and_edge_cases():
    rgb = np.zeros((48, 48, 3), np.uint8); rgb[:, :, 0] = 255
    assert np.array_equal(to_gray(rgb), to_gray(rgb[:, :, ::-1], "BGR"))
    assert int(to_gray(rgb)[0, 0]) == 76
    for value in (0, 255):
        assert np.isfinite(extract_features(np.full((8, 8), value, np.uint8))).all()
    assert expanded_box((0, 0, 10, 10), 12, 12, .3) == (0, 0, 12, 12)
    with pytest.raises(ValueError): DetectionConfig(scale_step=1).validate()
    with pytest.raises(ValueError): DetectionConfig(min_win_px=8).validate()


def test_nms_and_grouping():
    assert nms([], []) == []
    boxes = [(0, 0, 20, 20), (1, 1, 21, 21), (40, 40, 60, 60)]
    assert nms(boxes, [2, 1, 3]) == [2, 0]
    g = group_boxes([(0, 0, 20, 20, 2.0), (1, 1, 21, 21, 1.0), (40, 40, 60, 60, 3.0)], 0.0, 0.4)
    assert g.shape == (2, 6) and sorted(g[:, 5]) == [1, 2]


def test_config_defaults_match_pkl():
    M = pickle.load(open(HEAD_PKL, "rb")); cfg = DetectionConfig()
    assert (cfg.min_win_px, cfg.scale_step, cfg.stride_cells, cfg.det_thresh, cfg.min_neighbors, cfg.nms_iou) == \
           (M["min_win_px"], M["scale_step"], M["stride_cells"], M["det_thresh"], M["min_neighbors"], M["nms_iou"])
    model = load_head_model(HEAD_PKL)
    assert model.w.size == 1764 and model.reg_coef.shape == (4, 1764) and model.params["use_refine"]


def test_scan_matches_direct_svm_score():
    model = load_head_model(HEAD_PKL)
    win = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
    direct = float(model.w @ extract_features(win) + model.b)
    det = HeadDetector(HEAD_PKL)
    S = det._score_map(hog_blocks(win))
    assert S.shape == (1, 1) and abs(S[0, 0] - direct) < 1e-3


def test_load_rejects_bad_pickle(tmp_path):
    M = pickle.load(open(HEAD_PKL, "rb"))
    bad = tmp_path / "bad.pkl"
    pickle.dump({**M, "win": (48, 48)}, open(bad, "wb")); 
    with pytest.raises(ValueError, match="win"): load_head_model(bad)
    pickle.dump({**M, "w": M["w"][:900]}, open(bad, "wb"))
    with pytest.raises(ValueError, match="1764"): load_head_model(bad)
    pickle.dump({k: v for k, v in M.items() if k != "b"}, open(bad, "wb"))
    with pytest.raises(ValueError, match="Thiếu"): load_head_model(bad)


def test_full_run_and_limits():
    engine = HelmetPipeline(HEAD_PKL, HELMET)
    image = np.random.default_rng(1).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    result = engine.run(image, DetectionConfig(det_thresh=-100))
    assert result["stats"]["windows"] > 0 and len(result["detections"]) >= 1
    assert result["annotated_rgb"].shape == image.shape
    assert set(d["label"] for d in result["detections"]) <= {"helmet", "No_helmet"}
    assert engine.run(image, DetectionConfig(det_thresh=100))["detections"] == []
    assert engine.run(image, DetectionConfig(min_win_px=512))["stats"]["windows"] == 0
    with pytest.raises(ValueError, match="Quá nhiều"):
        engine.run(image, DetectionConfig(det_thresh=-100, max_windows=2))


def test_large_image_boxes_in_original_coordinates():
    det = HeadDetector(HEAD_PKL)
    image = np.full((2000, 3000, 3), 128, np.uint8)
    boxes, stats = det.detect(image, DetectionConfig(det_thresh=-100, min_win_px=300, max_image_side=800))
    assert boxes and all(0 <= b["box"][0] < b["box"][2] <= 3000 and 0 <= b["box"][1] < b["box"][3] <= 2000 for b in boxes)


def test_crop_uses_original_rgb():
    engine = HelmetPipeline(HEAD_PKL, HELMET)
    image = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    original = image.copy()
    engine.detector.detect = lambda *a, **k: ([{"box": (10, 12, 40, 45), "head_score": 2.0}], {})
    result = engine.run(image)
    np.testing.assert_array_equal(result["crops_rgb"][0], original[12:45, 10:40])
    np.testing.assert_array_equal(image, original)


def test_helmet_preprocessing_unchanged():
    for mean in (30, 120, 220):
        img = np.clip(np.random.default_rng(mean).normal(mean, 15, (48, 48)), 0, 255).astype(np.uint8)
        ref = img.copy(); m = ref[8:44, 8:40].mean()
        if not 90 <= m <= 160:
            m = np.clip(m / 255.0, 1e-3, 0.999)
            g = np.clip(np.log(128 / 255.0) / np.log(m), 0.6, 2.0)
            ref = (255 * (ref / 255.0) ** g).clip(0, 255).astype(np.uint8)
        ref = cv2.GaussianBlur(ref, (3, 3), sigmaX=0, sigmaY=0)
        ref = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(ref)
        assert np.array_equal(helmet_pre(img), ref)
        assert helmet_feats(img).size == 900
