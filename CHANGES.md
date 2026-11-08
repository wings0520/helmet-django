# File thay đổi và mục đích

| File | Thay đổi |
|---|---|
| app1/views.py, urls.py | Giữ quản lý ảnh/đăng nhập/tìm kiếm; thêm phân tích có tiến trình, kết quả, download và feedback; kiểm tra chủ sở hữu |
| app1/forms.py | Validate ảnh, tham số; mô tả/từ khóa tiếng Việt |
| app1/models.py, migrations/0002_detectionrun.py | Thêm bản ghi kết quả và cấu hình, giữ nguyên Image |
| app1/detection_service.py | Adapter RGB/EXIF, chữ ký và xuất PNG/ZIP/crop |
| templates/app1/base.html, home.html, auth.html, detail.html | Giao diện navy sáng, thư viện card, ảnh gốc/kết quả, feedback |
| static/app1/studio.css, studio.js | Responsive, progress từ server, ẩn kết quả khi cấu hình đổi |
| ImageIQ/settings.py, urls.py | Tiếng Việt, múi giờ VN, model/feedback paths, file ảnh qua view kiểm tra quyền |
| requirements.txt, ImageIQ/requirements.txt | Chuẩn hóa UTF-8, thêm dependency AI, giữ sklearn 1.6.1 |
| requirements-dev.txt, app1/tests.py, tests/ | Test tích hợp Django và module AI/SQLite |
| README.md, README_UPSTREAM.md, VALIDATION.md, CHANGES.md | Hướng dẫn, nguồn gốc, thay đổi và giới hạn kiểm tra |
| kaggle_inference.ipynb | Cập nhật tên thư mục project thành Helmet_Django |

`src/` sao chép nguyên từ project trước; hai model mới được đưa vào `models/` nguyên byte.
Không kèm env, .git, cache, database hoặc ảnh người dùng cũ. Giữ LICENSE của website gốc.
Không có app.py / Streamlit: người dùng đã xác nhận giữ Django ở yêu cầu cuối.

## Cập nhật bộ phát hiện đầu theo head_hog_svm.pkl

| File | Thay đổi |
|---|---|
| models/head_hog_svm.pkl | Model đầu mới (HOG 64×64 + LinearSVM + Ridge hiệu chỉnh khung) |
| src/head_model.py (mới) | Nạp và kiểm tra .pkl (tham số HOG, 1764 trọng số, hồi quy khung) |
| src/head_preprocessing.py | HOG 64×64 thô, bỏ gamma/Gaussian/CLAHE |
| src/head_detector.py | Pyramid + score map vector hoá, gộp cụm, min_neighbors, Ridge, NMS, loại khung lồng |
| src/helmet_preprocessing.py | Tự chứa bản 48×48 cũ để bộ phân loại mũ không đổi |
| app1/forms.py, views.py | Tham số mới: min_win_px, scale_step, stride_cells, det_thresh, min_neighbors, nms_iou |
| ImageIQ/settings.py, templates/app1/detail.html | Đường dẫn và mô tả model mới |
| tests/, app1/tests.py | Viết lại; kiểm tra khớp notebook và khớp tham số .pkl |
