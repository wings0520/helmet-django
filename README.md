# Helmet Studio — Django

Website quản lý ảnh ImageIQ được tích hợp phát hiện đầu và kiểm tra mũ bảo hiểm.
Theo xác nhận cuối cùng của người dùng, giữ **Django**, không chuyển sang Streamlit.
Chạy bằng `python manage.py runserver`; không dùng `streamlit run app.py`.

## Chạy trên Windows (PowerShell)

Khuyên dùng Python 3.11 hoặc 3.12. Mở terminal tại thư mục chứa `manage.py`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Mở http://127.0.0.1:8002 rồi tạo tài khoản. Nếu PowerShell không cho activate,
không cần đổi execution policy; dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Linux/macOS: `source .venv/bin/activate`, các lệnh Python còn lại giống nhau.
Hai model đã được đóng gói, không cần huấn luyện lại.

## Luồng sử dụng

1. Đăng ký / đăng nhập.
2. Tải ảnh trong thư viện: JPG/JPEG, PNG, BMP hoặc WebP; tối đa 50 MB và 25 megapixel.
3. Điền mô tả, từ khóa để tìm lại ảnh. Mỗi tài khoản xem ảnh của mình.
4. Mở ảnh, tùy chỉnh Cài đặt nâng cao nếu cần, bấm Phát hiện và phân loại.
5. Theo dõi tiến trình quét thực tế. Sau khi hoàn tất, xem ảnh gốc/kết quả,
   số đầu/có mũ/không mũ, bảng tọa độ và các crop.
6. Tải PNG hoặc ZIP gồm result.png, detections.json, crops/.
7. Gửi feedback; tên tùy chọn, đánh giá 1–5, nội dung bắt buộc.
   Viết feedback mới tạo mã gửi mới. Việc gửi lại cùng mã không tạo thêm bản ghi.

Không có camera/snapshot. Không tạo kết quả giả và không hiển thị score như xác suất.
Khi đổi tham số, kết quả đang hiển thị bị ẩn cho đến lần phân tích tiếp theo.
Gửi feedback và tải file không gọi lại inference.

## Cấu trúc

- `manage.py`, `ImageIQ/`: entrypoint và cấu hình Django.
- `app1/models.py`: giữ Image, thêm DetectionRun lưu cấu hình/kết quả từng lần chạy.
- `app1/views.py`, `urls.py`, `forms.py`: upload, đăng nhập, thư viện, tìm kiếm,
  API tiến trình, kết quả, tải file và feedback.
- `app1/detection_service.py`: đọc RGB/EXIF, chữ ký ảnh + cấu hình + model,
  xuất ảnh/crop/JSON/ZIP, không thay thuật toán.
- `templates/app1/base.html`, `home.html`, `auth.html`, `detail.html`: giao diện tiếng Việt.
- `static/app1/studio.css`, `studio.js`: giao diện responsive, cập nhật tiến trình,
  ẩn kết quả cũ khi thay cấu hình. Không phụ thuộc CDN.
- `src/`: head_preprocessing, head_model, head_detector viết lại theo notebook `head-detect-clean.ipynb`
  (HOG 64×64 + LinearSVM); helmet_preprocessing tự chứa bản 48×48 cũ; helmet_classifier, model_utils,
  pipeline, feedback_store giữ nguyên.
- `models/head_hog_svm.pkl`: model đầu (nguyên byte từ file gửi); `helmet_svm.joblib`: model mũ.
  `head_svm.joblib` cũ không còn được dùng.
- `tests/`: kiểm tra tiền xử lý, pipeline, NMS và SQLite.
- `app1/tests.py`: kiểm tra luồng Django và hai model thật.
- `kaggle_inference.ipynb`: notebook inference riêng; sửa PROJECT_DIR nếu đặt ở nơi khác.

## Cấu hình model và tiền xử lý

Đường dẫn model chỉ được đổi trên máy chủ tại `ImageIQ/settings.py`:
`HEAD_MODEL` (.pkl) và `HELMET_MODEL` (.joblib). Chỉ sử dụng file model mà bạn tin cậy.

Phát hiện đầu (tham số lấy từ `head_hog_svm.pkl`): cửa sổ 64×64, HOG 9 hướng, cell 8×8, block 2×2, L2-Hys
(1764 đặc trưng), ảnh xám thô, không gamma/CLAHE. Quét pyramid scale_step=1.1, min_win_px=40, stride 1 cell,
det_thresh=0.8, min_neighbors=1, group_iou=0.4, nms_iou=0.3, contain_iom=0.6, hiệu chỉnh khung Ridge bật,
box_scale=(1,1). Ảnh cạnh dài >1280 px được thu nhỏ khi quét (như notebook), khung quy về ảnh gốc.
Giao diện cho chỉnh: min_win_px, scale_step, stride_cells, det_thresh, min_neighbors, nms_iou, padding.

Phân loại mũ (không đổi): crop RGB gốc → xám → 48×48 → gamma thích nghi → Gaussian (3×3) → CLAHE
(clip=2, tile=4×4) → HOG 900 → pipeline StandardScaler + SVC RBF. Không gamma/CLAHE hai lần trên cùng crop.
Nhãn head: 0=background, 1=head; helmet: 0=No_helmet, 1=helmet.
`scikit-learn==1.6.1` được giữ nguyên.

## Dữ liệu và cập nhật từ website cũ

- `db.sqlite3`: tài khoản, ảnh, metadata kết quả; `media/`: ảnh gốc và file kết quả.
- `feedback.db` cạnh `manage.py`: bảng feedback, tạo khi gửi lần đầu;
  tồn tại thì ghi thêm, không xóa dữ liệu. Không lưu byte ảnh trong feedback.
- Bản ZIP bàn giao không chứa tài khoản, database, media hay feedback cá nhân từ ZIP gốc.
- Nếu cập nhật website đang sử dụng: sao lưu rồi giữ lại **db.sqlite3, media/, feedback.db**
  hiện có; chép code mới và chạy `python manage.py migrate`.
  Migration 0002 chỉ thêm bảng DetectionRun, không xóa bảng Image.
- Không công khai feedback hoặc file ảnh qua đường dẫn media trực tiếp.
  Ảnh và kết quả được phục vụ qua view kiểm tra chủ sở hữu.
- Đây là phiên bản chạy local/demo; một tiến trình xử lý một ảnh tại một thời điểm,
  yêu cầu đồng thời khác nhận thông báo bận. Không có hàng đợi nền bền vững.
  Nếu triển khai nhiều người dùng, cần cấu hình máy chủ, secret, HTTPS,
  ALLOWED_HOSTS và hàng đợi tác vụ; không dùng runserver cho production.

## Kiểm tra

```powershell
python -m pip install -r requirements-dev.txt
python manage.py check
python manage.py test
python -m pytest tests -q
```

Chi tiết phạm vi thực tế trong VALIDATION.md. Giao diện mới không tự làm tăng độ chính xác model.

## Nguồn gốc

Dựa trên ImageIQ / image-upload-download-in-django của Md Jahid Hasan.
Giữ LICENSE GPL-3.0 và README_UPSTREAM.md từ ZIP gốc. Phần bổ sung gồm giao diện
Helmet Studio, tích hợp model, feedback và kiểm thử (03/11/2026).
