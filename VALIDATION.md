# Kết quả kiểm tra — 03/11/2026

## Đã chạy

- `python manage.py check`: không có vấn đề.
- `python manage.py test`: 5 test tích hợp thành công.
- `python -m pytest tests -q`: 7 test thành công.
- `node --check static/app1/studio.js`: cú pháp JavaScript hợp lệ.
- Hai model trong ZIP khớp nguyên byte với hai file mới người dùng gửi.
- Tất cả file Python trong src/ khớp nguyên byte với project Streamlit trước đó.

## Phạm vi 5 test tích hợp Django

1. Upload PNG/JPEG/BMP/WebP qua multipart, tìm kiếm tiếng Việt; kiểm tra quyền sở hữu ảnh.
2. Dùng **hai model thật** chạy ảnh RGB tổng hợp 256×128, cửa sổ 128, bước trượt 1;
   kiểm tra 2 cửa sổ, 2 crop, nhãn hợp lệ, sự kiện progress và kết quả.
   Ngưỡng -100 chỉ dùng trong test để bắt buộc đi qua classifier, không đổi default ứng dụng.
   Mở trang kết quả bằng test client, kiểm tra các đường dẫn crop;
   tải và giải mã PNG, mở ZIP, đọc JSON và đối chiếu số kết quả.
3. Feedback UTF-8 được ghi vào SQLite, gửi lại không nhân bản;
   đổi cấu hình thì không gắn kết quả cũ vào feedback.
   Mock kiểm tra feedback / xem kết quả không được gọi lại model.
4. Tham số sai, upload hỏng, ảnh RGBA, không có cửa sổ phù hợp;
   lỗi lưu feedback vẫn render nguyên nội dung đã nhập và báo chưa lưu.
5. Đăng ký, đăng nhập, đăng xuất bằng POST; CSRF từ chối POST thiếu token.

## Phạm vi 7 test module

Tiền xử lý khớp code training; RGB/BGR; ảnh sáng/tối; HOG 900;
NMS; crop nguyên RGB; label mapping; model không tương thích;
SQLite persistence, validation và chống trùng.

## Chưa xác nhận được

- Chưa kiểm tra click end-to-end hoặc chụp render desktop/mobile trong trình duyệt:
  trình duyệt cloud không kết nối được máy chủ local của workspace,
  còn cài Chromium để test local bị giới hạn mạng.
- HTML được render bằng Django test client, JS chỉ kiểm tra cú pháp;
  vì vậy chưa xác nhận trực quan CSS và tiến trình fetch trong trình duyệt thực.
- Chưa chạy trực tiếp trên Windows của người dùng.
- Chưa benchmark ảnh lớn, nhiều người dùng hoặc độ chính xác trên tập ảnh thực có nhãn.
- Không khẳng định accuracy/precision/recall tăng do đổi giao diện.

## Kiểm tra thủ công khi chạy trên máy

Đăng ký → tải ảnh → mở Cài đặt nâng cao → phân tích → xem PNG/ZIP và crop.
Đổi ngưỡng để kiểm tra kết quả cũ ẩn đi. Chạy lại và gửi feedback;
kiểm tra kết quả vẫn còn, gửi lại không có hàng mới. Mở kích thước màn hình nhỏ
để kiểm tra cột xếp dọc và ảnh giữ tỷ lệ. Kiểm tra thêm ảnh thật của dataset.
