from django import forms
from .models import Image
from src.head_detector import DetectionConfig as _D

class ImageForm(forms.ModelForm):
    keywords = forms.CharField(required=False, max_length=255, label='Từ khóa')
    class Meta:
        model = Image
        fields = ['photo', 'caption', 'keywords']
        labels = {'photo': 'Chọn ảnh', 'caption': 'Tên / mô tả ảnh'}
        widgets = {'photo': forms.FileInput(attrs={'accept': '.jpg,.jpeg,.png,.bmp,.webp'})}
    def clean_photo(self):
        photo = self.cleaned_data['photo']
        from PIL import Image as PILImage
        if photo.size > 50 * 1024 * 1024:
            raise forms.ValidationError('Ảnh tối đa 50 MB.')
        try:
            with PILImage.open(photo) as im:
                if im.format not in ('JPEG','PNG','BMP','WEBP') or im.width * im.height > 25_000_000:
                    raise ValueError()
            photo.seek(0)
        except Exception:
            raise forms.ValidationError('Cần ảnh JPG, PNG, BMP hoặc WebP dưới 25 megapixel.')
        return photo

class DetectionForm(forms.Form):
    # Mặc định lấy từ DetectionConfig (= tham số trong models/head_hog_svm.pkl).
    min_win_px = forms.IntegerField(label='Đầu nhỏ nhất (pixel)', initial=_D.min_win_px, min_value=32, max_value=2048)
    scale_step = forms.FloatField(label='Hệ số pyramid', initial=_D.scale_step, min_value=1.02, max_value=2)
    stride_cells = forms.IntegerField(label='Bước trượt (cell)', initial=_D.stride_cells, min_value=1, max_value=8)
    det_thresh = forms.FloatField(label='Ngưỡng điểm đầu', initial=_D.det_thresh)
    min_neighbors = forms.IntegerField(label='Số cửa sổ tối thiểu / cụm', initial=_D.min_neighbors, min_value=1, max_value=50)
    nms_iou = forms.FloatField(label='IoU NMS', initial=_D.nms_iou, min_value=0, max_value=1)
    padding = forms.FloatField(label='Mở rộng crop mỗi phía', initial=0, min_value=0, max_value=.3)
    def clean(self):
        data = super().clean()
        import math
        if any(not math.isfinite(v) for v in data.values()):
            raise forms.ValidationError('Thông số phải là số hữu hạn.')
        return data
