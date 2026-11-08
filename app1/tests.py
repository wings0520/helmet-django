import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image as PILImage
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings, Client
from django.urls import reverse
from django.conf import settings
from .models import Image, DetectionRun
from .forms import DetectionForm
from .detection_service import decode_rgb
from contextlib import closing

class WorkflowTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.override=override_settings(MEDIA_ROOT=self.temp.name,FEEDBACK_DB=Path(self.temp.name)/'feedback.db')
        self.override.enable();self.addCleanup(self.override.disable)
        self.user=User.objects.create_user('tester',password='strong-test-123')
        self.client.force_login(self.user)
    def upload(self,format='PNG'):
        data=io.BytesIO();PILImage.new('RGB',(256,128),(75,105,135)).save(data,format=format)
        ext={'JPEG':'jpg','PNG':'png','BMP':'bmp','WEBP':'webp'}[format]
        response=self.client.post(reverse('home'),{'photo':SimpleUploadedFile('test.'+ext,data.getvalue()),'caption':'Công trường','keywords':'helmet'})
        self.assertEqual(response.status_code,302)
        return Image.objects.latest('pk')
    def config(self,**kwargs):
        result={k:v.initial for k,v in DetectionForm().fields.items()};result.update(kwargs);return result
    def run_detection(self,image,**kwargs):
        response=self.client.post(reverse('detect',args=[image.pk]),self.config(**kwargs))
        self.assertEqual(response.status_code,200)
        events=[json.loads(line) for line in b''.join(response.streaming_content).splitlines()]
        self.assertEqual(events[-1]['status'],'done',events)
        return DetectionRun.objects.latest('pk'),events
    def test_upload_formats_search_and_ownership(self):
        for fmt in ('PNG','JPEG','BMP','WEBP'): self.upload(fmt)
        self.assertContains(self.client.get('/?search=Công'),'Công trường')
        image=Image.objects.first()
        self.assertEqual(self.client.get(reverse('original',args=[image.pk])).status_code,200)
        other=User.objects.create_user('other',password='strong-test-123');self.client.force_login(other)
        for name in ('detail','original'):
            self.assertEqual(self.client.get(reverse(name,args=[image.pk])).status_code,404)
        self.assertEqual(self.client.post(reverse('detect',args=[image.pk]),self.config()).status_code,404)
        self.client.logout();self.assertEqual(self.client.get('/').status_code,302)
    def test_actual_models_downloads_feedback_and_stale_config(self):
        image=self.upload();run,events=self.run_detection(image,det_thresh=-100)
        self.assertGreater(run.result['stats']['windows'],0)
        count=len(run.result['detections']);self.assertGreaterEqual(count,1)
        self.assertTrue(any(x.get('status')=='progress' for x in events))
        page=self.client.get(reverse('detail',args=[image.pk])+f'?run={run.token}')
        self.assertContains(page,'Kết quả phân tích')
        self.assertContains(page,'crop_001.png')
        for name in ('result.png','result.zip','crop_001.png'):
            response=self.client.get(reverse('artifact',args=[run.token,name]))
            self.assertEqual(response.status_code,200);data=b''.join(response.streaming_content)
            if name.endswith('.zip'):
                z=zipfile.ZipFile(io.BytesIO(data));self.assertIn('detections.json',z.namelist())
                self.assertEqual(len(json.loads(z.read('detections.json'))['detections']),count)
            else: self.assertEqual(PILImage.open(io.BytesIO(data)).format,'PNG')
        token=self.client.session[f'feedback_{image.pk}']
        payload={**run.config,'token':token,'run':str(run.token),'user_name':'Thông','rating':5,'comment':'Kiểm tra phản hồi'}
        with patch('app1.views.HelmetPipeline',side_effect=AssertionError('Inference must not rerun')):
            for _ in range(2): self.assertEqual(self.client.post(reverse('feedback',args=[image.pk]),payload).status_code,302)
            self.assertEqual(self.client.get(reverse('detail',args=[image.pk])+f'?run={run.token}').status_code,200)
        with closing(sqlite3.connect(Path(self.temp.name)/'feedback.db')) as db, db:
            rows=db.execute('select comment,result_json from feedback').fetchall()
        self.assertEqual(len(rows),1);self.assertIsNotNone(rows[0][1])
        self.client.post(reverse('feedback',args=[image.pk]),{'new':'1'})
        payload.update(token=self.client.session[f'feedback_{image.pk}'],det_thresh=50)
        self.client.post(reverse('feedback',args=[image.pk]),payload)
        with closing(sqlite3.connect(Path(self.temp.name)/'feedback.db')) as db, db:
            self.assertIsNone(db.execute('select result_json from feedback order by id desc').fetchone()[0])
        self.assertEqual(DetectionRun.objects.count(),1)
    def test_validation_empty_and_failure_preserves_comment(self):
        image=self.upload()
        invalid=self.client.post(reverse('detect',args=[image.pk]),self.config(min_win_px=8))
        self.assertEqual(invalid.status_code,400)
        run,_=self.run_detection(image,min_win_px=512)
        self.assertEqual(run.result['stats']['windows'],0)
        page=self.client.get(reverse('detail',args=[image.pk])+f'?run={run.token}')
        self.assertContains(page,'không có nghĩa ảnh chắc chắn không có người')
        token=self.client.session[f'feedback_{image.pk}']
        with patch('app1.views.save_feedback',side_effect=OSError('Disk full')):
            response=self.client.post(reverse('feedback',args=[image.pk]),{'token':token,'comment':'Nội dung giữ lại','rating':4})
        self.assertEqual(response.status_code,400);self.assertContains(response,'Nội dung giữ lại',status_code=400)
        self.assertContains(response,'Chưa lưu được',status_code=400)
    def test_bad_upload_and_decode(self):
        response=self.client.post('/',{'photo':SimpleUploadedFile('bad.jpg',b'bad'),'keywords':''})
        self.assertEqual(response.status_code,200);self.assertEqual(Image.objects.count(),0)
        data=io.BytesIO();PILImage.new('RGBA',(20,10),(255,0,0,0)).save(data,format='PNG')
        rgb=decode_rgb(data.getvalue());self.assertEqual(rgb.shape,(10,20,3));self.assertTrue((rgb==255).all())
    def test_auth_and_csrf(self):
        self.client.logout()
        self.assertEqual(self.client.post('/signup/',{'username':'newuser','password1':'Valid-password-2026','password2':'Valid-password-2026'}).status_code,302)
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/logout/').status_code,405)
        self.client.post('/logout/')
        self.assertEqual(self.client.post('/login/',{'username':'newuser','password':'Valid-password-2026'}).status_code,302)
        strict=Client(enforce_csrf_checks=True);strict.force_login(self.user)
        self.assertEqual(strict.post('/logout/').status_code,403)

    def test_form_defaults_match_pkl(self):
        import pickle
        M=pickle.load(open(settings.HEAD_MODEL,'rb'));f=DetectionForm().fields
        for form_key,pkl_key in (('min_win_px','min_win_px'),('scale_step','scale_step'),('stride_cells','stride_cells'),
                                 ('det_thresh','det_thresh'),('min_neighbors','min_neighbors'),('nms_iou','nms_iou')):
            self.assertEqual(f[form_key].initial,M[pkl_key],form_key)
