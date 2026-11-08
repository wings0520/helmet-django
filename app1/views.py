import hashlib
import json
import queue
import threading
import uuid
from pathlib import Path
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import FileResponse, HttpResponse, Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST
from .models import Image, DetectionRun
from .forms import ImageForm, DetectionForm
from .detection_service import decode_rgb, signature, save_artifacts
from src.pipeline import HelmetPipeline
from src.head_detector import DetectionConfig
from src.feedback_store import save_feedback

_worker_slot = threading.BoundedSemaphore(1)

@login_required
def home(request):
    form = ImageForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        image = form.save(commit=False); image.user = request.user; image.save()
        messages.success(request,'Đã tải ảnh lên. Bạn có thể bắt đầu phân tích.')
        return redirect('detail', pk=image.pk)
    query = request.GET.get('search','').strip()
    images = Image.objects.filter(user=request.user).order_by('-date')
    if query: images = images.filter(Q(caption__icontains=query)|Q(keywords__icontains=query))
    return render(request,'app1/home.html',{'form':form,'img':images,'search_query':query})

def login_view(request):
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        login(request,form.get_user()); return redirect('home')
    return render(request,'app1/auth.html',{'form':form,'signup':False})

def signup(request):
    form = UserCreationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user=form.save(); login(request,user); return redirect('home')
    return render(request,'app1/auth.html',{'form':form,'signup':True})

@require_POST
def logout_view(request):
    logout(request); return redirect('login')

def owned_image(request,pk):
    return get_object_or_404(Image,pk=pk,user=request.user)

def owned_run(request,token):
    return get_object_or_404(DetectionRun,token=token,image__user=request.user)

def feedback_key(image): return f'feedback_{image.pk}'

def detail_context(request,image,run=None,feedback=None):
    if run:
        try:
            if signature(Path(image.photo.path).read_bytes(),run.config) != run.signature: run=None
        except (OSError,ValueError): run=None
    key=feedback_key(image)
    if key not in request.session: request.session[key]=uuid.uuid4().hex
    data=run.config if run else None
    return {'image':image,'run':run,'config_form':DetectionForm(initial=data),
            'feedback_token':request.session[key], 'feedback':feedback or {},
            'helmet_count':sum(d['label']=='helmet' for d in run.result['detections']) if run else 0,
            'nohelmet_count':sum(d['label']=='No_helmet' for d in run.result['detections']) if run else 0}

@login_required
def detail(request,pk):
    image=owned_image(request,pk)
    run=None
    if request.GET.get('run'):
        try: token=uuid.UUID(request.GET['run'])
        except ValueError: raise Http404('Mã kết quả không hợp lệ.')
        run=get_object_or_404(DetectionRun,token=token,image=image)
    return render(request,'app1/detail.html',detail_context(request,image,run))

@login_required
def original(request,pk):
    image=owned_image(request,pk)
    return FileResponse(image.photo.open('rb'), as_attachment=request.GET.get('download')=='1',filename=Path(image.photo.name).name)

@login_required
@require_POST
def detect(request,pk):
    image=owned_image(request,pk); form=DetectionForm(request.POST)
    if not form.is_valid(): return JsonResponse({'error':str(form.errors.as_text())},status=400)
    config=form.cleaned_data.copy();cfg_data=config.copy();padding=cfg_data.pop('padding')
    cfg=DetectionConfig(**cfg_data)
    try:
        cfg.validate();data=Path(image.photo.path).read_bytes();rgb=decode_rgb(data);sig=signature(data,config)
    except Exception as exc: return JsonResponse({'error':str(exc)},status=400)
    if not _worker_slot.acquire(blocking=False):
        return JsonResponse({'error':'Một ảnh đang được xử lý. Vui lòng thử lại sau.'},status=429)
    events=queue.Queue()
    def worker():
        try:
            engine=HelmetPipeline(settings.HEAD_MODEL,settings.HELMET_MODEL)
            result=engine.run(rgb,cfg,padding=padding,progress=lambda done,total:events.put(('progress',{'done':done,'total':total})))
            events.put(('result',result))
        except Exception as exc: events.put(('error',str(exc)))
        finally: _worker_slot.release()
    threading.Thread(target=worker,daemon=True).start()
    def stream():
        yield json.dumps({'status':'loading'})+'\n'
        while True:
            kind,value=events.get()
            if kind=='progress': yield json.dumps({'status':'progress',**value})+'\n'
            elif kind=='error':
                yield json.dumps({'error':value})+'\n';break
            else:
                try:
                    token=uuid.uuid4();metadata=save_artifacts(token,value)
                    DetectionRun.objects.create(image=image,token=token,signature=sig,config=config,result=metadata)
                    yield json.dumps({'status':'done','url':reverse('detail',args=[pk])+f'?run={token}'})+'\n'
                except Exception as exc: yield json.dumps({'error':str(exc)})+'\n'
                break
    response=StreamingHttpResponse(stream(),content_type='application/x-ndjson')
    response['Cache-Control']='no-store';response['X-Accel-Buffering']='no'
    return response

@login_required
def artifact(request,token,name):
    run=owned_run(request,token)
    allowed={'result.png','result.zip'}|{f"crop_{d['id']:03d}.png" for d in run.result['detections']}
    if name not in allowed: return HttpResponse(status=404)
    path=Path(settings.MEDIA_ROOT)/'results'/str(run.token)/name
    if not path.is_file(): return HttpResponse('File kết quả không còn tồn tại. Hãy phân tích lại.',status=404)
    return FileResponse(path.open('rb'),as_attachment=name.endswith('.zip') or request.GET.get('download')=='1',filename=name)

@login_required
@require_POST
def feedback(request,pk):
    image=owned_image(request,pk);key=feedback_key(image)
    if request.POST.get('new'):
        request.session[key]=uuid.uuid4().hex
        url=reverse('detail',args=[pk])
        if request.POST.get('run'):
            prior=get_object_or_404(DetectionRun,token=request.POST['run'],image=image)
            url+=f'?run={prior.token}'
        return redirect(url)
    run=None;ctx=None
    try:
        if request.POST.get('token') != request.session.get(key): raise ValueError('Mã feedback hết hạn. Hãy tải lại trang.')
        if request.POST.get('run'):
            candidate=get_object_or_404(DetectionRun,token=request.POST['run'],image=image)
            form=DetectionForm(request.POST)
            if form.is_valid() and signature(Path(image.photo.path).read_bytes(),form.cleaned_data)==candidate.signature:
                run=candidate;ctx={**run.result,'config':run.config}
        added=save_feedback(settings.FEEDBACK_DB,request.POST['token'],request.POST.get('user_name',''),
            int(request.POST.get('rating',3)),request.POST.get('comment',''),image_name=Path(image.photo.name).name,
            image_sha256=hashlib.sha256(Path(image.photo.path).read_bytes()).hexdigest(),result=ctx)
        messages.success(request,'Đã lưu feedback. Cảm ơn bạn!' if added else 'Feedback này đã được lưu. Chọn Viết feedback mới để gửi thêm.')
        url=reverse('detail',args=[pk])+(f'?run={run.token}' if run else '')
        return redirect(url)
    except Exception as exc:
        messages.error(request,f'Chưa lưu được feedback: {exc}')
        return render(request,'app1/detail.html',detail_context(request,image,run,request.POST),status=400)
