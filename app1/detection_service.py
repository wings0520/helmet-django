"""Django adapter; inference modules and model bytes remain unchanged."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
from PIL import Image, ImageOps
from django.conf import settings

def decode_rgb(data):
    with Image.open(BytesIO(data)) as im:
        if im.width * im.height > 25_000_000:
            raise ValueError('Ảnh vượt quá 25 megapixel.')
        im = ImageOps.exif_transpose(im)
        if im.mode == 'RGBA' or 'transparency' in im.info:
            rgba = im.convert('RGBA')
            im = Image.alpha_composite(Image.new('RGBA', rgba.size, (255,255,255,255)), rgba)
        return np.asarray(im.convert('RGB'))

def png_bytes(rgb):
    out=BytesIO(); Image.fromarray(rgb).save(out, format='PNG'); return out.getvalue()

def signature(data, config):
    h=hashlib.sha256(data)
    h.update(json.dumps(config, sort_keys=True).encode())
    for path in (settings.HEAD_MODEL, settings.HELMET_MODEL):
        h.update(Path(path).read_bytes())
    return h.hexdigest()

def save_artifacts(token, result):
    folder=Path(settings.MEDIA_ROOT)/'results'/str(token)
    folder.mkdir(parents=True,exist_ok=True)
    result_png=png_bytes(result['annotated_rgb'])
    (folder/'result.png').write_bytes(result_png)
    metadata={k: result[k] for k in ('detections','stats')}
    with zipfile.ZipFile(folder/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.png', result_png)
        z.writestr('detections.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        for crop,row in zip(result['crops_rgb'],result['detections']):
            blob=png_bytes(crop);name=f"crop_{row['id']:03d}.png"
            (folder/name).write_bytes(blob)
            z.writestr(f"crops/{row['id']:03d}_{row['label']}.png",blob)
    return metadata
