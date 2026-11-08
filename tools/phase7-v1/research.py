"""Bounded public text research only; never fetch photographs/assets/archives."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/phase7-data-strategy-v1/sources'
SOURCES = {
    'lvis-categories.py': 'https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py',
    'lvis-image-counts.py': 'https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_category_image_count.py',
    'lvis-readme.md': 'https://raw.githubusercontent.com/lvis-dataset/lvis-api/master/README.md',
    'lvis-home.html': 'https://www.lvisdataset.org/',
    'polyhaven-furniture.json': 'https://api.polyhaven.com/assets?type=models&category=furniture',
    'polyhaven-license.html': 'https://polyhaven.com/license',
    'polyhaven-api.md': 'https://raw.githubusercontent.com/Poly-Haven/Public-API/master/README.md',
    'blenderproc-setup.py': 'https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/setup.py',
    'blenderproc-room.py': 'https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/advanced/random_room_constructor/main.py',
    'objects365-download.html': 'https://www.objects365.org/download.html',
    'objects365-classes.py': 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/datasets/objects365.py',
    'openimages-description.html': 'https://storage.googleapis.com/openimages/web/factsfigures_v7.html',
}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def run():
    ledger = []
    total = 0
    opener = urllib.request.build_opener(NoRedirect)
    for name, url in SOURCES.items():
        path = OUT / name
        if path.exists():
            raise FileExistsError(path)
        record = dict(url=url, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
        try:
            with opener.open(urllib.request.Request(url, headers={'User-Agent': 'FurnitureAI-Phase7-public-metadata-research/1.0'}), timeout=30) as r:
                assert r.status == 200
                data = r.read(1_000_001)
                total += len(data)
                if len(data) > 1_000_000 or total > 5_000_000:
                    raise ValueError('Research text budget exceeded')
                ctype = r.headers.get('Content-Type', '')
                if not any(t in ctype for t in ('text/', 'json', 'javascript', 'octet-stream')):
                    raise ValueError('Unexpected research media type')
                data.decode('utf-8')
                with path.open('xb') as f:
                    f.write(data)
                record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), file=name, content_type=ctype)
        except Exception as e:
            record.update(error=str(e))
        ledger.append(record)
    with (OUT / 'research-ledger.json').open('x') as f:
        json.dump(dict(records=ledger, response_body_bytes=total, archive_or_photo_downloads=0), f, indent=2)
    print(json.dumps(ledger, indent=2))

if __name__ == '__main__':
    run()
