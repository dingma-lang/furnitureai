"""Replay saved, photograph-specific JSON-LD evidence. No provider requests."""
import gzip
import html
import io
import json
from html.parser import HTMLParser
from urllib.parse import urlparse
from common import ROOT, read, safe, sha

class LD(HTMLParser):
    def __init__(self):
        super().__init__(); self.active = False; self.parts = []; self.docs = []; self.title = None
    def handle_starttag(self, tag, attrs):
        if tag == 'meta' and dict(attrs).get('property') == 'og:title':
            self.title = dict(attrs).get('content')
        if tag == 'script' and dict(attrs).get('type') == 'application/ld+json':
            self.active = True; self.parts = []
    def handle_data(self, data):
        if self.active: self.parts.append(data)
    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.docs.append(json.loads(''.join(self.parts))); self.active = False

def photo(url):
    u = urlparse(url)
    parts = u.path.strip('/').split('/')
    if u.hostname not in ('flickr.com', 'www.flickr.com') or len(parts) != 3 or parts[0] != 'photos' or not parts[2].isdigit():
        raise ValueError('Not an exact source-photo URL')
    return 'flickr-photo:' + parts[2], parts[1].lower()

def replay(body, expected_photo, licence, creator_url):
    if body[:2] == b'\x1f\x8b':
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as f: body = f.read(4_194_305)
    if len(body) > 4_194_304: raise ValueError('Evidence decompression bound')
    p = LD(); p.feed(body.decode('utf-8')); objects = []
    def walk(x):
        if isinstance(x, dict):
            if x.get('@type') == 'ImageObject':
                try: identity = photo(x.get('acquireLicensePage', ''))[0]
                except ValueError: identity = None
                if identity == expected_photo: objects.append(x)
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    for d in p.docs: walk(d)
    objects = list({json.dumps(o, sort_keys=True): o for o in objects}.values())
    if len(objects) != 1: raise ValueError('Missing or conflicting same-photo evidence')
    obj = objects[0]; lic = obj.get('license', '').replace('http://', 'https://').rstrip('/') + '/'
    if lic != licence or lic != 'https://creativecommons.org/licenses/by/2.0/':
        raise ValueError('Saved photo licence contradiction')
    author = obj.get('author', {})
    title = obj.get('name') or obj.get('headline') or p.title
    if not isinstance(author, dict) or not author.get('name') or not title:
        raise ValueError('Missing source attribution')
    account = urlparse(author.get('url', '')).path.strip('/').split('/')
    expected = urlparse(creator_url).path.strip('/').split('/')
    if len(account) != 2 or account[0] != 'photos' or [x.lower() for x in account] != [x.lower() for x in expected] or account[1].lower() != photo(obj['acquireLicensePage'])[1]:
        raise ValueError('Creator account not bound to exact photo')
    return dict(creator=html.unescape(author['name']), creator_url=author['url'],
                title=title, source_url=obj['acquireLicensePage'],
                license_url=lic, creator_group='flickr-account:' + account[1].lower())

def proof(source, original, base):
    iid = source['image_id']; kind = source['source']
    if kind == 'legacy_openimages':
        record_path = base / original['licence_evidence']; r = read(record_path)
        evidence = base / r['response_file']; digest = r['response_sha256']
        creator_url = r['author_url']; timestamp = r['checked_utc']; licence = r['license_url']
    elif kind == 'phase6a_openimages':
        record_path = base / 'licensing-evidence' / (iid + '.json'); r = read(record_path)
        evidence = base / 'licensing-evidence' / (iid + '.response'); digest = r['response_sha256']
        creator_url = r['creator_url']; timestamp = r['checked_utc']; licence = r['license_url']
    elif kind == 'phase6d_coco':
        record_path = safe(original['rights_record']); r = read(record_path)
        if sha(record_path) != original['rights_record_sha256']: raise ValueError('Rights record changed')
        ev = [x for x in r['evidence_requests'] if x.get('method') == 'GET' and x.get('sha256')]
        if len(ev) != 1: raise ValueError('Ambiguous evidence selection')
        evidence = ROOT / 'data/coco/rights-audits/phase6c-v1' / ev[0]['file']; digest = ev[0]['sha256']
        creator_url = r['creator_url']; timestamp = r['checked_utc']; licence = r['license_url']
    else:
        ledger = read(base / 'license-and-provenance-ledger.json')['records']
        if kind == 'phase7_LVIS':
            r = next(x for x in ledger if x['coco_image_id'] == original['coco_image_id'])
            record_path = safe(r['rights_record'])
            if sha(record_path) != r['rights_record_sha256']: raise ValueError('Rights record changed')
            evidence = safe(r['evidence_file']); digest = r['evidence_sha256']
            creator_url = r['creator_url']; timestamp = r['license_checked_utc']; licence = r['photo_license']
        elif kind == 'phase8c_LVIS':
            r = next(x for x in ledger if x['acquisition']['image_id'] == iid)['rights']
            record_path = safe(r['rights_record'])
            if sha(record_path) != r['rights_record_sha256']: raise ValueError('Rights record changed')
            evidence = safe(r['evidence_file']); digest = r['evidence_sha256']
            creator_url = r['creator_url']; timestamp = r['evidence_checked_utc']; licence = r['license_url']
        else: raise ValueError('Unsupported source')
    if not evidence.resolve().is_relative_to(ROOT) or sha(evidence) != digest:
        raise ValueError('Saved licensing evidence changed')
    result = replay(evidence.read_bytes(), source['source_photo_id'], licence, creator_url)
    result.update(evidence_file=str(evidence.relative_to(ROOT)), evidence_sha256=digest,
                  rights_record=str(record_path.relative_to(ROOT)), rights_record_sha256=sha(record_path),
                  original_verification_utc=timestamp, offline_replay=True,
                  annotation_license='https://creativecommons.org/licenses/by/4.0/',
                  current_ownership_or_other_releases_verified=False)
    return result
