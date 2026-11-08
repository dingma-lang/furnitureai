"""Fetch the first original rights page, not any photograph."""
from network import *
if __name__=='__main__':
    m=load_manifest();r=m['images'][0];budget=Budget()
    try:
        rec=budget.fetch(r['source_landing_url'],'licensing/'+r['image_id']+'.response',1048576,reserve_after=42932701)
        text=budget.decoded(rec)
        print(json.dumps(dict(status=rec['status'],body_bytes=rec['actual_body_bytes'],decoded_bytes=len(text),encoding=rec['headers'].get('Content-Encoding'),saved_file=rec['file']),indent=2))
        import re
        for match in list(re.finditer(r'(?i)(license|licence|5053563451)',text))[:35]:print(text[max(0,match.start()-100):match.start()+240])
    finally:budget.close()
