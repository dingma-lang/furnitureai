"""Reuse sealed, tested same-photo JSON-LD parser; no network or historical writes."""
import importlib.util
import sys
from audit_core import ROOT,STATUSES

sys.path.insert(0,str(ROOT/'tools/phase6c-v1'))
spec=importlib.util.spec_from_file_location('phase7_rights_source_parser',ROOT/'tools/phase6c-v1/evidence.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
allowed_page=old.allowed_page
account=old.account
canonical_id=old.canonical_id

def classify(text,row,url):
 adapted=dict(source_identity=row['source_photo_id'],metadata_license_url=row['metadata_license_url'])
 result=old.classify(text,adapted,url)
 result['licensing_status']=dict(zip(old.STATUSES,STATUSES))[result['licensing_status']]
 if result.get('attribution'):result['attribution']=result['attribution'].replace('COCO Consortium / CC BY4.0','LVIS / Gupta, Dollar, Girshick / CC BY4.0')
 return result
