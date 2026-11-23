"""Isolated reuse of tested same-photo source parser;no rights from dataset IDs."""
import importlib.util,re,sys,urllib.parse
from common9 import *
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
legacy_common=load_module('_E9_legacy_rights_common',ROOT/'tools/phase6c-v1/common.py')
previous=sys.modules.get('common');sys.modules['common']=legacy_common
try:legacy=load_module('_E9_legacy_rights_parser',ROOT/'tools/phase6c-v1/evidence.py')
finally:
    if previous is None:sys.modules.pop('common',None)
    else:sys.modules['common']=previous
allowed_page=legacy_common.allowed_page
account=legacy_common.account
license_kind=legacy_common.license_kind
def classify(text,row,url):
    old=legacy.classify(text,dict(source_identity=row['source_photo_id'],metadata_license_url=row['metadata_licence_url']),url)
    state=old.pop('licensing_status')
    status=STATUS_A if state==legacy_common.STATUSES[0] else STATUS_C if state==legacy_common.STATUSES[2] else STATUS_B
    # Explicit opt-out signals conflict with automatic source clearance. Their
    # absence is not invented affirmative consent or a legal interpretation.
    signals=re.findall(r'<meta\b[^>]*(?:noai|noimageai)[^>]*>',text,re.I)
    if signals:status=STATUS_B;old['reason']='Source exposes AI-use restriction/opt-out metadata;requires separate permission review'
    return dict(old,rights_status=status,source_parser_status=state,AI_restriction_meta_tags=signals,
        AI_training_specific_permission='No explicit permission is invented;source licence scope,hosting conditions and other rights remain separately assessed',
        source_ownership_scope='Documented source-provided creator/licence claim;not independent proof of chain of title,privacy,publicity,property releases or third-party content rights')
