"""Reuse the sealed bounded JSON/ZIP reader without changing historical code."""
import importlib.util
import sys
import zipfile
from core import *

sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
spec=importlib.util.spec_from_file_location('sealed_phase7_audit',ROOT/'tools/phase7-v1/audit.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
stream=old.stream

def verify_archives():
 records=[];preflight=read(OUT/'preflight.json')
 for row in proposal()['objects']:
  path=OUT/row['expected_local_filename']
  if path.stat().st_size!=row['bytes'] or sha(path)!=preflight['published_sha256'][path.name]:raise ValueError('Archive published checksum/size mismatch')
  with zipfile.ZipFile(path) as z:
   name=old.safe_lvis_member(z);entry=z.getinfo(name)
   with z.open(name) as f:
    size=0
    while b:=f.read(1048576):
     size+=len(b)
     if size>entry.file_size or size>4*1024**3:raise ValueError('Archive expansion limit')
   if size!=entry.file_size:raise ValueError('Uncompressed size mismatch')
   records.append(dict(file=path.name,compressed_bytes=path.stat().st_size,sha256=sha(path),member=name,uncompressed_bytes=size,crc32=f'{entry.CRC:08x}',crc_verified=True))
 space()
 result=dict(status='PASS',archives=records,archive_bytes=sum(r['compressed_bytes'] for r in records),full_extracted_bytes=sum(r['uncompressed_bytes'] for r in records),full_extraction_performed=False,
  archive_plus_full_extraction_bytes=sum(r['compressed_bytes']+r['uncompressed_bytes'] for r in records),processing='Verified ZIP streaming; no full extraction',allocated_bytes=allocated())
 commit(OUT/'archive-integrity.json',result);return result

if __name__=='__main__':print(verify_archives())
