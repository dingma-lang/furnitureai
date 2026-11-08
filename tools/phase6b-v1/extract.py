"""Inspect every ZIP entry; extract only two bounded instance JSON members."""
import stat
import zipfile
from pathlib import PurePosixPath
from common import *

def inspect_archive(path):
 with zipfile.ZipFile(path) as z:
  names=set();records=[];selected=[]
  for item in z.infolist():
   name=item.filename;p=PurePosixPath(name);mode=item.external_attr>>16
   if name in names or p.is_absolute() or '..' in p.parts or '\\' in name or '\x00' in name or ':' in name or item.flag_bits&1:raise ValueError('Unsafe or duplicate archive entry')
   names.add(name)
   if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):raise ValueError('Special/link ZIP member')
   if item.file_size<0 or item.compress_size<0 or (item.file_size and not item.compress_size):raise ValueError('Invalid ZIP member sizes')
   if item.file_size>2000000000 or item.file_size/max(item.compress_size,1)>200:raise ValueError('Expansion bomb')
   records.append(dict(name=name,uncompressed_bytes=item.file_size,compressed_bytes=item.compress_size,crc32=f'{item.CRC:08x}',selected=name in MEMBERS))
   if name in MEMBERS:
    if item.is_dir() or item.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Unexpected selected member')
    selected.append(item)
  if {x.filename for x in selected}!=set(MEMBERS):raise ValueError('Required instance JSON absent')
  total=sum(x.file_size for x in selected)
  if total>EXTRACT_MAX:raise RuntimeError('Required extraction exceeds650MB proposal ceiling')
  return dict(members=records,selected_extracted_bytes=total,all_uncompressed_bytes=sum(x['uncompressed_bytes'] for x in records),selected_names=list(MEMBERS))

def run():
 path=OUT/'annotations_trainval2017.zip'
 if path.stat().st_size!=SIZE or sha(path)!=SHA:raise ValueError('Pinned ZIP integrity required')
 info=inspect_archive(path);space(info['selected_extracted_bytes']+DERIVED_MAX)
 write(OUT/'archive-inspection.json',info)
 target=OUT/'extracted';target.mkdir()
 rows=[]
 with zipfile.ZipFile(path) as z:
  for name in MEMBERS:
   dest=target/name;dest.parent.mkdir(exist_ok=True);partial=dest.with_suffix('.json.part')
   if dest.exists():raise FileExistsError('Extraction cannot overwrite a completed file')
   record=z.getinfo(name);count=0
   with z.open(name) as source,partial.open('xb') as f:
    while True:
     chunk=source.read(min(1048576,record.file_size-count+1))
     if not chunk:break
     count+=len(chunk)
     if count>record.file_size:raise ValueError('Member exceeds declared size')
     f.write(chunk)
    f.flush();os.fsync(f.fileno())
   if count!=record.file_size:raise ValueError('Truncated extraction')
   # ZipExtFile reaches EOF and checks the member CRC before this rename.
   os.replace(partial,dest);rows.append(dict(file=str(dest.relative_to(OUT)),bytes=count,sha256=sha(dest),crc_verified=True))
 commit(OUT/'extraction-complete.json',dict(archive_sha256=SHA,members=rows,total_extracted_bytes=sum(x['bytes'] for x in rows),skipped_members=[r['name'] for r in info['members'] if not r['selected']],photographs_extracted=0))
 print('SECURE EXTRACTION COMPLETE',info['selected_extracted_bytes'],'bytes; allocation',allocated(),flush=True)

if __name__=='__main__':run()
