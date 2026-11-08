"""Match individual source ImageObject licence/creator; no visual evidence."""
from html.parser import HTMLParser
from common import *

class LD(HTMLParser):
 def __init__(self):super().__init__();self.active=False;self.parts=[];self.docs=[];self.og_title=None
 def handle_starttag(self,tag,attrs):
  if tag=='script' and dict(attrs).get('type')=='application/ld+json':self.active=True;self.parts=[]
  if tag=='meta' and dict(attrs).get('property')=='og:title':self.og_title=dict(attrs).get('content')
 def handle_data(self,text):
  if self.active:self.parts.append(text)
 def handle_endtag(self,tag):
  if tag=='script' and self.active:self.docs.append(json.loads(''.join(self.parts)));self.active=False

def classify(text,row,fetch_url):
 parser=LD();parser.feed(text);objects=[];photo=row['source_identity'].removeprefix('flickr-photo:')
 def walk(x):
  if isinstance(x,dict):
   if x.get('@type')=='ImageObject' and canonical_id(x.get('acquireLicensePage',''))==photo:objects.append(x)
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for d in parser.docs:walk(d)
 objects=list({json.dumps(x,sort_keys=True):x for x in objects}.values())
 if not objects:return dict(licensing_status=STATUSES[1],reason='No unique source licence evidence bound to this exact photo ID')
 if len(objects)!=1:return dict(licensing_status=STATUSES[4],reason='Conflicting matching ImageObject evidence')
 obj=objects[0];rawlic=obj.get('license','');kind,lic=license_kind(rawlic)
 author=obj.get('author',{});canonical=obj['acquireLicensePage']
 base=dict(source_url=canonical,source_license_url=rawlic,license_url=lic,metadata_license_url=row['metadata_license_url'],title=obj.get('name') or obj.get('headline') or parser.og_title,source_copyright_notice=obj.get('copyrightNotice'),source_claimed_copyright_holder=obj.get('copyrightHolder'),
  method='Matching individual Flickr ImageObject photo ID, original COCO Flickr identity, source licence/version and creator account. HEAD is object metadata only.',
  other_rights='Source copyright-licence evidence only; no independent property, publicity/privacy, trademark or model-release clearance; no visual screening.')
 if not isinstance(author,dict) or not author.get('name') or not author.get('url') or account(author.get('url',''))!=account(canonical):return dict(base,licensing_status=STATUSES[1],reason='Creator credit missing or not bound to matching source account')
 base.update(creator=author['name'],creator_url=author['url'],creator_aliases=sorted({x for x in (account(fetch_url),account(canonical),account(author['url'])) if x}),
  attribution='Retain supplied creator/title, original photo URL, exact licence URL, notices and indication of resize/crop/annotation changes; preserve COCO Consortium / CC BY4.0 annotation attribution separately.')
 if not base['title']:return dict(base,licensing_status=STATUSES[1],reason='Missing photo-title attribution evidence')
 if kind=='outside_policy':return dict(base,licensing_status=STATUSES[2],reason='Source licence outside approved CC0/CC BY profile; no assertion that every other licence forbids commercial use')
 if kind=='unknown':return dict(base,licensing_status=STATUSES[1],reason='Missing or unrecognized specific source licence/version')
 if license_kind(row['metadata_license_url'])[1]!=lic:return dict(base,licensing_status=STATUSES[4],reason='Current specific licence differs from frozen COCO metadata claim')
 return dict(base,licensing_status=STATUSES[0],reason='Photo-specific creator/licence evidence verified under strict CC0/CC BY policy')
