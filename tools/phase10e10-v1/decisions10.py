"""Independent access/rights/content/annotation/partition gates; source claims only."""
import html,re,urllib.parse
from common10 import *
from html.parser import HTMLParser
class Text(HTMLParser):
    def __init__(self):super().__init__();self.bits=[]
    def handle_data(self,d):self.bits.append(d)
def plain(v):
    p=Text();p.feed(v or '');return re.sub(r'\s+',' ',' '.join(p.bits)).strip()
def cc_kind(url):
    u=urllib.parse.urlsplit(url)
    if u.scheme not in ('https','http') or u.hostname!='creativecommons.org':return 'unknown'
    if re.fullmatch(r'/licenses/by/(1\.0|2\.0|2\.5|3\.0|4\.0)/?',u.path):return 'CC_BY'
    if re.fullmatch(r'/publicdomain/zero/1\.0/?',u.path):return 'CC0'
    if '/by-nc' in u.path or '/by-nd' in u.path:return 'restricted'
    if '/by-sa/' in u.path:return 'CC_BY_SA_pending_policy'
    if '/publicdomain/mark/' in u.path:return 'public_domain_jurisdiction_pending'
    return 'unknown'
def category_hints(text):
    # Hints cannot establish objects, annotation completeness or visual difficulty.
    s=text.lower();patterns={'Sofa':r'\b(sofa|couch|settee)s?\b','Chair':r'\b(chair|chairs|bar stool|bar stools|seating stool|seating stools)\b',
       'Table':r'\b(table|tables|dining room|dining rooms)\b','Bed':r'\b(bed|beds|bedroom|bedrooms)\b',
       'Desk':r'\b(desk|desks)\b','Cabinet':r'\b(cabinet|cabinets|cupboard|cupboards)\b','Lamp':r'\b(lamp|lamps)\b'}
    return [k for k,p in patterns.items() if re.search(p,s)]
def assert_unique(rows):
    if len(rows)>MAX_CANDIDATES or len({r['candidate_id'] for r in rows})!=len(rows):raise ValueError('200 unique metadata candidate bound')
