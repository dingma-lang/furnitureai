"""Phase10C.5 local-only contracts; no inference/training execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c5-v1'
REPORT=ROOT/'reports/phase10c5-v1'
sys.path.insert(0,str(ROOT/'tools/phase10c3-v1'))
from preflight import TAXONOMY,read,save,sha,digest,manifest,optimizer_member
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
SCHEMA='phase10c5-fast-review-v1'

def save_versioned(path,value):
    """Never replace even this phase's generated reports on a later rerun."""
    path=Path(path)
    if path.exists() and read(path)==value:return path
    candidate=path;version=2
    while candidate.exists():
        candidate=path.with_name(path.stem+'-v'+str(version)+path.suffix);version+=1
        if candidate.exists() and read(candidate)==value:return candidate
    save(candidate,value);return candidate
GROUPS=('people','people_parts','plants_flowers','animals','wall_art','windows_doors','screens','kitchen_appliances','tableware_small_objects','clothing_bags','furniture_parts','walls_floors_ceilings','fragments_reflections_shadows')

def normal(name):return ' '.join(name.lower().replace('_',' ').split())

NAMES={
 'people':{'person','man','woman','boy','girl'},
 'people_parts':{'human face','human head','human hand','human leg','human foot','hand','foot','head'},
 'plants_flowers':{'plant','houseplant','potted plant','flower','flower arrangement','sunflower','christmas tree','tree'},
 'animals':{'animal','cat','dog','bird','horse','sheep','cow','elephant','bear','zebra','giraffe','rabbit','squirrel','turtle','fish','domestic cat'},
 'wall_art':{'painting','picture frame','poster','photograph','artwork','sculpture','statue'},
 'windows_doors':{'window','door','window blind','curtain','blinds'},
 'screens':{'television','television set','tv','computer monitor','monitor (computer equipment) computer monitor','laptop','laptop computer','computer keyboard','keyboard','computer mouse'},
 'kitchen_appliances':{'refrigerator','microwave','microwave oven','oven','toaster','toaster oven','coffee maker','dishwasher','kettle','blender','mixer'},
 'tableware_small_objects':{'cup','coffee cup','teacup','dixie cup','mug','wine glass','glass','bottle','bowl','plate','paper plate','salad plate','dish','fork','knife','spoon','book','vase','flowerpot','tissue box','napkin','pencil','pencil box','pencil holder','pen','remote','remote control','clock','wall clock','cell phone','telephone','mobile phone','umbrella','teddy bear'},
 'clothing_bags':{'clothing','shirt','polo shirt','t shirt','tie','handbag','bag','backpack','suitcase','tote bag','duffel bag','clutch bag','shopping bag','shoulder bag','plastic bag','coat','jacket','jeans','shoe','shoes','hat','blanket','pillow'},
 'furniture_parts':{'drawer','door handle','doorknob','tabletop','table leg','chair leg','cabinet door'},
 'walls_floors_ceilings':{'wall','floor','ceiling'},
 'fragments_reflections_shadows':{'shadow','reflection'},
}
FURNITURE={'chair','armchair','rocking chair','folding chair','deck chair','highchair','stool','music stool','bar stool','bench','sofa','couch','sofa bed','bed','table','dining table','coffee table','desk','cabinet','cupboard','nightstand','bookshelf','bookcase','shelf','lamp','table lamp','floor lamp','ottoman','footstool','step stool','wheelchair'}

def category_group(name):
    n=normal(name)
    if n in FURNITURE:return 'protected_furniture'
    for group,names in NAMES.items():
        if n in names:return group
    return None

def overlap(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def normalized_box(native,w,h):
    if min(w,h)<=0:raise ValueError('Invalid source dimensions')
    x,y,bw,bh=native;b=[x/w,y/h,(x+bw)/w,(y+bh)/h]
    if not(0<=b[0]<b[2]<=1.00000001 and 0<=b[1]<b[3]<=1.00000001):raise ValueError('Unsafe or out-of-frame native box; no silent clipping')
    if b[2]>1 or b[3]>1:raise ValueError('Native box exceeds original image')
    return b
