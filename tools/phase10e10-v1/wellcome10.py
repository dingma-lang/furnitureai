"""First-party Wellcome image-level catalogue evidence; unknown people/groups stay unknown."""
import urllib.parse
from common10 import *
from decisions10 import cc_kind,category_hints,plain
THEMES=['living room','office','dining room','kitchen','lamp','cabinet','ward','sofa','chair','desk','table','interior','furniture','sitting room']
def query_url(theme):
    return 'https://api.wellcomecollection.org/catalogue/v2/images?'+urllib.parse.urlencode(dict(query=theme,pageSize=20,include='source.contributors,source.genres,source.subjects',**{'locations.license':'cc-by,cc0','source.genres.label':'Photographs'}))
def adapt(item,receipt,theme):
    source=item.get('source',{});locs=item.get('locations',[])
    licences={l.get('license',{}).get('url') for l in locs if l.get('license')};lic=next(iter(licences)) if len(licences)==1 else None
    kind=cc_kind(lic or '');credits=sorted({l.get('credit') for l in locs if l.get('credit')})
    contributors=[dict(name=c.get('agent',{}).get('label'),id=c.get('agent',{}).get('id'),roles=c.get('roles',[]),primary=c.get('primary',False)) for c in source.get('contributors',[])]
    names=sorted({c['name'] for c in contributors if c['name']})
    terms=[a for l in locs for a in l.get('accessConditions',[]) if a.get('terms') or a.get('status',{}).get('id') not in ('open',None)]
    status='B_unresolved';reason='Missing licence, attribution, or conflicting source terms'
    if kind=='restricted':status='C_unsuitable';reason='Image-specific NC/ND licence conflicts with proposed policy'
    elif kind in ('CC_BY','CC0') and (credits or names) and len(licences)==1 and not terms:
        status='A_rights_supported';reason='First-party catalogue attaches CC BY/CC0 and attribution to this exact image; ownership and noncopyright rights are not guaranteed'
    title=source.get('title','');subjects=[s.get('label','') for s in source.get('subjects',[])];genres=[g.get('label','') for g in source.get('genres',[])]
    texts=' '.join([title,*subjects]);info=next((l.get('url') for l in locs if l.get('locationType',{}).get('id')=='iiif-image'),None)
    # Contributor metadata may include organisations and non-photographer roles.
    known_photo_names=[c['name'] for c in contributors if c['name'] and any('photograph' in r.get('label','').lower() for r in c['roles'])]
    if names:
        group='wellcome-declared-contributor:'+hashlib.sha256('|'.join(names).casefold().encode()).hexdigest()[:24]
        basis='All named contributor metadata grouped conservatively; photographer role/aliases not necessarily established'
    else:
        group='wellcome-unknown-creator-shared';basis='All unknown creators provisionally grouped together; not treated as independent rooms or photographers'
    nativeid=urllib.parse.urlsplit(info or '').path.split('/')[-2] if info else None
    return dict(candidate_id='wellcome-image:'+str(item['id']),source='Wellcome Collection',source_image_id=item['id'],source_work_id=source.get('id'),
       canonical_source_url='https://wellcomecollection.org/works/'+source['id'] if source.get('id') else None,
       image_metadata_URL=info,original_image_id=nativeid,exact_download_URL=None,expected_bytes=None,
       photographer=known_photo_names or None,declared_contributors=contributors,rights_holder=None,licence_grant_evidence_provider='Wellcome Collection first-party image catalogue',
       credit=credits,licence_url=lic,licence_family=kind,rights_status=status,rights_reason=reason,access_terms=terms,
       attribution_requirements=dict(source_credits=credits,declared_contributors=names,licence_URL=lic,changes_must_be_indicated=kind=='CC_BY',source_link_required=kind=='CC_BY'),
       commercial_conditions='CC BY permits commercial copying/adaptation subject to its conditions; CC0 dedicates covered rights. No endorsement, ownership warranty, privacy/publicity or trademark clearance inferred.',
       modification_redistribution_conditions='Retain credit, link licence, indicate modifications; do not impose inconsistent restrictions on licensed photograph',
       intended_ML_use='Conditionally supported for photograph copying/annotation by the attached grant; not an explicit guarantee of model rights or all jurisdictions. Image-level third-party/personal rights still need review.',
       evidence=dict(url=receipt['url'],sha256=receipt['sha256'],response_path=receipt['file'],checked_UTC=receipt['checked_utc']),
       metadata_title=title,metadata_subjects=subjects,metadata_genres=genres,discovery_themes=[theme],category_hints=category_hints(texts),
       distractor_hints=[w for w in ('person','patient','plant','television','appliance','kitchen','clutter','office') if w in texts.lower()],
       width=None,height=None,aspect_ratio=item.get('aspectRatio'),native_annotations=dict(boxes=0,masks=0,negative_completeness=False),
       provider_access=dict(status='documented_public_no_auth_API',evidence='https://raw.githubusercontent.com/wellcomecollection/catalogue-api/main/README.md'),
       visual_relevance=dict(status='metadata_only',photograph_genre_claim='Photographs' in genres,room_scene_verified=False,visual_difficulty_verified=False,people_present_unverified=True),
       source_group_id=group,group_basis=basis,capture_group_id=None,creator_identity_verified=False,
       partition_eligibility='pending_source_and_capture_identity_checks',final_partition=None,photo_download_approved=False,
       verified_negative_regions=0,Chair_absence_verified=False,human_confirmed=False,
       eligible_for_later_download_proposal=status=='A_rights_supported' and bool(info))
