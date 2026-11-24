"""Local visual-review records, inactive E15 intake and read-only gallery."""
import csv
import html
import io
from itertools import combinations
from PIL import Image,ImageOps
from common16 import *
sys.path.insert(0,str(ROOT/'tools/phase10e15-v1'))
from intake15 import validate,SCHEMA,TAXONOMY,FRAME
from screen16 import perceptual


OBSERVATIONS={
 'commons-page:150832261':dict(suitability='C',
    reason='Tight computer/work-surface close-up, not a room-context furniture scene. Whole Desk versus generic worktop cannot be established from this crop.',
    clear_categories=[],possible_categories=['Desk','Table'],
    observed='Two laptops, large monitor, keyboard, mouse, cables, sticky notes and a severely cropped wooden work surface. No clearly identifiable whole Chair or seating stool established.',
    confusions=['screens/laptops','monitor stand','keyboard/mouse','cables','wooden surface edges'],
    annotation_requirements='Do not label a whole Desk from the title. Preserve partial worktop ambiguity; no source boxes exist. Not recommended for furniture-positive annotation.',
    hard_negative_potential='Computer/monitor and desk-edge distractors only; no verified Chair absence, negative rectangle or query evidence.',
    privacy='Readable screen/code content and notes require privacy review. No identities or text are transcribed into labels.',
    quality='Sharp equipment detail, but strong foreground cropping and almost no room context.'),
 'commons-page:178637758':dict(suitability='B',
    reason='Genuine residential holiday scene with useful storage/table context, but softness, clipping, lamp extent and ottoman/seating ambiguity require review.',
    clear_categories=['Table','Cabinet'],possible_categories=['Lamp','Chair'],
    observed='Enclosed television storage cabinet, clipped foreground ornate coffee-table-like furniture, partial lamp on the right and partial drawer storage. Striped ottoman/footstool-like object has unresolved seating function.',
    confusions=['ottoman/footstool-like object','television and reflections','Christmas tree and lights','fireplace/mantel','candles','cards/frames','cups and decorations'],
    annotation_requirements='Review visible Table/Cabinet extents, truncation and lamp type. Do not silently make the ottoman-like object Chair or Chair-negative. Preserve partial drawers versus whole storage units.',
    hard_negative_potential='Decorations, fireplace, TV, handles and striped footstool-like furniture are potential confusion probes, not trusted negatives without class-specific evidence.',
    privacy='No live person is evident; small framed material/cards may need review. No blanket privacy release is inferred.',
    quality='Moderate softness and uneven indoor light; date imprint and border truncation; adequate for large-object review.'),
 'commons-page:195256255':dict(suitability='A',
    reason='Best modern-room candidate: furniture in realistic window light, partial occlusion, reflective floor and through-glass context.',
    clear_categories=['Sofa','Chair','Table'],possible_categories=[],
    observed='Connected sectional sofa clipped at the left edge, a large foreground coffee table, and two patio chairs visible through the glass. A small table between patio chairs needs separate extent review.',
    confusions=['chair reflections/shadows','window/door frames','curtains','candles and candlesticks','wine bucket/bottles/glasses','outdoor tub-like structure','outdoor trees'],
    annotation_requirements='One connected sectional object, not multiple sofa parts; preserve truncation. Annotate actual patio chairs separately from reflections. Review coffee-table function/extent and the smaller occluded patio table; do not label storage panels as a second Cabinet.',
    hard_negative_potential='Candlesticks versus Lamp, door frames/reflections versus Chair, glossy surfaces and patio objects. No new trusted negative region is activated.',
    privacy='No person or plainly identifying personal document is evident; copyright declaration does not establish all property/privacy rights.',
    quality='High resolution and clear large furniture; dark interior against bright outdoors, occlusion and reflections are useful challenges.')
}


def duplicates(records):
    context=read(ROOT/'data/open-images/preparation/phase10e1-v1/training-context.json')
    refs=[r['row'] for r in context['entries']]+context['development']
    if len(refs)!=78:raise ValueError('Exact59 optimization/19 exposed development references required')
    reference=[]
    for r in refs:
        if r['historical_partition']!='train' or r['internal_partition'] not in ('optimization','inner_development'):
            raise ValueError('Reserved/frozen reference disallowed')
        p=ROOT/r['path']
        if sha(p)!=r['sha256']:raise ValueError('Historical photo changed')
        with Image.open(p) as im:h=perceptual(ImageOps.exif_transpose(im).convert('RGB'))
        reference.append(dict(image_id=r['image_id'],sha256=r['sha256'],**h))
    def distances(a,b):return {k:(int(a[k],16)^int(b[k],16)).bit_count() for k in ('dhash','ahash')}
    exact=[];near=[];cross=[]
    for a,b in combinations(records,2):
        if a['sha256']==b['sha256']:exact.append([a['source_id'],b['source_id']])
        d=distances(a['quality'],b['quality'])
        if min(d.values())<=6:near.append(dict(pair=[a['source_id'],b['source_id']],**d))
    for a in records:
        for b in reference:
            d=distances(a['quality'],b)
            if a['sha256']==b['sha256'] or min(d.values())<=6:
                cross.append(dict(pilot=a['source_id'],reference=b['image_id'],exact=a['sha256']==b['sha256'],**d))
    return dict(exact_pilot_pairs=exact,perceptual_pilot_flags=near,exposed_reference_flags=cross,
        reference_count=78,frozen_or_reserved_photos_decoded=0,heuristics_not_independence_proof=True,
        verified_physical_room_or_capture_independence=False)


def run():
    closed=read(OUT/'acquisition-closed.json')
    if closed['stop'] or closed['downloaded_count']!=3:raise ValueError('Exact successful closed pilot required')
    records=closed['records'];exclusions=read(ROOT/'data/open-images/preparation/phase10e6-v1/independent-exclusion-registry.json')
    intake_results=[];gallery=[];ledger=read(OUT/'rights-ledger.json')
    for r in records:
        ident=r['source_id'].split(':')[1];p=ROOT/r['file'];obs=OBSERVATIONS[r['source_id']]
        if sha(p)!=r['sha256']:raise ValueError('Acquired image changed')
        # Actual public copyright evidence is separate from absent human/third-party permissions.
        intake=dict(schema=SCHEMA,taxonomy_version=TAXONOMY,photo_ids=[r['source_id']],
            creator_id=r['source']['source_group_id'],source_group_id=r['source']['source_group_id'],
            capture_group_id='unknown-capture:'+r['source']['source_group_id'],
            room_group_ids=['unknown-room:'+r['source']['source_group_id']],
            source=r['source']['source_page'],credit=r['rights']['creator_credit']+'; '+r['rights']['license'],
            purpose='training',exposure_status='screened_by_assistant_no_detector_inference',
            permission_basis='file_specific_public_licence_declaration_not_a_new_creator_grant',
            permission_evidence=str((OUT/'rights-ledger.json').relative_to(ROOT)),permission_evidence_sha256=sha(OUT/'rights-ledger.json'),
            permission_date=r['rights']['checked_utc'][:10],explicit_confirmation=False,
            privacy_and_third_party_rights_resolved=False,orientation_explicitly_checked=False,
            scopes={k:False for k in ('storage','annotation_and_modification','model_training_or_evaluation','commercial_model_distribution')},
            photo=dict(image_sha256=r['sha256'],width=r['quality']['display_dimensions'][0],height=r['quality']['display_dimensions'][1],
                       bytes=r['bytes'],box_frame=FRAME,original_filename=p.name),
            annotations=[],source_annotations=None,class_completeness={c:dict(status='unknown') for c in ('sofa','chair','table','bed','desk','cabinet','lamp')},
            human_reviewed=False,annotations_active=False,training_authorized=False,assistant_visual_screening=obs)
        result=validate(intake,exclusions=exclusions,image_path=p)
        if not result['image_content_verified'] or result['annotations_active'] or result['eligible_for_training']:
            raise ValueError('Intake identity/gating failure')
        emit(OUT/'intake'/(ident+'.json'),intake)
        emit(OUT/'intake'/(ident+'-validation.json'),result)
        intake_results.append(dict(source_id=r['source_id'],actual_image_content_verified=True,
            status=result['status'],expected_missing_human_or_privacy_confirmations=result['errors'],
            native_boxes_available=0,annotations_imported_or_fabricated=0,annotations_active=False))
        item=dict(**r,visual_review=obs,human_reviewed=False,assessment='assistant visual screening, not exhaustive labels',
                  independent_evaluation_eligible=False,source_capture_identity='unverified; uploader-account bucket only')
        gallery.append(item)
    emit(OUT/'visual-review.json',dict(records=gallery,suitability_counts={'A':1,'B':1,'C':1},
        qualified_hard_negatives=0,activated_annotations=0,models_run=0,seven_class_absence_certified=False))
    emit(REPORT/'visual-review.json',read(OUT/'visual-review.json'))
    emit(REPORT/'intake-validation.json',dict(records=intake_results,native_annotation_import='not applicable: zero source annotations',
        source_import_correction_regressions='E15 synthetic tests only; real photographs have no fabricated boxes',
        all_human_confirmations_preserved_unconfirmed=True,independent_evaluation_allocation=0))
    emit(REPORT/'duplicate-screening.json',duplicates(records))
    emit(REPORT/'network-and-provenance.json',closed['network'])
    emit(REPORT/'image-and-rights-manifest.json',dict(records=[dict(source_id=r['source_id'],file=r['file'],bytes=r['bytes'],
        dimensions=r['quality']['display_dimensions'],sha256=r['sha256'],source_sha1=r['sha1'],rights=r['rights'],
        source_group_id=r['source']['source_group_id'],suitability=OBSERVATIONS[r['source_id']]['suitability'],
        training_active=False,evaluation_active=False) for r in records]))
    text=io.StringIO();writer=csv.writer(text);writer.writerow(['source_id','file','bytes','width','height','sha256','source_sha1','creator','licence','source_url'])
    for r in records:writer.writerow([r['source_id'],r['file'],r['bytes'],*r['quality']['display_dimensions'],r['sha256'],r['sha1'],r['rights']['creator_credit'],r['rights']['license'],r['rights']['source_object_url']])
    save(REPORT/'image-sha256-and-provenance.csv',text.getvalue().encode())
    esc=html.escape;sections=[]
    for r in gallery:
        ident=r['source_id'].split(':')[1];v=r['visual_review'];rights=r['rights']
        details=''.join('<p><strong>'+esc(k.replace('_',' ').title())+': </strong>'+esc(str(v[k]))+'</p>' for k in
                        ('observed','confusions','annotation_requirements','hard_negative_potential','privacy','quality'))
        sections.append('<section><h2>'+esc(r['source']['native_file_title'])+' — '+v['suitability']+'</h2><a href="/images/'+ident+'.jpg"><img src="/previews/'+ident+'.jpg" alt="'+esc(r['source']['native_file_title'])+'"></a><p>Click for unchanged original. '+esc(rights['creator_credit'])+' — <a href="'+esc(rights['license_url'].replace('http:','https:'))+'">'+esc(rights['license'])+'</a> — <a href="'+esc(rights['source_page'])+'">source</a>. Preview is an EXIF-oriented resized derivative.</p><p>'+esc(v['reason'])+'</p><p>Observed candidate categories: '+esc(', '.join(v['clear_categories']) or 'none established')+'. Uncertain: '+esc(', '.join(v['possible_categories']) or 'none assigned')+'. These are not verified labels.</p>'+details+'<p>Source/capture independence unverified. No training/evaluation activation or human annotations.</p></section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Furniture AI — exact three-photo pilot</title><style>body{font:17px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;line-height:1.5}section{border-top:1px solid #ddd;margin:2rem 0}img{max-width:100%;height:auto}aside{background:#fff0cc;padding:1rem}</style><h1>Modern-room suitability pilot</h1><aside>Three approved originals only. Assistant visual screening, not ground truth. A:1 / B:1 / C:1. No detector inference, negative labels, dataset activation or independent evaluation membership.</aside><p><a href="/intake.html">Open isolated E15 intake page</a> — choose the matching local original; grants and labels are not auto-confirmed.</p>'+''.join(sections)+'</html>'
    save(OUT/'gallery.html',page.encode())
    print('Gallery and three inactive E15 intake records created; A1/B1/C1; no boxes fabricated')


if __name__=='__main__':run()
