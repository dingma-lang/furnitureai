"""Read immutable model metadata and saved CPU analysis; zero model forwards."""
import ast
from g5_common import *

def run():
    installed=next((ROOT/'.venv/lib').glob('python*/site-packages/transformers/models/owlv2/modeling_owlv2.py'))
    tree=ast.parse(installed.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Owlv2ClassPredictionHead')
    code=ast.get_source_segment(installed.read_text(),cls)
    for token in ('dense0','logit_shift','logit_scale','elu','einsum'):
        if token not in code:raise ValueError('Installed prediction interface changed')
    old=read(ROOT/'data/open-images/preparation/phase10c-v1/analysis.json')
    selected=read(ROOT/'data/open-images/experiments/phase10b-heads-v1/comparison-v1/results.json')
    checkpoints=[dict(label='original',path='models/owlv2-original-baseline-20261103/model.safetensors',sha256='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7')]
    checkpoints.append(dict(label='phase5c',**selected['immutable_model_hashes_verified']['adapted']))
    seals={rel:value for value,rel in (line.split('  ',1) for line in (ROOT/'data/open-images/experiments/phase10b-heads-v1/comparison-v1/CHECKPOINT-SHA256SUMS').read_text().splitlines())}
    for k in ('heads-original-lr1e-5','heads-original-lr3e-5'):
        entry=selected['entries'][k]
        p=entry['checkpoint_path'] if k.endswith('1e-5') else entry['final_recovery_checkpoint']
        expected=entry['checkpoint_sha256'] if k.endswith('1e-5') else seals[str(Path(p).relative_to('data/open-images/experiments/phase10b-heads-v1'))]
        checkpoints.append(dict(label=k+'-E5',path=p,sha256=expected))
    for c in checkpoints:
        p=ROOT/c['path']
        if sha(p)!=c['sha256']:raise ValueError('Checkpoint changed')
        c['bytes']=p.stat().st_size
    result=dict(status='READ_ONLY_CPU_AUDIT',checkpoints=checkpoints,
        installed_prediction_code_sha256=sha(installed),saved_weight_delta_report_sha256=sha(ROOT/'data/open-images/preparation/phase10c-v1/analysis.json'),
        changes=old['head_weight_changes'],
        confirmed=dict(original_is_text_conditioned_open_vocabulary=True,seven_queries_are_not_pretraining_class_limit=True,
            local_optimization_target_queries=list(CLASSES),local_nonfurniture_training_classes=[],
            adapted_modules=['class_head','box_head'],class_head_shared_between_arbitrary_text_queries=True,
            class_score='sigmoid((normalized image/text cosine + image-dependent shift) * image-dependent positive scale)',
            objectness_head_frozen=True,postprocessing='sigmoid class logits, max over furniture prompts; no objectness multiplication',
            local_backbone_text_vision_frozen=True,threshold=.15,classification_focal_alpha=.25,classification_focal_gamma=2,
            loss_coefficients=dict(classification=1,L1=5,GIoU=2),negative_mask_change='Phase10B Chair-positive-only unmatched queries lacked trustworthy Chair negative supervision'),
        untested=['Person/Plant recognition on these rooms: no saved non-furniture-query predictions',
            'General-object catastrophic forgetting versus calibration drift: shared head changes permit both, but not measured',
            'Whether auxiliary labels outperform region negatives; requires separately authorized controlled experiments'],
        preserve=dict(always_freeze=['vision encoder','text encoder','objectness'],
            consider_separate_ablation=['freeze/anchor shared classification shift and scale','freeze all original outputs with furniture-only residual']),
        docs=['https://huggingface.co/google/owlv2-base-patch16-ensemble','https://huggingface.co/docs/transformers/model_doc/owlv2'],
        model_forwards=0,optimizer_updates=0,weights_changed=False)
    destination=save_versioned(REPORT/'model-audit.json',result);print('MODEL AUDIT PASS;no model construction/forward;',destination.name)

if __name__=='__main__':run()
