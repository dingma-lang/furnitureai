"""Small report tables/plot and disabled next-step proposal, CPU metadata only."""
import csv,os,platform,sys
import numpy as np
from common14 import *


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states=['original','c8_corrected_e5','c11_e5','c13_e5'];names=['Original','C8 B E5','C11 E5','C13 E5']
    stats=read(REPORT14/'query-assignment-summary.json')['state_summaries'];mechanism=read(REPORT14/'training-mechanism.json');historical=read(REPORT14/'historical-verification.json')
    path=REPORT14/'training-query-comparison.csv'
    with path.open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['state','positive_terms','Chair_negative_terms','ignored_Chair_p_ge_015','positive_output_gradient_L1','source_negative_output_gradient_L1','counterfactual_source_BCE_output_gradient_L1','mean_classification_loss','mean_source_negative_loss'])
        for key in states:
            s=stats[key];writer.writerow([key,sum(s['positive_terms_by_category']),s['negative_terms_by_category'][1],s['context']['high_Chair_ignored_queries'],s['logit_positive_L1'],s['logit_source_negative_L1'],s['counterfactual_source_BCE_logit_gradient_L1'],s['mean_classification_loss'],s['mean_source_negative_loss']])
    plt.rcParams.update({'svg.hashsalt':'phase10c14-v1','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained');x=np.arange(4)
    axes[0].bar(x-.19,[stats[s]['logit_positive_L1'] for s in states],.36,label='Positive')
    axes[0].bar(x+.19,[stats[s]['logit_source_negative_L1'] for s in states],.36,label='Source negative');axes[0].set_yscale('log');axes[0].set_ylabel('Sum |dL/dlogit| over 24 training photos');axes[0].set_title('Sparse objective gradient imbalance');axes[0].legend()
    axes[1].bar(x,[stats[s]['context']['high_Chair_ignored_queries'] for s in states],color='#a66a39');axes[1].set_ylabel('Raw queries; no NMS or FP claim');axes[1].set_title('Ignored Chair scores ≥0.15')
    colors=['#477891','#8b5a91']
    for offset,state,color in zip([-.18,.18],['c11_e5','c13_e5'],colors):axes[2].bar(np.arange(7)+offset,[mechanism['cached_training_logit_drift'][state]['mean_logit_change_by_category'][c] for c in CLASSES],.34,label=state,color=color)
    axes[2].set_xticks(np.arange(7),CLASSES,rotation=45,ha='right');axes[2].set_ylabel('Mean original-relative logit change');axes[2].set_title('Drift includes unsupported categories');axes[2].legend()
    for ax in axes[:2]:ax.set_xticks(x,names,rotation=25,ha='right')
    fig.suptitle('Saved optimization-training probes only · reconstructed CPU derivatives · no new model inference',fontsize=11)
    p=REPORT14/'training-query-evidence.svg'
    if p.exists():raise FileExistsError(p)
    fig.savefig(p,metadata={'Date':None});plt.close(fig)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    old=read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json');bundle=SOURCE/'source-negative-evidence.json'
    save(REPORT14/'next-experiment-proposal.json',dict(schema='phase10c14-source-negative-gradient-proposal-v1',approved=False,training_authorized=False,GPU_authorized=False,Phase10D_authorized=False,
        hypothesis='Existing verified Chair negatives are downweighted too strongly by gamma2 focal loss;remove only their focal modulation while preserving positive/native losses and ignorance masks',
        intervention=dict(existing_source_negative='0.75 * sigmoid(z)^2 * softplus(z) / number_active_GT',proposed_source_negative='0.75 * softplus(z) / number_active_GT',positive_and_native_negative='Unchanged alpha=.25,gamma=2 focal loss',matching='Unchanged executed C8 Hungarian cost and whole-query source mask',other_changes=False,implemented_in_trainer=False),
        training_data=dict(manifest=old['data']['manifest'],sha256=old['data']['manifest_sha256'],training_rows_sha256=old['data']['training_rows_sha256'],optimization_photos=59,known_groups=56,source_negative_regions=41,evidence_path=str(bundle.relative_to(ROOT)),evidence_sha256=sha(bundle),new_labels=0,unknown_regions='Ignored;never generic background'),
        trainable_parameter_names_for_later_training=old['trainable_keys'],frozen_calibration=old['frozen_calibration_keys'],all_other_parameters_frozen=True,
        next_approval=dict(type='Zero-update GPU gradient smoke only;not training',photos='Exact24 optimization photos frozen in C6 proposed-smoke-photos.json',photo_manifest_sha256=sha(SOURCE/'proposed-smoke-photos.json'),initialization='Immutable original base plus exact saved C11 final head;diagnose an already inflated state without altering it',base_sha256=old['initialization']['sha256'],head=historical['checkpoints']['c11'],max_forwards=24,max_optimizer_updates=0,max_GPU_seconds=180,max_allocated_VRAM_bytes=6979321856,max_new_output_bytes=134217728,destination='data/open-images/experiments/phase10c15-source-negative-gradient-smoke-v1/',no_destination_created=True,
            procedure='One FP16 forward/photo;two analytic loss components on identical outputs;autograd.grad for positive/native/source/projection/box gradient norms and directions;no optimizer,weight updates,model.grad accumulation,development scoring,or additional forwards',
            pass_requirements=['All source/photo/model/config hashes match','Identical matching,positive/native masks and source-region attribution between loss formulas','No positive,stool,protected or unknown terms newly marked negative','Finite output/head gradient components with explicit nonzero checks on covered source terms','Report any zero or dominant source contribution;do not infer training benefit from loss/gradient magnitude'],stop_on_failure=True,automatic_training_afterwards=False),
        possible_later_separate_training=dict(authorized=False,arms=1,reference='Historical C11 same frozen-calibration eight-tensor scope,no C13 L2SP',initialization='Original pretrained,not continuation of C11',seed=6106,LR=1e-5,max_epochs=5,max_updates=75,max_GPU_seconds_including_diagnostics=600,max_allocated_VRAM_bytes=6979321856,changes='Only additional source-negative focal modulation removed;no lambda,matching,threshold,data or seed search',requires='Successful separately approved gradient smoke and a new approval bound to exact configuration/code'),
        evaluation=dict(primary='Predeclared final endpoint,unchanged partial-label matching and confidence0.15/AP protocol',comparison='Saved original,C11,and later one-arm endpoint under identical definitions',metrics=['precision and recall at0.15','PR-envelope precision at original recall','AP50/AP75','scored TP/FP/FN and per-class results','unknown unmatched counts separately','Sofa/Bed regressions','query logit drift','source-negative/projection gradient contribution'],acceptance='Only supports further investigation if matched-recall precision improves without Sofa/Bed recall or localization regression;diagnostic19photo panel cannot establish independent success',independent_goal='80% precision AND90%recall on a future fully annotated,identity-isolated seven-class real-room set;no evaluation access now',new_evaluation_or_frozen_scoring=False),
        risks=['BCE can still be too weak or can over-dominate;output derivatives do not establish head gradients under AMP','Same limited41regions/24covered regions may not constrain unseen confusions','Chair-only supervision cannot solve all seven-category coverage','Teacher preservation can reproduce errors or suppress new positives','Already exposed19-photo development panel is diagnostic,not acceptance evidence']))
    import torch
    save(REPORT14/'execution-provenance.json',dict(parent_commit='3676f652aa39fa0ab6ca83f1a1a5be12e33a94ef',branch='experiment/phase10c14-training-query-audit',project=str(ROOT),Python=sys.executable,Python_version=platform.python_version(),torch=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_initialized=torch.cuda.is_initialized(),model_forwards=0,GPU_training=0,optimizer_updates=0,downloads=0,environment_changes=[],new_review_decisions=0,source_annotations_activated=0,development_used_for_mining_or_training=False,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE14.iterdir()) if p.is_file()},private_output_root=str(RAW14.relative_to(ROOT)),public_report_root=str(REPORT14.relative_to(ROOT))))
    print('C14 small report tables/plot and disabled proposal written',flush=True)


if __name__=='__main__':run()
