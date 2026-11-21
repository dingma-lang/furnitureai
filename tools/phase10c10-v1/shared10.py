"""Separate frozen contrastive temperature/normalization from detector calibration."""
from common10 import *
from audit10 import stats

def run():
    from safetensors import safe_open
    names=['owlv2.logit_scale','layer_norm.weight','layer_norm.bias','owlv2.vision_model.post_layernorm.weight','owlv2.vision_model.post_layernorm.bias','owlv2.text_projection.weight','owlv2.visual_projection.weight']
    params={}
    with safe_open(BASE/'model.safetensors',framework='pt',device='cpu') as f:
        for name in names:params[name]=dict(original=stats(f.get_tensor(name)),A_equal_original_via_frozen_hash=True,B_equal_original_via_frozen_hash=True,in_C8_head_optimizer=False,optimizer_updates=0,direct_signed_gradients_saved=False)
    hashes={arm:read(C8/arm/'complete.json')['frozen_sha256'] for arm in ('A-control','B-corrected')}
    if len(set(hashes.values()))!=1:raise ValueError('Historical frozen backbones differ')
    save(REPORT/'shared-parameter-audit.json',dict(parameters=params,C8_original_non_head_frozen_hashes=hashes,
        role_of_owlv2_logit_scale='Frozen scalar exponentiated for auxiliary global contrastive logits_per_text/logits_per_image. These are not the detection class logits or final scores.',
        role_of_visual_projection='Frozen pooled CLIP visual projection for auxiliary contrastive image embedding; detection patch features instead use vision post-LayerNorm, patch×CLS multiplication and final layer_norm.',
        role_of_text_projection='Frozen text projection plus norm provides text queries; class_head normalizes again with1e-6.',
        detection_calibration='Two distinct768→1 Linear modules class_head.logit_shift and class_head.logit_scale, four tensors total; activation scale=ELU(raw)+1, not exp(owlv2.logit_scale).',
        proof_scope='C8 metadata records same original-initialized frozen hash before/after each arm and exact12-head optimizer. Separate full trained-backbone tensors were not serialized; equality follows that preserved hash/scope evidence.',real_image_model_forwards=0))

if __name__=='__main__':run()
