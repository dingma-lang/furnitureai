"""Genuine capture interface; no box fusion between distinct camera coordinates."""
def compare_group(views,evidence):
 if not evidence or not evidence.get('same_physical_scene_verified') or not evidence.get('provenance'):raise ValueError('Verified same-scene evidence required; creator/product ID is insufficient')
 if len({v['view_id'] for v in views})!=len(views) or len({v['image_sha256'] for v in views})!=len(views):raise ValueError('Duplicate capture, not independent views')
 categories={v['view_id']:sorted({p['label'] for p in v['predictions']}) for v in views}
 tracks=evidence.get('independently_verified_object_correspondences')
 return dict(per_view_categories=categories,per_view_predictions={v['view_id']:v['predictions'] for v in views},
  object_identity_consistency_evaluated=bool(tracks),verified_correspondences=tracks,
  category_absence_is_not_a_missed_object=True,boxes_combined_across_views=False)

def combine_camera_boxes(*args,**kwargs):raise ValueError('Cross-camera box fusion needs independently validated geometric registration; not implemented')
