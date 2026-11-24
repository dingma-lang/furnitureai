"""Generated test-only pixels and fixed queries; no real photograph or model output."""
import copy
import math
from pathlib import Path

from PIL import Image

from furniture_bench.cache import file_identity
from furniture_bench.data import CATEGORIES, digest, write_json
from furniture_bench.postprocessing import FilterConfig, filter_detections


def create_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, queries = [], {}
    for index, labels in enumerate((CATEGORIES[:3], CATEGORIES[3:], ('chair',))):
        image_id = f'SYNTHETIC-ONLY-{index}'
        path = root/f'{image_id}.png'
        image = Image.new('RGB', (120, 80), (30+index*40, 70, 140))
        if index == 1:
            exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)  # Displayed dimensions are 80 x 120.
            size = (80, 120)
        else:
            image.save(path); size = (120, 80)
        objects, predictions = [], []
        for k, label in enumerate(labels):
            box = [2+k*18, 3, 16+k*18, 23]
            objects.append(dict(object_id=f'object-{k}', label=label, box=box))
            predictions.append(dict(label=label, box=box, score=.9-k*.03, query_id=k))
        # One duplicate to suppress, one low-confidence candidate, one unmatched.
        predictions += [dict(predictions[0], score=.7, query_id=10),
                        dict(label='lamp', box=[1, 30, 12, 42], score=.05, query_id=11),
                        dict(label='chair', box=[size[0]-22, 40, size[0]-2, 60], score=.8, query_id=12)]
        for p in predictions:
            p['logits'] = [math.log(p['score']/(1-p['score'])) if c == p['label'] else -9. for c in CATEGORIES]
        # Last image is partial: known positives still match, unknowns are ignored.
        incomplete = index == 2
        rows.append(dict(image_id=image_id, file=path.name, image_sha256=digest(path),
            rights=dict(source='generated synthetic unit-test pixels', license='CC0-1.0', allowed_for_benchmark=True),
            objects=objects, source_group_id=f'SYNTHETIC-GROUP-{index}',
            partition='development' if index == 2 else 'training',
            completeness={c: 'partial' if incomplete else 'complete' for c in CATEGORIES},
            annotation=dict(status='synthetic_fixture', human_reviewed=False),
            protected_regions=[dict(box=[size[0]-22, 40, size[0]-2, 60], reason='synthetic protected seating')],
            ignore_regions=[dict(box=[size[0]-22, 40, size[0]-2, 60], reason='synthetic uncertainty')]
                if not incomplete else [dict(box=[0, 0, *size], reason='synthetic incomplete coverage')]))
        queries[image_id] = predictions
    manifest = root/'input.json'
    write_json(manifest, dict(schema_version=1, images=rows))
    weights, processor = root/'synthetic-weights.txt', root/'synthetic-processor.json'
    weights.write_text('explicitly synthetic; not loadable model weights\n')
    processor.write_text('{"synthetic":true,"preprocessing":"EXIF-RGB"}\n')
    model_files, processor_files = {'weights':weights}, {'processor':processor}
    spec = dict(model=dict(model_id='SYNTHETIC-NO-MODEL',
                          checkpoint_sha256=file_identity(model_files)['sha256'],
                          processor_sha256=file_identity(processor_files)['sha256']),
                categories=list(CATEGORIES), prompts=['synthetic '+c for c in CATEGORIES],
                query_categories=list(CATEGORIES), taxonomy_version='synthetic-chair-includes-seating-stools-v1',
                preprocessing=dict(orientation='EXIF-transpose', color='RGB', box_frame='oriented-xyxy-pixels'),
                settings=dict(precision='synthetic-float64', candidate_floor=0., adapter='fixed-queries-v1'))
    return manifest, queries, spec, model_files, processor_files


class SyntheticDetector:
    def __init__(self, queries, spec):
        self.queries, self.spec = copy.deepcopy(queries), copy.deepcopy(spec)
        self.calls = 0
        self.last_candidates = None
        self.filter_config = FilterConfig()
        self.last_filter_report = None

    def metadata(self):
        return dict(model_id=self.spec['model']['model_id'], categories=self.spec['categories'],
                    query_texts=self.spec['prompts'], query_categories=self.spec['query_categories'],
                    filter_config=FilterConfig(threshold=0.).to_dict())

    def detect(self, image):
        self.calls += 1
        # The blue channel survives the synthetic EXIF transformation; red identifies the row.
        index = (image.getpixel((0, 0))[0]-30)//40
        self.last_candidates = copy.deepcopy(self.queries[f'SYNTHETIC-ONLY-{index}'])
        predictions, self.last_filter_report = filter_detections(self.last_candidates, self.filter_config)
        return predictions, 0.


def semantic_result(run, metrics):
    return dict(images=[dict(image_id=r['image_id'], width=r['width'], height=r['height'],
                            image_sha256=r['image_sha256'], detections=[
                                {k:p[k] for k in ('label','box','score','query_id','logits')}
                                for p in r['detections']]) for r in run['images']],
                detection=metrics['detection'], per_category=metrics['per_category'],
                ignored_predictions=metrics['ignored_predictions'], per_image=metrics['per_image'])
