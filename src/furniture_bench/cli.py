from __future__ import annotations

import argparse
import json
import sys

from .data import read_json


def model_options(parser):
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow on-demand Hugging Face downloads; default is cache/local only",
    )
    parser.add_argument("--detector-model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument(
        "--detector-revision", default="main", help="pin a hub commit for reproducibility"
    )
    parser.add_argument("--embedding-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--batch-size", type=int, default=8)


def parser():
    root = argparse.ArgumentParser(description="Local experimental furniture model benchmark")
    commands = root.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="detect, crop, visualize, and optionally retrieve")
    detect.add_argument("--manifest", required=True)
    detect.add_argument("--output", required=True, help="new experiment directory")
    detect.add_argument("--threshold", type=float, default=0.15)
    detect.add_argument("--nms-iou", type=float, default=0.5)
    detect.add_argument(
        "--prompt-bank", help="JSON category-to-query mapping; canonical labels preserved"
    )
    detect.add_argument(
        "--filter-config", help="JSON filter settings; overrides threshold/NMS flags"
    )
    detect.add_argument("--index", help="optional local FAISS catalog directory")
    detect.add_argument("--sam31-checkpoint", help="optional local SAM 3.1 multiplex checkpoint")
    model_options(detect)
    index = commands.add_parser("index", help="embed a licensed catalog and build a FAISS index")
    index.add_argument("--catalog", required=True)
    index.add_argument("--output", required=True, help="new index directory")
    model_options(index)
    evaluate = commands.add_parser(
        "evaluate", help="evaluate saved predictions without loading models"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument("--require-human-review", action="store_true")
    visualize = commands.add_parser("visualize", help="redraw saved detections on a room photo")
    visualize.add_argument("--image", required=True)
    visualize.add_argument("--predictions", required=True)
    visualize.add_argument("--image-id", required=True)
    visualize.add_argument("--output", required=True)
    correction = commands.add_parser("correct", help="append a human correction for future review")
    correction.add_argument(
        "--input", required=True, help="correction JSON; see docs/data-format.md"
    )
    correction.add_argument("--output", required=True, help="append-only JSONL log")
    compare = commands.add_parser(
        "compare-filters", help="evaluate saved raw candidates under filter configurations"
    )
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--capture", required=True)
    compare.add_argument(
        "--configs", required=True, help="JSON mapping setting names to filter configurations"
    )
    compare.add_argument("--output", required=True)
    compare.add_argument("--require-human-review", action="store_true")
    sheets = commands.add_parser(
        "annotation-sheets", help="export reference boxes for human review"
    )
    sheets.add_argument("--manifest", required=True)
    sheets.add_argument("--output", required=True)
    rebind = commands.add_parser(
        "rebind-capture", help="bind changed annotations to identical captured image bytes"
    )
    rebind.add_argument("--manifest", required=True)
    rebind.add_argument("--capture", required=True)
    rebind.add_argument("--output", required=True)
    rebind.add_argument("--reason", required=True)
    return root


def make_embedder(args):
    from .embedding.siglip2 import Siglip2Embedder

    return Siglip2Embedder(
        args.embedding_model,
        device=args.device,
        allow_download=args.allow_download,
        revision=args.embedding_revision,
        batch_size=args.batch_size,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "detect":
            from .detection.owlv2 import Owlv2Detector
            from .pipeline import run_experiment
            from .postprocessing import FilterConfig
            from .retrieval import CatalogIndex

            policy = FilterConfig.from_dict(
                {
                    "threshold": args.threshold,
                    "nms_iou": args.nms_iou,
                    **(read_json(args.filter_config) if args.filter_config else {}),
                }
            )

            detector = Owlv2Detector(
                args.detector_model,
                device=args.device,
                **policy.to_dict(),
                allow_download=args.allow_download,
                revision=args.detector_revision,
                prompt_bank=read_json(args.prompt_bank) if args.prompt_bank else None,
            )
            catalog = CatalogIndex.load(args.index) if args.index else None
            segmenter = None
            if args.sam31_checkpoint:
                from .segmentation.sam31 import Sam31Segmenter

                segmenter = Sam31Segmenter(args.sam31_checkpoint)
            try:
                run = run_experiment(
                    args.manifest,
                    args.output,
                    detector,
                    embedder=make_embedder(args) if catalog else None,
                    catalog=catalog,
                    segmenter=segmenter,
                )
            finally:
                if catalog:
                    catalog.close()
            failures = sum(row["status"] != "ok" for row in run["images"])
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "images": len(run["images"]),
                        "failed_or_partial": failures,
                    }
                )
            )
            return 2 if failures else 0
        if args.command == "index":
            from .retrieval import build_catalog

            metadata = build_catalog(args.catalog, args.output, make_embedder(args))
            print(json.dumps({"output": args.output, "products": metadata["product_count"]}))
        elif args.command == "evaluate":
            from .evaluation import evaluate

            report = evaluate(
                args.manifest,
                args.predictions,
                args.output,
                args.iou,
                require_human_review=args.require_human_review,
            )
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "detection": report["detection"],
                        "processing_failures": report["processing_failures"]["count"],
                    }
                )
            )
        elif args.command == "visualize":
            from .visualization import visualize

            rows = read_json(args.predictions)["images"]
            row = next((row for row in rows if row["image_id"] == args.image_id), None)
            if row is None:
                raise ValueError(f"unknown image_id: {args.image_id}")
            visualize(args.image, row["detections"], args.output)
            print(args.output)
        elif args.command == "correct":
            from .corrections import save_correction

            saved = save_correction(args.output, read_json(args.input))
            print(json.dumps({"output": args.output, "correction_id": saved["correction_id"]}))
        elif args.command == "compare-filters":
            from .experiments import compare_filters

            rows = compare_filters(
                args.manifest,
                args.capture,
                read_json(args.configs),
                args.output,
                args.require_human_review,
            )
            print(json.dumps({"output": args.output, "configurations": rows}))
        elif args.command == "annotation-sheets":
            from .annotations import annotation_sheets

            annotation_sheets(args.manifest, args.output)
            print(args.output)
        elif args.command == "rebind-capture":
            from .experiments import rebind_capture

            rebind_capture(args.manifest, args.capture, args.output, args.reason)
            print(args.output)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
