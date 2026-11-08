from __future__ import annotations

import argparse
from pathlib import Path

from furniture_bench.data import read_json, write_json


def main(argv=None):
    root = argparse.ArgumentParser(
        description="Automatic visible room structure and unitless relative depth"
    )
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True, help="new experiment directory")
    run.add_argument("--segmentation-model", required=True)
    run.add_argument("--depth-model", required=True)
    run.add_argument("--furniture-predictions")
    run.add_argument("--reuse-segmentation", action="append", default=[])
    run.add_argument("--reuse-depth", action="append", default=[])
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--min-confidence", type=float, default=0.55)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--require-human-review", action="store_true")
    correction = commands.add_parser("correct")
    correction.add_argument("--scene", required=True)
    correction.add_argument("--corrections", required=True)
    correction.add_argument("--output", required=True)
    args = root.parse_args(argv)
    try:
        if Path(args.output).exists():
            raise ValueError("output exists; earlier experiments are never overwritten")
        if args.command == "run":
            from .pipeline import run_layout

            report = run_layout(
                args.manifest,
                args.output,
                args.segmentation_model,
                args.depth_model,
                args.furniture_predictions,
                args.reuse_segmentation,
                args.reuse_depth,
                args.threads,
                args.min_confidence,
            )
            failures = sum(r["status"] != "ok" for r in report["images"])
            print(
                {
                    "output": args.output,
                    "images": len(report["images"]),
                    "failed_or_partial": failures,
                }
            )
            return 2 if failures else 0
        if args.command == "evaluate":
            from .evaluation import evaluate_layout

            report = evaluate_layout(
                args.manifest, args.run, args.output, args.require_human_review
            )
            print(
                {
                    "human_reviewed_images": report["human_reviewed_images"],
                    "floor_iou": report["floor_iou"],
                    "wall_iou": report["wall_iou"],
                }
            )
        else:
            from .corrections import apply_corrections

            write_json(
                args.output, apply_corrections(read_json(args.scene), read_json(args.corrections))
            )
            print(args.output)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2
