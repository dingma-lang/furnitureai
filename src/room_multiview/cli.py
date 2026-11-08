import argparse
from pathlib import Path

from furniture_bench.data import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Start with one photo; progressively register overlapping room photographs"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for key in ("manifest", "output"):
        run.add_argument("--" + key, required=True)
    for key in (
        "previous",
        "layout-run",
        "segmentation-model",
        "depth-model",
        "furniture-predictions",
        "scale",
    ):
        run.add_argument("--" + key)
    evaluation = commands.add_parser("evaluate")
    for key in ("project", "reference", "output"):
        evaluation.add_argument("--" + key, required=True)
    comparison = commands.add_parser("compare")
    for key in ("single", "multi", "output"):
        comparison.add_argument("--" + key, required=True)
    sensor = commands.add_parser("sensor")
    for key in ("manifest", "layout-run", "output"):
        sensor.add_argument("--" + key, required=True)
    sensor.add_argument("--previous")
    sensor.add_argument("--stride", type=int, default=4)
    sensor_eval = commands.add_parser("evaluate-sensor")
    for key in ("project", "heldout-manifest", "reference", "output"):
        sensor_eval.add_argument("--" + key, required=True)
    refinement = commands.add_parser("refine")
    for key in ("project", "output"):
        refinement.add_argument("--" + key, required=True)
    refinement.add_argument("--max-evaluations", type=int, default=50)
    sensor_scale = commands.add_parser("scale-from-sensor")
    for key in ("project", "manifest", "output"):
        sensor_scale.add_argument("--" + key, required=True)
    odometry = commands.add_parser("recover-odometry")
    for key in ("manifest", "worker-python", "output"):
        odometry.add_argument("--" + key, required=True)
    odometry.add_argument("--anchor-mode", choices=["first", "largest-component"], default="first")
    adoption = commands.add_parser("recover-fuse")
    for key in ("baseline-project", "worker-directory", "heldout-manifest", "output"):
        adoption.add_argument("--" + key, required=True)
    adoption.add_argument(
        "--variant", choices=["hybrid", "hybrid-icp", "pose-graph"], default="hybrid"
    )
    recovery_eval = commands.add_parser("evaluate-recovery")
    for key in ("worker-directory", "reference", "baseline-project", "primary-manifest", "output"):
        recovery_eval.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    existed = Path(args.output).exists()
    try:
        if existed:
            raise ValueError("output exists; choose a new path")
        if args.command == "run":
            from .pipeline import run_project

            state = run_project(
                args.manifest,
                args.output,
                args.previous,
                args.layout_run,
                args.segmentation_model,
                args.depth_model,
                args.furniture_predictions,
                args.scale,
            )
            print(
                {
                    "views": len(state["views"]),
                    "registered": len(state["shared_geometry"]["cameras"]),
                    "landmarks": len(state["shared_geometry"]["landmarks"]),
                    "output": args.output,
                }
            )
        elif args.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate(args.project, args.reference, args.output)
            print({"registered": result["registered_views"], "landmarks": result["landmarks"]})
        elif args.command == "sensor":
            from .sensor_pipeline import run_sensor

            state = run_sensor(
                args.manifest, args.layout_run, args.output, args.previous, args.stride
            )
            s = state["sensor_reconstruction"]
            print(
                {
                    "registered": len(s["cameras"]),
                    "points": s["point_cloud"]["points"],
                    "planes": len(s["surface_planes"]),
                    "output": args.output,
                }
            )
        elif args.command == "evaluate-sensor":
            from .sensor_evaluation import evaluate_sensor

            report = evaluate_sensor(
                args.project, args.heldout_manifest, args.reference, args.output
            )
            print(
                {
                    "registered": report["registered_views"],
                    "heldout_median_metres": report["predicted_to_heldout_sensor_geometry"][
                        "median_metres"
                    ],
                }
            )
        elif args.command == "scale-from-sensor":
            from furniture_bench.data import load_manifest

            from .sensor_pipeline import load_sensor
            from .sensor_scale import scale_from_anchor_depth

            state = read_json(args.project)
            anchor = state["shared_geometry"]["anchor_view_id"]
            row = next(
                (r for r in load_manifest(args.manifest, "rooms") if r["image_id"] == anchor), None
            )
            if row is None:
                raise ValueError("anchor sensor observation absent from manifest")
            if row.get("image_sha256") != state["view_assets"][anchor]["image_sha256"]:
                raise ValueError("anchor RGB source bytes differ")
            depth, cal = load_sensor(row, args.manifest)
            scaled, report = scale_from_anchor_depth(
                state,
                depth,
                cal["intrinsics"],
                row["sensor_depth"]["source"],
                row["sensor_depth"]["sha256"],
            )
            Path(args.output).mkdir(parents=True)
            write_json(Path(args.output) / "room.json", scaled)
            write_json(Path(args.output) / "calibration.json", report)
            print({"status": report["status"], "metres_per_unit": report["metres_per_unit"]})
        elif args.command == "recover-odometry":
            from .recovery import run_odometry_worker

            report = run_odometry_worker(args.manifest, args.worker_python, args.output, args.anchor_mode)
            print({"variants": report["variants"], "output": args.output})
        elif args.command == "recover-fuse":
            from .recovery import fuse_recovered_cameras

            state = fuse_recovered_cameras(
                args.baseline_project,
                args.worker_directory,
                args.variant,
                args.heldout_manifest,
                args.output,
            )
            print(
                {
                    "registered_primary_views": len(state["sensor_reconstruction"]["cameras"]),
                    "output": args.output,
                }
            )
        elif args.command == "evaluate-recovery":
            from .recovery_evaluation import evaluate_variants

            report = evaluate_variants(
                args.worker_directory,
                args.reference,
                args.baseline_project,
                args.primary_manifest,
                args.output,
            )
            print(
                {
                    name: value["registered_original_phase5_views"]
                    for name, value in report["variants"].items()
                }
            )
        elif args.command == "refine":
            from .refinement import refine_project
            from .visualization import write_outputs

            refined, report = refine_project(read_json(args.project), args.max_evaluations)
            Path(args.output).mkdir(parents=True)
            write_json(Path(args.output) / "room.json", refined)
            write_json(Path(args.output) / "refinement.json", report)
            write_outputs(refined, args.output)
            print(report)
        else:
            from .visualization import compare

            compare(args.single, args.multi, args.output)
            print(args.output)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        import sys

        print(f"error: {exc}", file=sys.stderr)
        if (
            args.command in ("run", "sensor", "refine", "recover-odometry", "recover-fuse")
            and not existed
            and Path(args.output).is_dir()
        ):
            write_json(
                Path(args.output) / "failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "artificial_geometry_substituted": False,
                },
            )
        return 2
