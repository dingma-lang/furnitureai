from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Editable metric room geometry demonstration")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="export the known-dimension replacement demonstration")
    demo.add_argument("--output", required=True, help="new output directory")
    check = commands.add_parser("check", help="check an edited room JSON")
    check.add_argument("--room", required=True)
    replace = commands.add_parser("replace", help="test a replacement and save only if it fits")
    replace.add_argument("--room", required=True)
    replace.add_argument("--remove-id", required=True)
    replace.add_argument("--replacement", required=True)
    replace.add_argument("--output", required=True)
    dimensions = commands.add_parser(
        "room-from-dimensions", help="create an empty room from declared metric dimensions"
    )
    dimensions.add_argument("--input", required=True)
    dimensions.add_argument("--output", required=True)
    calibrate = commands.add_parser(
        "calibrate-floor", help="estimate planar perspective from user-supplied correspondences"
    )
    calibrate.add_argument("--controls", required=True)
    calibrate.add_argument("--output", required=True)
    project = commands.add_parser(
        "project-contacts", help="project tentative box-bottom floor contacts, not object centres"
    )
    project.add_argument("--calibration", required=True)
    project.add_argument("--predictions", required=True)
    project.add_argument("--image-id", required=True)
    project.add_argument("--output", required=True)
    associate = commands.add_parser(
        "associate", help="attach reviewed photo observations without applying metric geometry"
    )
    associate.add_argument("--room", required=True)
    associate.add_argument("--observation", required=True)
    associate.add_argument("--entity-id", required=True)
    associate.add_argument("--reviewer", required=True)
    associate.add_argument("--evidence", required=True)
    associate.add_argument("--demo-only", action="store_true")
    associate.add_argument(
        "--contact-estimate",
        help="optional ground-contact JSON; never applied as object-centre geometry",
    )
    associate.add_argument("--output", required=True)
    associate.add_argument("--link-output", required=True)
    assess = commands.add_parser(
        "assess", help="evaluate separate geometric fit and arrangement targets"
    )
    assess.add_argument("--room", required=True)
    assess.add_argument("--policy", required=True)
    assess.add_argument("--output", required=True)
    return root


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new_output(path):
    if Path(path).exists():
        raise ValueError("output already exists; choose a new path")


def _extended_command(args):
    from .model import load_scene, save_json

    _new_output(args.output)
    if args.command == "room-from-dimensions":
        from .calibration import room_from_dimensions

        specification = _read(args.input)
        room = room_from_dimensions(
            specification["dimensions"], specification.get("room_id", "provided-room")
        )
        save_json(args.output, room)
    elif args.command == "calibrate-floor":
        import hashlib

        from .calibration import calibrate_floor

        config = _read(args.controls)
        if config["image"].get("file"):
            image_path = Path(args.controls).parent / config["image"]["file"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != config["image"]["sha256"]:
                raise ValueError("calibration source image checksum mismatch")
        save_json(args.output, calibrate_floor(config))
    elif args.command == "project-contacts":
        from .calibration import project_detection_contacts

        run = _read(args.predictions)
        record = next((r for r in run["images"] if r["image_id"] == args.image_id), None)
        if record is None:
            raise ValueError("unknown prediction image ID")
        save_json(args.output, project_detection_contacts(_read(args.calibration), record))
    elif args.command == "associate":
        from .integration import associate_observation

        _new_output(args.link_output)
        if Path(args.output).resolve() == Path(args.link_output).resolve():
            raise ValueError("room and link output paths must differ")
        updated, link = associate_observation(
            load_scene(args.room),
            _read(args.observation),
            args.entity_id,
            args.reviewer,
            args.evidence,
            args.demo_only,
            contact_estimate=_read(args.contact_estimate) if args.contact_estimate else None,
        )
        save_json(args.output, updated)
        save_json(args.link_output, link)
    elif args.command == "assess":
        from .clearance import assess_arrangement

        report = assess_arrangement(load_scene(args.room), _read(args.policy))
        save_json(args.output, report)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "geometrically_valid": report["geometrically_valid"],
                    "clearance_targets_met": report["clearance_targets_met"],
                    "warning_count": len(report["warnings"]),
                }
            )
        )
        return (
            2
            if not report["geometrically_valid"]
            else (0 if report["clearance_targets_met"] else 3)
        )
    print(args.output)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in (
            "room-from-dimensions",
            "calibrate-floor",
            "project-contacts",
            "associate",
            "assess",
        ):
            return _extended_command(args)
        if args.command == "demo":
            from .demo import run_demo

            report = run_demo(args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "replacement_fits": report["accepted_replacement"]["accepted"],
                    }
                )
            )
            return 0
        from .model import load_scene

        scene = load_scene(args.room)
        if args.command == "check":
            from .collision import check_scene

            report = check_scene(scene)
            print(json.dumps(report, indent=2))
            return 0 if report["fits"] else 2
        from .model import propose_replacement, save_json

        if Path(args.output).exists():
            raise ValueError("output already exists; choose a new JSON path")
        replacement = json.loads(Path(args.replacement).read_text(encoding="utf-8"))
        updated, report = propose_replacement(scene, args.remove_id, replacement)
        if report["accepted"]:
            save_json(args.output, updated)
        print(json.dumps(report, indent=2))
        return 0 if report["accepted"] else 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
