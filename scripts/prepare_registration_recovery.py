"""Local archive reuse, excluding original Phase 5 held-out RGB and measured depth."""

import argparse
from pathlib import Path

from prepare_rgbd_room import prepare

from furniture_bench.data import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=".cache/datasets/rgbd_dataset_freiburg1_room.tgz")
    parser.add_argument("--heldout", default="data/phase5/tum-room-dense/heldout.json")
    parser.add_argument("--primary", default="data/phase5/tum-room-dense/reconstruction.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    heldout = read_json(args.heldout)["images"]
    excluded = {r["source_rgb_index"] for r in heldout}
    output = prepare(args.archive, args.output, [i for i in range(480, 601) if i not in excluded])
    doc = read_json(output / "all.json")
    # Depth frames can be shared by adjacent RGB timestamps. Exclude any such alias too.
    heldout_depth = {r["sensor_depth"]["sha256"] for r in heldout}
    doc["images"] = [r for r in doc["images"] if r["sensor_depth"]["sha256"] not in heldout_depth]
    for r in doc["images"]:
        r["role"] = "tracking_input_excludes_original_phase5_heldout"
    write_json(output / "eligible.json", doc)
    write_json(
        output / "experiment-protocol.json",
        {
            "original_heldout_manifest_sha256": digest(args.heldout),
            "excluded_original_heldout_rgb_indices": sorted(excluded),
            "eligible_tracking_frames": len(doc["images"]),
            "primary_comparison_ids": [r["image_id"] for r in read_json(args.primary)["images"]],
            "reference_poses_used_for_tracking": False,
            "partition": "eligible.json overrides generated modulo-four subsets and preserves the original Phase 5 six-view holdout; no new reference selection.",
        },
    )
    print(Path(output).resolve())


if __name__ == "__main__":
    main()
