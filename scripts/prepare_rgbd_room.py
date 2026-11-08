"""Prepare a small genuine TUM RGB-D subset with disjoint reconstruction/reference roles."""

import argparse
import tarfile
from pathlib import Path

import numpy as np

from furniture_bench.data import digest, write_json


def prepare(archive_path, output, indices):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new subset directory")
    with tarfile.open(archive_path) as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        rgb = sorted(name for name in members if "/rgb/" in name and name.endswith(".png"))
        depths = sorted(name for name in members if "/depth/" in name and name.endswith(".png"))
        if any(i < 0 or i >= len(rgb) for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("invalid/duplicate RGB indices")
        depth_times = np.array([float(Path(p).stem) for p in depths])
        gt_path = next(name for name in members if name.endswith("/groundtruth.txt"))
        gt_bytes = archive.extractfile(members[gt_path]).read()
        gt = np.array(
            [
                [float(x) for x in line.split()]
                for line in gt_bytes.decode().splitlines()
                if line and not line.startswith("#")
            ]
        )
        wanted = {rgb[i] for i in indices}
        for i in indices:
            timestamp = float(Path(rgb[i]).stem)
            wanted.add(depths[int(np.argmin(abs(depth_times - timestamp)))])
        payloads = {}
        with tarfile.open(archive_path, mode="r|gz") as stream:
            for member in stream:
                if member.name in wanted:
                    payloads[member.name] = stream.extractfile(member).read()
        output.mkdir(parents=True)
        (output / "groundtruth.txt").write_bytes(gt_bytes)
        rows, references = [], []
        rights = {
            "allowed_for_benchmark": True,
            "source": "TUM RGB-D / Sturm, Engelhard, Endres, Burgard, Cremers, IROS 2012",
            "license": "CC BY 4.0; https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license",
        }
        source = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
        for i in indices:
            member = rgb[i]
            timestamp = float(Path(member).stem)
            di = int(np.argmin(abs(depth_times - timestamp)))
            delta = float(abs(depth_times[di] - timestamp))
            if delta > 0.02:
                raise ValueError(f"no RGB-D timestamp association within 20ms at RGB index {i}")
            image_id = "tum-room-" + Path(member).stem
            filename = f"rgb-{i:04d}.png"
            depth_file = f"depth-{i:04d}.png"
            (output / filename).write_bytes(payloads[member])
            (output / depth_file).write_bytes(payloads[depths[di]])
            rows.append(
                {
                    "image_id": image_id,
                    "file": filename,
                    "image_sha256": digest(output / filename),
                    "timestamp": timestamp,
                    "source_rgb_index": i,
                    "objects": None,
                    "layout_annotations": None,
                    "rights": rights,
                    "calibration": {
                        "intrinsics": [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
                        "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
                        "status": "dataset_calibrated",
                        "source": source,
                    },
                    "sensor_depth": {
                        "file": depth_file,
                        "sha256": digest(output / depth_file),
                        "encoding": "uint16_depth_z",
                        "units_per_metre": 5000,
                        "rgb_depth_delta_seconds": delta,
                        "status": "sensor_measured",
                        "source": "TUM Kinect pre-registered depth; dataset's published default pinhole backprojection",
                        "calibration": {
                            "intrinsics": [[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]],
                            "distortion": [0] * 5,
                            "status": "estimated",
                            "source": source
                            + "; recommended default for registered depth, not precision RGB calibration",
                        },
                        "uncertainty": {
                            "depth_error_metres": None,
                            "rgb_depth_alignment_error_pixels": None,
                            "note": "Sensor noise, missing depth, timestamp motion and approximate registered-depth pinhole model; no validated error bound.",
                        },
                    },
                }
            )
            nearest = gt[np.argmin(abs(gt[:, 0] - timestamp))]
            references.append(
                {
                    "image_id": image_id,
                    "rgb_timestamp": timestamp,
                    "pose_timestamp": float(nearest[0]),
                    "timestamp_delta_seconds": float(abs(nearest[0] - timestamp)),
                    "position_metres": nearest[1:4].tolist(),
                    "quaternion_xyzw": nearest[4:].tolist(),
                }
            )
    room_id = Path(archive_path).stem.replace(".tar", "")
    # Every fourth sampled view is held out from all camera fitting and fusion.
    for name, subset in (
        ("all", rows),
        ("reconstruction", [r for j, r in enumerate(rows) if j % 4 != 3]),
        ("heldout", [r for j, r in enumerate(rows) if j % 4 == 3]),
    ):
        if subset:
            write_json(
                output / (name + ".json"),
                {"schema_version": 1, "room_id": room_id, "images": subset},
            )
    write_json(
        output / "reference-poses.json",
        {"source": "TUM motion capture; evaluation only", "poses": references},
    )
    write_json(
        output / "receipt.json",
        {
            "archive_sha256": digest(archive_path),
            "archive_bytes": Path(archive_path).stat().st_size,
            "selected_sorted_rgb_indices": indices,
            "total_rgb_frames": len(rgb),
            "rights": rights,
            "partition": "sample ordinal modulo 4 == 3 held out; no reference pose used for selection",
            "depth_mode": "genuine sensor depth for explicitly RGB-D experiments; forbidden in RGB-only inference",
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(prepare(args.archive, args.output, args.indices))


if __name__ == "__main__":
    main()
