"""Select fixed chronological TUM RGB frames; reference poses are evaluation-only."""

import argparse
import tarfile
from pathlib import Path

import numpy as np

from furniture_bench.data import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="data/multiview/tum-desk")
    parser.add_argument("--indices", type=int, nargs="+", default=[30, 60, 90])
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("subset output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    rights = {
        "allowed_for_benchmark": True,
        "source": "TUM RGB-D benchmark / Sturm et al., IROS 2012",
        "license": "CC BY 4.0; https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license",
    }
    with tarfile.open(args.archive) as archive:
        rgb = sorted(
            (m for m in archive.getmembers() if "/rgb/" in m.name and m.name.endswith(".png")),
            key=lambda m: m.name,
        )
        gt_member = next(m for m in archive.getmembers() if m.name.endswith("/groundtruth.txt"))
        gt_bytes = archive.extractfile(gt_member).read()
        (out / "groundtruth.txt").write_bytes(gt_bytes)
        gt = np.array(
            [
                [float(v) for v in line.split()]
                for line in gt_bytes.decode().splitlines()
                if line and not line.startswith("#")
            ]
        )
        images, reference = [], []
        # Predeclared sampling; selection does not use estimated pose or ground truth.
        if len(set(args.indices)) != len(args.indices) or any(
            i < 0 or i >= len(rgb) for i in args.indices
        ):
            raise ValueError("choose unique indices within the available RGB sequence")
        for index in args.indices:
            member = rgb[index]
            filename = Path(member.name).name
            timestamp = float(Path(filename).stem)
            data = archive.extractfile(member).read()
            (out / filename).write_bytes(data)
            image_id = "tum-desk-" + Path(filename).stem
            images.append(
                {
                    "image_id": image_id,
                    "file": filename,
                    "image_sha256": digest(out / filename),
                    "rights": rights,
                    "objects": None,
                    "layout_annotations": None,
                    "role": "multiview_experiment",
                    "timestamp": timestamp,
                    "calibration": {
                        "intrinsics": [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
                        "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
                        "status": "dataset_calibrated",
                        "source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#calibration_of_the_color_camera",
                    },
                }
            )
            nearest = gt[np.argmin(abs(gt[:, 0] - timestamp))]
            reference.append(
                {
                    "image_id": image_id,
                    "rgb_timestamp": timestamp,
                    "pose_timestamp": float(nearest[0]),
                    "timestamp_delta_seconds": float(abs(nearest[0] - timestamp)),
                    "position_metres": nearest[1:4].tolist(),
                    "quaternion_xyzw": nearest[4:8].tolist(),
                }
            )
    write_json(
        out / "rooms.json", {"schema_version": 1, "room_id": "tum-freiburg1-desk", "images": images}
    )
    write_json(
        out / "reference-poses.json",
        {
            "source": "TUM motion-capture groundtruth.txt",
            "used_for_reconstruction": False,
            "poses": reference,
        },
    )
    write_json(
        out / "receipt.json",
        {
            "archive": str(Path(args.archive).resolve()),
            "archive_sha256": digest(args.archive),
            "source_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
            "selected_sorted_rgb_indices": args.indices,
            "total_rgb_frames": len(rgb),
            "reference_depth_images_extracted": False,
            "rights": rights,
        },
    )
    print(out)


if __name__ == "__main__":
    main()
