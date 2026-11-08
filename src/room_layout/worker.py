"""One model per subprocess. Never imports/loads OWLv2 or native FAISS."""

from __future__ import annotations

import argparse
import faulthandler
import resource
import sys
from pathlib import Path

import numpy as np

from furniture_bench.data import digest, load_image, load_manifest, write_json
from furniture_bench.runtime import environment, get_torch

from .models import LocalModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("segmentation", "depth"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    rows = load_manifest(args.manifest, "rooms")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch = get_torch()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = LocalModel(args.model, args.stage)
    report = {"schema_version": 1, "stage": args.stage, "images": [], "environment": environment()}
    try:
        model.load()
        for index, row in enumerate(rows):
            record = {"image_id": row["image_id"], "status": "error"}
            try:
                image = load_image(row["resolved_file"])
                arrays, seconds = model.predict(image)
                for value in arrays.values():
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite model output")
                file = f"{index:03d}.npz"
                np.savez_compressed(output / file, **arrays)
                record.update(
                    status="ok",
                    file=file,
                    width=image.width,
                    height=image.height,
                    image_sha256=digest(row["resolved_file"]),
                    inference_seconds=seconds,
                )
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["images"].append(record)
            print(args.stage, row["image_id"], record["status"], flush=True)
    except Exception as exc:
        report["load_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["model"] = model.metadata()
        native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = (
            native_peak if sys.platform == "darwin" else native_peak * 1024
        )
        write_json(output / "report.json", report)
    return (
        2 if report.get("load_error") or any(r["status"] != "ok" for r in report["images"]) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
