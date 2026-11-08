"""Explicit, pinned official checkpoints; never invoked by imports or tests."""

import argparse
import shutil
from pathlib import Path

MODELS = (
    (
        "openmmlab/upernet-convnext-tiny",
        "876ffc56b819a829448f7e81e9f8606deef6fb65",
        "segmentation-876ffc56",
        "MIT",
    ),
    (
        "depth-anything/Depth-Anything-V2-Small-hf",
        "5426e4f0f36572d16453bbda7a8389317b1bef99",
        "depth-5426e4f0",
        "Apache-2.0",
    ),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/models/room-layout")
    parser.add_argument(
        "--allow-download", action="store_true", help="authorize about 340 MB of weights"
    )
    args = parser.parse_args(argv)
    if not args.allow_download:
        parser.error(
            "no files downloaded; review reports/room-layout-experiment/MODEL_SELECTION.md, then pass --allow-download"
        )
    from huggingface_hub import snapshot_download
    from transformers import DepthAnythingForDepthEstimation, UperNetForSemanticSegmentation

    assert DepthAnythingForDepthEstimation and UperNetForSemanticSegmentation
    target = Path(args.output).resolve()
    existing_parent = target
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 750 * 1024**2:
        raise RuntimeError("need at least 750 MiB free for checkpoints and temporary files")
    print(
        "CPU-compatible architectures available. Run stages sequentially; measured process peaks were 2.62/1.54 GiB."
    )
    for repo, revision, folder, license_name in MODELS:
        print(f"{repo}@{revision}: declared checkpoint license {license_name}", flush=True)
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=target / folder,
            allow_patterns=[
                "README.md",
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
