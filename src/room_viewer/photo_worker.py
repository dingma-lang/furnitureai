"""OWLv2 worker adapter, using the existing detector and experiment pipeline."""
import argparse
import faulthandler

from furniture_bench.data import read_json


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser()
    for key in ("manifest", "model", "prompts", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    from furniture_bench.detection.owlv2 import Owlv2Detector
    from furniture_bench.pipeline import run_experiment
    from furniture_bench.runtime import get_torch
    torch = get_torch()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    detector = Owlv2Detector(args.model, device="cpu", allow_download=False, threshold=.2, nms_iou=.5, prompt_bank=read_json(args.prompts))
    report = run_experiment(args.manifest, args.output, detector)
    if report["images"][0]["status"] == "error":
        raise RuntimeError("OWLv2 failed; inspect the saved worker report")


if __name__ == "__main__":
    main()
