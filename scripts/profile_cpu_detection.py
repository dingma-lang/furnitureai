"""Profile one CPU detection run using the existing detector and experiment pipeline.

No checkpoint downloads occur here. Supply an already downloaded local model.
Native crashes print faulthandler diagnostics; use /usr/bin/time for an outer record.
"""

from __future__ import annotations

import argparse
import faulthandler
import resource
import sys
import threading
from pathlib import Path
from time import perf_counter

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.detection.owlv2 import Owlv2Detector
from furniture_bench.pipeline import run_experiment
from furniture_bench.runtime import environment, get_torch


class MemorySampler:
    def __init__(self):
        try:
            import psutil
        except ImportError:
            self.process = None
        else:
            self.process = psutil.Process()
        self.peak = 0
        self.samples = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def rss(self):
        return self.process.memory_info().rss if self.process else None

    def mark(self, label):
        self.samples[label] = self.rss()

    def _sample(self):
        while not self.stop_event.wait(0.05):
            self.peak = max(self.peak, self.rss() or 0)

    def __enter__(self):
        self.mark("before_torch_import")
        self.peak = self.samples["before_torch_import"] or 0
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.mark("after_run")
        self.stop_event.set()
        self.thread.join()

    def report(self):
        native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "process_rss_samples_bytes": self.samples,
            "sampled_peak_process_rss_bytes": self.peak if self.process else None,
            "sampling_interval_seconds": 0.05 if self.process else None,
            "os_peak_process_rss_bytes": native_peak
            if sys.platform == "darwin"
            else native_peak * 1024,
            "scope": "whole process RSS; not model-only allocations; no GPU",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--output", required=True, help="new experiment directory")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--prompt-bank")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    faulthandler.enable(all_threads=True)
    started = perf_counter()
    profile = {
        "kind": "real_pretrained_cpu_inference",
        "device": "cpu",
        "errors": [],
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_path": str(Path(args.model_path).resolve()),
        "cpu_threads": args.threads,
    }
    exit_code = 0
    with MemorySampler() as memory:
        try:
            torch = get_torch()
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            memory.mark("after_torch_import")
            detector = Owlv2Detector(
                model_id=args.model_path,
                device="cpu",
                threshold=args.threshold,
                allow_download=False,
                prompt_bank=read_json(args.prompt_bank) if args.prompt_bank else None,
            )
            print("Loading local OWLv2 checkpoint on CPU", flush=True)
            detector.load()
            memory.mark("after_model_load")
            print("Running room detection", flush=True)
            run = run_experiment(args.manifest, args.output, detector)
            memory.mark("after_inference")
            profile["detector"] = detector.metadata()
            profile["images"] = run["images"]
            profile["status"] = "ok" if all(r["status"] == "ok" for r in run["images"]) else "error"
            if profile["status"] != "ok":
                exit_code = 2
            weights = Path(args.model_path) / "model.safetensors"
            profile["weights_sha256"] = digest(weights)
            print(f"Detection status: {profile['status']}", flush=True)
        except Exception as exc:
            profile["status"] = "error"
            profile["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            exit_code = 2
            raise
        finally:
            profile["wall_seconds_including_import_and_load"] = perf_counter() - started
            profile["environment"] = environment()
            profile["memory"] = memory.report()
            write_json(args.profile, profile)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
