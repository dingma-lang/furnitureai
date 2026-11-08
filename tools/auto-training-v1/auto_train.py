#!/usr/bin/env python3
"""Bounded, resumable TRAIN-ONLY furniture classifier search.

Does not load validation/test features, touch original model/checkpoints or invoke
Phase 2B evaluation. Internal product/family-group folds are model-selection data.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PHASE2A = PROJECT / "data/abo/experiments/phase2a-v1"
DEFAULT_OUTPUT = PROJECT / "data/abo/experiments/auto-training-v1"
CLASSES = ("sofa", "chair", "table", "bed", "desk", "cabinet", "lamp")
SEED = 20261104
FOLDS = 3
MAX_EPOCHS = 65
PATIENCE = 12

CONFIGS = [
    {"name": "linear-lr1e3-wd1e4", "kind": "linear", "lr": 0.001, "wd": 0.0001},
    {"name": "linear-lr3e4-wd1e4", "kind": "linear", "lr": 0.0003, "wd": 0.0001},
    {"name": "linear-lr1e3-wd1e2", "kind": "linear", "lr": 0.001, "wd": 0.01},
    {"name": "linear-lr3e3-wd1e3", "kind": "linear", "lr": 0.003, "wd": 0.001},
    {"name": "mlp64-d20-lr1e3", "kind": "mlp", "hidden": 64, "dropout": 0.2, "lr": 0.001, "wd": 0.0001},
    {"name": "mlp128-d30-lr5e4", "kind": "mlp", "hidden": 128, "dropout": 0.3, "lr": 0.0005, "wd": 0.001},
    {"name": "mlp128-d40-lr1e3", "kind": "mlp", "hidden": 128, "dropout": 0.4, "lr": 0.001, "wd": 0.01},
    {"name": "mlp64-d40-lr3e4", "kind": "mlp", "hidden": 64, "dropout": 0.4, "lr": 0.0003, "wd": 0.001},
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_original_module():
    script = PHASE2A / "experiment.py"
    if not script.is_file():
        raise FileNotFoundError(f"Original experiment script not found: {script}")
    spec = importlib.util.spec_from_file_location("original_phase2a_experiment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_group_token(value):
    if value is None or isinstance(value, (list, dict)):
        return None
    s = str(value).strip()
    return s if s and s.lower() not in {"none", "null", "unknown", "unresolved"} else None


def grouping_tokens(row):
    fields = ("abo_product_id", "group_id", "family_group_id", "human_audited_family_group_id", "capture_cluster_id", "decoded_pixel_sha256")
    return {(key, token) for key in fields if (token := valid_group_token(row.get(key))) is not None}


def check_cross_partition_group_overlap(training, all_approved):
    """Fail on *known* exact tokens shared across train vs held out.

    This cannot certify that unresolved physical product aliases are different.
    """
    train = collections.defaultdict(set)
    held = collections.defaultdict(set)
    for r in training:
        for token in grouping_tokens(r):
            train[token].add(r["image_id"])
    for r in all_approved:
        if r.get("proposed_split") in {"validation", "test"}:
            for token in grouping_tokens(r):
                held[token].add(r["image_id"])
    overlap = sorted(set(train) & set(held))
    if overlap:
        raise ValueError(f"Known train/holdout identity overlap (first 10): {overlap[:10]}")


def union_find_groups(rows):
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        x, y = find(a), find(b)
        if x != y:
            parent[max(x, y)] = min(x, y)

    seen = {}
    for i, row in enumerate(rows):
        for key in grouping_tokens(row):
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    grouped = collections.defaultdict(list)
    for i in range(len(rows)):
        grouped[find(i)].append(i)
    return sorted(grouped.values(), key=lambda indices: min(rows[i]["image_id"] for i in indices))


def create_folds(rows, folds=FOLDS):
    """Stratify connected identity groups (not individual images) across folds."""
    if not rows:
        raise ValueError("Empty training data")
    if any(r.get("proposed_split") != "train" or not r.get("weak_train_eligible") for r in rows):
        raise ValueError("Non-approved TRAIN row in candidate training features")
    products = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("furniture_class") not in CLASSES:
            raise ValueError(f"Unknown class {r.get('furniture_class')}")
        products[r["abo_product_id"]].append(i)
    for pid, indices in products.items():
        if len({rows[i]["furniture_class"] for i in indices}) != 1:
            raise ValueError(f"Conflicting categories in product {pid}")
    connected = union_find_groups(rows)
    by_class = collections.defaultdict(list)
    for indices in connected:
        cats = {rows[i]["furniture_class"] for i in indices}
        if len(cats) != 1:
            raise ValueError(f"Conflicting classes in linked identity group: {cats}")
        by_class[next(iter(cats))].append(indices)
    if set(by_class) != set(CLASSES):
        raise ValueError(f"Missing class groups: {set(CLASSES) - set(by_class)}")
    if any(len(groups) < folds for groups in by_class.values()):
        raise ValueError("Insufficient independent groups per class for requested fold count")
    assignments = [[] for _ in range(folds)]
    for cls in CLASSES:
        groups = by_class[cls]
        # Stable hash order, independent of CSV row order.
        groups.sort(key=lambda indices: hashlib.sha256(
            (f"{SEED}|{cls}|" + "|".join(sorted(rows[i]["image_id"] for i in indices))).encode()
        ).hexdigest())
        for j, group in enumerate(groups):
            assignments[j % folds].extend(group)
    fold_sets = [set(a) for a in assignments]
    if set.union(*fold_sets) != set(range(len(rows))):
        raise AssertionError("Incomplete fold coverage")
    for i in range(folds):
        for j in range(i + 1, folds):
            if fold_sets[i] & fold_sets[j]:
                raise AssertionError("Fold overlap")
    # Each product/family/capture group remains wholly in one fold.
    for linked in connected:
        if len({j for j, fold_set in enumerate(fold_sets) if any(i in fold_set for i in linked)}) != 1:
            raise AssertionError("Identity group leakage")
    for i in range(folds):
        remaining = set(range(len(rows))) - fold_sets[i]
        for cls in CLASSES:
            if not any(rows[j]["furniture_class"] == cls for j in remaining):
                raise ValueError(f"Fold {i} training data missing class {cls}")
    return [sorted(a) for a in assignments], connected


def load_verified_training():
    exp = load_original_module()
    exp.protected()
    arr, rows, meta = exp.load_training_features(exp.ROOT / "features-v1")
    if arr.shape != (275, 768) or len({r["abo_product_id"] for r in rows}) != 130:
        raise ValueError("Expected 275×768 features and 130 train-only products")
    check_cross_partition_group_overlap(rows, exp.approved_rows())
    folds, connected = create_folds(rows)
    return np.asarray(arr, dtype=np.float32), rows, meta, folds, connected


def features_fingerprint(meta):
    return meta["feature_sha256"]


def label_metrics(truth, pred):
    if not len(truth) or len(truth) != len(pred):
        raise ValueError("Invalid evaluation lengths")
    accur = float(np.mean(np.asarray(truth) == np.asarray(pred)))
    recalls, f1s, counts = [], [], []
    for c in range(len(CLASSES)):
        tp = sum(t == c and p == c for t, p in zip(truth, pred))
        fn = sum(t == c and p != c for t, p in zip(truth, pred))
        fp = sum(t != c and p == c for t, p in zip(truth, pred))
        counts.append(tp + fn)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {"accuracy": accur, "macro_f1": statistics.mean(f1s),
            "balanced_accuracy": statistics.mean(recalls), "per_class_counts": dict(zip(CLASSES, counts))}


def score_logits(rows, indices, logits):
    truth_img = [CLASSES.index(rows[i]["furniture_class"]) for i in indices]
    img_preds = np.argmax(logits, axis=1).tolist()
    product = collections.defaultdict(list)
    for j, i in enumerate(indices):
        product[rows[i]["abo_product_id"]].append(j)
    truth_p, pred_p = [], []
    for pid, pos in sorted(product.items()):
        tr = {truth_img[j] for j in pos}
        if len(tr) != 1:
            raise ValueError(f"Conflicting labels in {pid}")
        truth_p.append(next(iter(tr)))
        pred_p.append(int(logits[pos].mean(axis=0).argmax()))
    return {"product": label_metrics(truth_p, pred_p), "image": label_metrics(truth_img, img_preds),
            "product_count": len(product), "image_count": len(indices)}


def network(cfg, dimension):
    if cfg["kind"] == "linear":
        return torch.nn.Linear(dimension, len(CLASSES))
    if cfg["kind"] == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(dimension, cfg["hidden"]),
                                   torch.nn.ReLU(), torch.nn.Dropout(cfg["dropout"]),
                                   torch.nn.Linear(cfg["hidden"], len(CLASSES)))
    raise ValueError("Unsupported architecture")


def weights_by_products(rows, training_indices, device):
    by_class = collections.defaultdict(set)
    for idx in training_indices:
        r = rows[idx]
        by_class[r["furniture_class"]].add(r["abo_product_id"])
    counts = [len(by_class[c]) for c in CLASSES]
    if min(counts) == 0:
        raise ValueError(f"Development training split lacks class {counts}")
    raw = np.array([1 / math.sqrt(n) for n in counts], dtype=np.float64)
    raw /= np.mean(raw)
    return torch.tensor(raw, dtype=torch.float32, device=device), dict(zip(CLASSES, counts))


def epoch_sample(rows, indices, seed):
    """Choose one train view per product, but rotate across views over epochs."""
    by_pid = collections.defaultdict(list)
    for i in indices:
        by_pid[rows[i]["abo_product_id"]].append(i)
    rng = random.Random(seed)
    chosen = [rng.choice(sorted(v)) for pid, v in sorted(by_pid.items())]
    rng.shuffle(chosen)
    return chosen


def run_fold(array, rows, fold_idx, folds, cfg, parent, signature, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: no CPU fallback")
    fold_dir = parent / f"fold-{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("signature") != signature:
            raise ValueError("Existing fold result bound to different dataset/config")
        return result
    all_indices = set(range(len(rows)))
    val_ids = sorted(folds[fold_idx])
    train_ids = sorted(all_indices - set(val_ids))
    if not train_ids or not val_ids:
        raise ValueError("Empty train or development fold")
    if {rows[i]["abo_product_id"] for i in train_ids} & {rows[i]["abo_product_id"] for i in val_ids}:
        raise ValueError("Product leakage within fold")
    seed = SEED + fold_idx * 101 + int(hashlib.sha256(cfg["name"].encode()).hexdigest()[:8], 16) % 10000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = network(cfg, array.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    weights, counts = weights_by_products(rows, train_ids, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    x = torch.from_numpy(array.copy()).to(device)
    y = torch.tensor([CLASSES.index(r["furniture_class"]) for r in rows], dtype=torch.long, device=device)
    x.requires_grad_(False)
    progress_path = fold_dir / "progress.pt"
    start_epoch = 0
    best = {"epoch": 0, "macro_f1": -1.0, "balanced_accuracy": -1.0, "loss": float("inf")}
    best_state = None
    stale = 0
    curve = []
    if progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=True)
        if progress.get("signature") != signature:
            raise ValueError("Progress checkpoint does not match current dataset or configuration")
        model.load_state_dict(progress["model"])
        opt.load_state_dict(progress["optimizer"])
        start_epoch = int(progress["next_epoch"])
        best = progress["best"]
        best_state = progress["best_state"]
        stale = int(progress["stale"])
        curve = progress["curve"]
        print(f"Resuming {cfg['name']} fold {fold_idx+1} at epoch {start_epoch+1}", flush=True)
    started = time.monotonic()
    if stale >= patience:
        print(f"Prior run had already reached early stopping: {cfg['name']} fold {fold_idx+1}", flush=True)
    for epoch in range(start_epoch, max_epochs) if stale < patience else ():
        # Makes per-epoch dropout and one-view sampling deterministic across restarts.
        torch.manual_seed(seed + epoch * 7919)
        torch.cuda.manual_seed_all(seed + epoch * 7919)
        model.train()
        chosen = epoch_sample(rows, train_ids, seed + epoch * 7919)
        batch_losses = []
        for pos in range(0, len(chosen), 32):
            b = chosen[pos:pos+32]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x[b]), y[b])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Invalid gradients")
            opt.step()
            batch_losses.append((float(loss.detach()), len(b)))
        model.eval()
        with torch.inference_mode():
            logits_t = model(x[val_ids])
            val_loss = float(criterion(logits_t, y[val_ids]).detach())
            logits = logits_t.detach().cpu().numpy()
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite development loss")
        metrics = score_logits(rows, val_ids, logits)
        p = metrics["product"]
        key = (p["macro_f1"], p["balanced_accuracy"], -val_loss)
        oldkey = (best["macro_f1"], best["balanced_accuracy"], -best["loss"])
        improved = key > oldkey
        if improved:
            best = {"epoch": epoch + 1, "macro_f1": p["macro_f1"],
                    "balanced_accuracy": p["balanced_accuracy"], "loss": val_loss,
                    "metrics": metrics}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        row = {"epoch": epoch + 1, "train_loss": sum(v*n for v,n in batch_losses) / sum(n for _,n in batch_losses),
               "dev_loss": val_loss, "dev_product_macro_f1": p["macro_f1"],
               "dev_product_balanced_accuracy": p["balanced_accuracy"]}
        curve.append(row)
        # Atomic per-epoch restart point. No reusing this if code or cache changes.
        atomic_torch_save(progress_path, {"signature": signature, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "next_epoch": epoch + 1,
            "best": best, "best_state": best_state, "stale": stale, "curve": curve})
        if (epoch+1) % 5 == 0 or improved:
            print(f"{cfg['name']} fold={fold_idx+1}/{len(folds)} epoch={epoch+1} "
                  f"train_loss={row['train_loss']:.3f} dev_macroF1={p['macro_f1']:.3f} "
                  f"best={best['macro_f1']:.3f} stale={stale}", flush=True)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Missing best checkpoint")
    atomic_torch_save(fold_dir / "best-head.pt", {"state_dict": best_state, "config": cfg,
                       "signature": signature, "classes": CLASSES, "best_epoch": best["epoch"]})
    result = {"signature": signature, "config": cfg, "fold": fold_idx+1,
              "best_epoch": best["epoch"], "epochs_run": len(curve),
              "dev_product": best["metrics"]["product"], "dev_image": best["metrics"]["image"],
              "dev_product_count": best["metrics"]["product_count"],
              "dev_image_count": best["metrics"]["image_count"],
              "class_training_product_counts": counts, "seconds_this_session": time.monotonic()-started,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "metrics_scope": "train-derived internal development only; not independent holdout"}
    write_json(result_path, result)
    write_json(fold_dir / "curve.json", curve)
    progress_path.unlink(missing_ok=True)
    return result


def config_fingerprint(meta, rows, folds, cfg):
    payload = {"feature_sha256": features_fingerprint(meta),
               "fold_image_ids": [[rows[i]["image_id"] for i in fold] for fold in folds],
               "config": cfg, "source_code_sha256": digest(Path(__file__)),
               "seed": SEED, "classes": CLASSES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def summary_from_results(results):
    return {"mean_dev_product_macro_f1": statistics.mean(r["dev_product"]["macro_f1"] for r in results),
            "mean_dev_product_balanced_accuracy": statistics.mean(r["dev_product"]["balanced_accuracy"] for r in results),
            "mean_dev_product_accuracy": statistics.mean(r["dev_product"]["accuracy"] for r in results),
            "fold_macro_f1": [r["dev_product"]["macro_f1"] for r in results],
            "fold_best_epochs": [r["best_epoch"] for r in results],
            "n_folds": len(results)}


def create_leaderboard(out_dir):
    lines = []
    for c in CONFIGS:
        cfg_dir = out_dir / "experiments" / c["name"]
        p = cfg_dir / "result.json"
        failure = cfg_dir / "failure.json"
        if p.exists():
            r = read_json(p)
            lines.append({"experiment": c["name"], "status": "complete",
                          "dev_macro_f1": r["mean_dev_product_macro_f1"],
                          "dev_balanced_accuracy": r["mean_dev_product_balanced_accuracy"],
                          "n_folds": r["n_folds"]})
        elif failure.exists():
            lines.append({"experiment": c["name"], "status": "failed",
                          "dev_macro_f1": "", "dev_balanced_accuracy": "", "n_folds": ""})
        else:
            lines.append({"experiment": c["name"], "status": "pending",
                          "dev_macro_f1": "", "dev_balanced_accuracy": "", "n_folds": ""})
    lines.sort(key=lambda r: (r["status"] != "complete", -(r["dev_macro_f1"] if isinstance(r["dev_macro_f1"], float) else -1), r["experiment"]))
    path = out_dir / "leaderboard.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "status", "dev_macro_f1", "dev_balanced_accuracy", "n_folds"])
        writer.writeheader(); writer.writerows(lines)
    os.replace(tmp, path)
    return lines


def final_fit(array, rows, cfg, epochs, output, signature):
    """Train on TRAIN partition after selecting config; never use heldout images."""
    output_path = output / "final-train-only-head.pt"
    if output_path.exists():
        info_path = output / "final-metadata.json"
        if not info_path.exists():
            # An interruption between the atomic checkpoint and metadata commit is rare;
            # stop explicitly for provenance recovery instead of silently overwriting.
            raise RuntimeError("Final checkpoint exists without metadata; inspect and recover before restarting")
        if read_json(info_path).get("signature") != signature:
            raise ValueError("Different run already has a final checkpoint")
        return
    device = torch.device("cuda:0")
    seed = SEED + 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = network(cfg, array.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ids = list(range(len(rows)))
    weights, counts = weights_by_products(rows, ids, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    x = torch.from_numpy(array.copy()).to(device)
    y = torch.tensor([CLASSES.index(r["furniture_class"]) for r in rows], dtype=torch.long, device=device)
    curve = []
    for ep in range(epochs):
        model.train()
        torch.manual_seed(seed + ep * 7919)
        chosen = epoch_sample(rows, ids, seed + ep * 7919)
        losses = []
        for pos in range(0, len(chosen), 32):
            batch = chosen[pos:pos+32]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]), y[batch]); loss.backward()
            if not torch.isfinite(loss): raise ValueError("Nonfinite final fit loss")
            opt.step(); losses.append(float(loss.detach()))
        curve.append({"epoch": ep+1, "training_loss": statistics.mean(losses)})
    atomic_torch_save(output_path, {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "config": cfg, "classes": CLASSES, "feature_sha256": signature["feature_sha256"],
                        "source_code_sha256": signature["code_sha256"],
                        "seed": seed, "train_epochs": epochs,
                        "model_processor_fingerprint": signature["model_processor_fingerprint"]})
    write_json(output / "final-metadata.json", {"signature": signature,
        "checkpoint_sha256": digest(output_path), "selected_config": cfg,
        "train_only": True, "training_product_counts": counts,
        "selection_metric": "train-derived 3-fold development product macro-F1; model selection biased",
        "evaluation_note": "UNTESTED: do not claim holdout accuracy; no validation/test features read",
        "final_training_curve": curve})


def signature_info(meta):
    return {"feature_sha256": meta["feature_sha256"],
            "model_processor_fingerprint": meta["model_processor_fingerprint"],
            "code_sha256": digest(Path(__file__))}


def preflight_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; intentionally refusing CPU fallback")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    # This small-head pipeline uses <0.2 GiB, but avoid monopolising all GPU memory.
    torch.cuda.set_per_process_memory_fraction(0.65, device=0)
    torch.cuda.reset_peak_memory_stats(0)


def run_smoke(out_dir, array, rows, meta, folds):
    preflight_cuda()
    smoke = out_dir / "smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    cfg = CONFIGS[0]
    signature = config_fingerprint(meta, rows, folds, cfg) + "-smoke-v1"
    result = run_fold(array, rows, 0, folds, cfg, smoke, signature, max_epochs=3, patience=3)
    gate = {"passed": True, "feature_sha256": features_fingerprint(meta),
            "code_sha256": digest(Path(__file__)), "result": result,
            "scope": "three-epoch TRAIN-derived internal development smoke test"}
    write_json(out_dir / "smoke-gate.json", gate)
    print("SMOKE OK. Long run is NOT started. Output:", out_dir / "smoke-gate.json", flush=True)


def run_all(out_dir, array, rows, meta, folds):
    gatepath = out_dir / "smoke-gate.json"
    if not gatepath.exists():
        raise RuntimeError("Smoke gate missing. Run `smoke` first.")
    gate = read_json(gatepath)
    if gate.get("passed") is not True or gate["feature_sha256"] != features_fingerprint(meta) or gate["code_sha256"] != digest(Path(__file__)):
        raise RuntimeError("Smoke gate stale or bound to another code/dataset")
    with (out_dir / "training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training agent is active")
        preflight_cuda()
        write_json(out_dir / "run-meta.json", {"signature": signature_info(meta),
            "configs": CONFIGS, "fold_count": len(folds), "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "scope": "TRAIN partition only; locked holdout not used",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        for cfg in CONFIGS:
            if (out_dir / "stop-after-current.flag").exists():
                print("Stop-after-current requested. Exiting before next configuration.", flush=True)
                break
            cfgdir = out_dir / "experiments" / cfg["name"]
            cfgdir.mkdir(parents=True, exist_ok=True)
            signature = config_fingerprint(meta, rows, folds, cfg)
            if (cfgdir / "result.json").exists():
                existing = read_json(cfgdir / "result.json")
                if existing.get("signature") != signature:
                    raise ValueError(f"Config result has stale signature: {cfg['name']}")
                print(f"Skipping finished {cfg['name']}", flush=True)
                continue
            if (cfgdir / "failure.json").exists():
                print(f"Skipping failed {cfg['name']} (requires deliberate repair)", flush=True)
                continue
            print("START", cfg["name"], flush=True)
            try:
                results = [run_fold(array, rows, i, folds, cfg, cfgdir, signature)
                           for i in range(len(folds))]
                summary = summary_from_results(results)
                summary.update({"signature": signature, "config": cfg,
                    "metrics_scope": "internal train-derived development; not independent validation/test"})
                write_json(cfgdir / "result.json", summary)
                print("COMPLETE", cfg["name"], summary["mean_dev_product_macro_f1"], flush=True)
            except Exception as exc:
                write_json(cfgdir / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(),
                           "signature": signature})
                print(f"FAILED {cfg['name']}: {exc}", file=sys.stderr, flush=True)
            create_leaderboard(out_dir)
        ranking = [r for r in create_leaderboard(out_dir) if r["status"] == "complete"]
        if not ranking:
            print("No successful configurations. No final checkpoint.", flush=True)
            return
        if (out_dir / "stop-after-current.flag").exists():
            print("Stopped by request; no final selection until next run.", flush=True)
            return
        if any(r["status"] != "complete" for r in create_leaderboard(out_dir)):
            print("Some configurations are pending or failed. No final checkpoint until resolved.", flush=True)
            return
        winner = ranking[0]
        cfg = next(c for c in CONFIGS if c["name"] == winner["experiment"])
        result = read_json(out_dir / "experiments" / cfg["name"] / "result.json")
        n_epochs = max(1, round(statistics.median(result["fold_best_epochs"])))
        print(f"Best development configuration: {cfg['name']} at {n_epochs} epochs (median CV optimum)", flush=True)
        final_fit(array, rows, cfg, n_epochs, out_dir, signature_info(meta))
        print("TRAIN-ONLY CHECKPOINT READY. No official holdout scoring performed.", flush=True)


def status(out_dir):
    print("Output:", out_dir)
    if not out_dir.exists():
        print("Not started")
        return
    for name in ("smoke-gate.json", "run-meta.json", "final-metadata.json"):
        print(name, "present" if (out_dir / name).exists() else "not present")
    if (out_dir / "leaderboard.csv").exists():
        print((out_dir / "leaderboard.csv").read_text(encoding="utf-8"))
    if (out_dir / "stop-after-current.flag").exists():
        print("STOP AFTER CURRENT FLAG PRESENT")


def describe(rows, folds, connected, meta):
    all_products = {r["abo_product_id"] for r in rows}
    print(json.dumps({"feature_shape": [len(rows), meta["dimension"]],
        "features_sha256": meta["feature_sha256"], "train_products": len(all_products),
        "distinct_connected_identity_groups": len(connected),
        "training_class_product_counts": dict(collections.Counter({p: next(r["furniture_class"] for r in rows if r["abo_product_id"] == p) for p in all_products}.values())),
        "internal_development_folds": [{"fold":i+1,"images":len(f),
            "products":len({rows[j]["abo_product_id"] for j in f})} for i,f in enumerate(folds)],
        "experiments": [c["name"] for c in CONFIGS],
        "official_holdout_used": False,
        "warning": "Physical identity aliases may remain unresolved. Weak labels are NOT human-verified. CV metrics are model-selection data, not final test accuracy."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "plan", "smoke", "run", "status", "stop-after-current", "clear-stop"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.action == "status":
        status(out); return
    if args.action == "stop-after-current":
        if not out.exists():
            raise RuntimeError("Training output directory not created yet")
        (out / "stop-after-current.flag").write_text("user requested\n")
        print("Stop requested; exits between configurations."); return
    if args.action == "clear-stop":
        (out / "stop-after-current.flag").unlink(missing_ok=True)
        print("Stop flag cleared. Use `run` to resume when ready."); return
    array, rows, meta, folds, connected = load_verified_training()
    describe(rows, folds, connected, meta)
    if args.action in {"check", "plan"}:
        if args.action == "check":
            print("CHECK OK: no training launched")
        return
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "smoke":
        run_smoke(out, array, rows, meta, folds)
    else:
        run_all(out, array, rows, meta, folds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; last complete epoch checkpoint preserved.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
