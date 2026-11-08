"""Bounded resolution/cadence/feature comparison, preserving geometric thresholds."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from furniture_bench.data import digest, read_json, write_json
from room_multiview.arkitscenes import orient_calibration, trajectory_references
from room_multiview.features import extract_features, match_features, opencv
from room_multiview.geometry import Policy, estimate_pair, rotation_error


def alternate_features(image, kind):
    cv = opencv()
    if kind == "sift":
        return extract_features(image)
    gray = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)
    detector = cv.SIFT_create(nfeatures=6000, contrastThreshold=0.01, edgeThreshold=15) if kind == "rootsift" else cv.AKAZE_create(threshold=0.0005)
    keys, descriptors = detector.detectAndCompute(gray, None)
    width = 128 if kind == "rootsift" else 61
    if descriptors is None:
        descriptors = np.empty((0, width), np.float32 if kind == "rootsift" else np.uint8)
    if kind == "rootsift" and len(descriptors):
        descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    return {"pixels": np.array([k.pt for k in keys], float).reshape(-1, 2), "descriptors": descriptors}


def mutual_matches(first, second, kind):
    if kind != "akaze":
        return match_features(first, second, 0.75 if kind == "sift" else 0.8)
    cv = opencv()
    a, b = first["descriptors"], second["descriptors"]
    if min(len(a), len(b)) < 2:
        return np.empty((0, 2), int)
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    def accepted(x, y):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(x, y, k=2) if len(pair) == 2 for m, n in [pair] if m.distance < 0.8 * n.distance}
    f, r = accepted(a, b), accepted(b, a)
    return np.array([(i, j) for i, j in f.items() if r.get(j) == i], int).reshape(-1, 2)


def pose_reference(row):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(row["quaternion_xyzw"]).as_matrix()
    result[:3, 3] = row["position_metres"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "vga", "assets", "output"):
        parser.add_argument("--" + key, required=True)
    a = parser.parse_args()
    data, vga, assets, output = map(Path, (a.data, a.vga, a.assets, a.output))
    if output.exists():
        raise ValueError("choose a new comparison output")
    output.mkdir(parents=True)
    receipt = read_json(vga / "receipt.json")
    rows = {r["source_rgb_index"]: r for r in read_json(data / "tracking-usable.json")["images"]}
    records = {r["source_rgb_index"]: r for r in receipt["records"]}
    sources = {}
    quarter_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    for index, record in records.items():
        row = rows[index]
        native = Image.open(data / row["file"]).convert("RGB")
        if digest(vga / record["file"]) != record["sha256"]:
            raise ValueError("VGA source changed")
        w, h, fx, fy, cx, cy = np.fromstring((vga / record["intrinsics_file"]).read_text(), sep=" ")
        k, _, _, turns = orient_calibration([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], (int(w), int(h)), "Left")
        high = Image.fromarray(np.rot90(np.asarray(Image.open(vga / record["file"]).convert("RGB")), turns))
        scale = np.diag([192 / high.width, 256 / high.height, 1.0])
        for mode, image, cal, timestamp in (
            ("lowres", native, row["calibration"], row["timestamp"]),
            ("vga", high, {"intrinsics": k.tolist(), "distortion": [0] * 5, "status": "dataset_calibrated", "source": "official VGA pincam with upright image rotation"}, record["vga_timestamp"]),
            ("vga-downsampled", high.resize((192, 256), Image.Resampling.LANCZOS), {"intrinsics": (scale @ k).tolist(), "distortion": [0] * 5, "status": "dataset_calibrated", "source": "VGA calibration scaled with downsampled image"}, record["vga_timestamp"]),
        ):
            sources[mode, index] = {"image": image, "cal": cal, "timestamp": timestamp}
    # References are read only for evaluation; never supplied to extraction or geometry.
    reference_rows = trajectory_references(assets / "lowres_wide.traj", [(f"{mode}:{index}", s["timestamp"]) for (mode, index), s in sources.items()], quarter_rotation)
    references = {r["image_id"]: pose_reference(r) for r in reference_rows}
    start = perf_counter()
    results = []
    configurations = ("sift", "rootsift", "akaze")
    for mode in ("lowres", "vga-downsampled", "vga"):
        for kind in configurations:
            features, feature_seconds = {}, 0.0
            for index in records:
                before = perf_counter()
                features[index] = alternate_features(sources[mode, index]["image"], kind)
                feature_seconds += perf_counter() - before
            for gap in receipt["gaps_raw_indices"]:
                pair_rows = []
                for anchor in receipt["anchors"]:
                    b = anchor + gap
                    left, right = sources[mode, anchor], sources[mode, b]
                    before = perf_counter()
                    matches = mutual_matches(features[anchor], features[b], kind)
                    geometry = estimate_pair(features[anchor]["pixels"][matches[:, 0]], features[b]["pixels"][matches[:, 1]], left["cal"], right["cal"], left["image"].size, right["image"].size, Policy())
                    geometry.pop("points", None)
                    row = {"raw_indices": [anchor, b], "timestamps": [left["timestamp"], right["timestamp"]], "features": [len(features[anchor]["pixels"]), len(features[b]["pixels"])], "matches": len(matches), "geometry": geometry, "matching_geometry_seconds": perf_counter() - before, "reference_audit": None}
                    if geometry["status"] == "accepted" and f"{mode}:{anchor}" in references and f"{mode}:{b}" in references:
                        expected = np.linalg.inv(references[f"{mode}:{b}"]) @ references[f"{mode}:{anchor}"]
                        estimate = np.array(geometry["camera_b_from_a"])
                        baseline = float(np.linalg.norm(expected[:3, 3]))
                        direction_error = float(np.degrees(np.arccos(np.clip(np.dot(estimate[:3, 3], expected[:3, 3]) / max(np.linalg.norm(estimate[:3, 3]) * baseline, 1e-12), -1, 1)))) if baseline >= 0.02 else None
                        angle_error = rotation_error(estimate[:3, :3], expected[:3, :3])
                        row["reference_audit"] = {"reference_baseline_metres": baseline, "rotation_error_degrees": angle_error, "translation_direction_error_degrees": direction_error, "metric_translation_error": None, "passes_audit": angle_error <= 3 and direction_error is not None and direction_error <= 15 and not geometry["planar_or_rotation_risk"], "scale": "unknown", "reference_kind": "sensor_correlated_ARKit_trajectory"}
                    pair_rows.append(row)
                results.append({"resolution": mode, "configuration": kind, "gap_raw_frames": gap, "gap_seconds_approximately": gap / 60, "pairs": pair_rows, "feature_extraction_seconds_all_32_views": feature_seconds, "geometrically_accepted": sum(r["geometry"]["status"] == "accepted" for r in pair_rows), "reference_audit_passed": sum(bool(r["reference_audit"] and r["reference_audit"]["passes_audit"]) for r in pair_rows), "rejections": dict(Counter(r["geometry"]["reason"] for r in pair_rows if r["geometry"]["status"] != "accepted"))})
    report = {"geometry_policy_unchanged": Policy().to_dict(), "audit_policy": {"max_rotation_degrees": 3, "max_translation_direction_degrees": 15, "minimum_reference_baseline_metres": 0.02, "reject_planar_or_rotation_risk": True}, "source_receipt_sha256": digest(vga / "receipt.json"), "anchors": receipt["anchors"], "configurations": results, "seconds": perf_counter() - start, "registration_claim": "Two-view experiments only; no connected room, metric dimensions or accuracy improvement inferred from acceptance counts", "reference_poses_used_for_inference": False, "selection_uses_reference_poses": False, "heldout_depth_used": False}
    write_json(output / "comparison.json", report)
    print([{k: r[k] for k in ("resolution", "configuration", "gap_raw_frames", "geometrically_accepted", "reference_audit_passed")} for r in results])


if __name__ == "__main__":
    main()
