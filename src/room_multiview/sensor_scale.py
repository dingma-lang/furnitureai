"""Calibrate an RGB map using genuine observed anchor depth, never evaluation poses."""

import copy

import numpy as np

from .rgbd import sample_depth


def scale_from_anchor_depth(project, depth, intrinsics, source, source_sha256):
    geometry = project["shared_geometry"]
    if geometry.get("metric_scale"):
        raise ValueError(
            "preserve existing confirmed scale; resolve explicitly before sensor calibration"
        )
    anchor = geometry["anchor_view_id"]
    camera = np.asarray(geometry["cameras"][anchor]["camera_from_world"])
    if not np.allclose(camera, np.eye(4)):
        raise ValueError("sensor-scale baseline requires the map anchor camera as identity")
    points = []
    pixels = []
    landmark_ids = []
    for landmark in geometry["landmarks"]:
        observation = next((o for o in landmark["observations"] if o["view_id"] == anchor), None)
        if observation and landmark["position"][2] > 0:
            points.append(landmark["position"])
            pixels.append(observation["pixel"])
            landmark_ids.append(landmark["id"])
    if len(points) < 20:
        raise ValueError("insufficient anchor landmarks for sensor scale")
    points = np.array(points)
    observed, valid = sample_depth(depth, pixels, intrinsics)
    ratios = observed[:, 2] / points[:, 2]
    good = valid & np.isfinite(ratios) & (ratios > 0)
    if good.sum() < 20:
        raise ValueError("insufficient valid measured depths for scale")
    median = float(np.median(ratios[good]))
    mad = float(np.median(abs(ratios[good] - median)))
    good &= abs(ratios - median) <= max(0.03 * median, 3 * 1.4826 * mad)
    if good.sum() < 20:
        raise ValueError("inconsistent sensor-to-map scale samples")
    factor = float(np.median(ratios[good]))
    relative_scatter = float(1.4826 * np.median(abs(ratios[good] - factor)) / factor)
    if relative_scatter > 0.1:
        raise ValueError("sensor-to-map scale dispersion too large")
    updated = copy.deepcopy(project)
    calibration = {
        "metres_per_unit": factor,
        "status": "sensor_assisted_estimated_scale",
        "source": source,
        "sensor_depth_sha256": source_sha256,
        "units": "m",
        "samples": int(good.sum()),
        "support_landmark_ids": [key for key, keep in zip(landmark_ids, good) if keep],
        "relative_robust_scatter": relative_scatter,
        "calibration_uncertainty": {
            "sensor_systematic_error_metres": None,
            "note": "Sample MAD is not a certified error bound; sensor and approximate calibration errors remain.",
        },
        "reference_poses_used": False,
        "reference_depth_kind": "reconstruction_input_sensor_measurement_not_heldout_evaluation",
        "physical_accuracy_verified": False,
    }
    updated["shared_geometry"]["metric_scale"] = calibration
    updated["coordinate_frames"]["transforms"]["world_to_metres"] = {
        "scale": factor,
        "rotation": np.eye(3).tolist(),
        "translation": [0, 0, 0],
        "source": calibration,
    }
    updated["coordinate_frames"]["physical_metres"]["available"] = True
    updated["physical_accuracy_verified"] = False
    updated["fit_readiness"]["ready"] = False
    return updated, calibration
