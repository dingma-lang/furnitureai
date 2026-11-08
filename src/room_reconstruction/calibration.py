"""User-assisted floor homography: planar perspective, not verified reconstruction."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .model import number, validate_scene


def validate_dimensions(dimensions):
    for key in ("width", "depth", "height"):
        number(dimensions.get(key), key, positive=True)
    if dimensions.get("status") not in ("provided_unverified", "measured", "demo_specified"):
        raise ValueError(
            "dimension status must be provided_unverified, measured, or demo_specified"
        )
    if not isinstance(dimensions.get("source"), str) or not dimensions["source"].strip():
        raise ValueError("room dimensions require a source")
    if dimensions["status"] == "measured" and not dimensions.get("measurement_evidence"):
        raise ValueError("measured room dimensions require measurement evidence")


def room_from_dimensions(dimensions, room_id="provided-room", wall_thickness=0.15):
    validate_dimensions(dimensions)
    width, depth, height = [float(dimensions[k]) for k in ("width", "depth", "height")]
    number(wall_thickness, "wall_thickness", positive=True)
    half = wall_thickness / 2
    wall_endpoints = [
        ("south", [0, -half], [width, -half]),
        ("east", [width + half, 0], [width + half, depth]),
        ("north", [width, depth + half], [0, depth + half]),
        ("west", [-half, depth], [-half, 0]),
    ]
    return validate_scene(
        {
            "schema_version": 1,
            "room_id": room_id,
            "units": "m",
            "coordinate_system": "right-handed-z-up",
            "dimensions": [width, depth, height],
            "provenance": {
                "kind": "provided_dimensions",
                "dimensions": dict(dimensions),
                "derived_from_photo": False if dimensions["status"] == "demo_specified" else None,
                "correspondence_to_photo": "not_established",
                "physical_accuracy_verified": False,
            },
            "floor": {
                "id": "floor",
                "boundary": [[0, 0], [width, 0], [width, depth], [0, depth]],
                "z": 0,
                "thickness": 0.15,
            },
            "walls": [
                {
                    "id": name,
                    "start": start,
                    "end": end,
                    "height": height,
                    "thickness": wall_thickness,
                    "interior_side": "left",
                }
                for name, start, end in wall_endpoints
            ],
            "openings": [],
            "furniture": [],
        }
    )


def _points(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 4 or not np.isfinite(result).all():
        raise ValueError("at least four finite 2D point pairs are required")
    return result


def _normalize(points):
    centre = points.mean(axis=0)
    spread = np.sqrt(np.mean(np.sum((points - centre) ** 2, axis=1)))
    if spread < 1e-10:
        raise ValueError("control points have no spatial extent")
    scale = np.sqrt(2) / spread
    transform = np.array(
        [[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]]
    )
    normalized = np.column_stack((points, np.ones(len(points)))) @ transform.T
    return normalized[:, :2], transform


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("projection is on or too close to the perspective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def estimate_homography(floor_points, image_points):
    floor_points, image_points = _points(floor_points), _points(image_points)
    if len(floor_points) != len(image_points):
        raise ValueError("floor and image correspondence counts differ")
    world, tw = _normalize(floor_points)
    image, ti = _normalize(image_points)
    rows = []
    for (x, y), (u, v) in zip(world, image):
        rows.extend(
            [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
        )
    design = np.asarray(rows)
    if np.linalg.matrix_rank(design, tol=1e-9) < 8:
        raise ValueError("degenerate/collinear calibration correspondences")
    _, _, vh = np.linalg.svd(design, full_matrices=True)
    normalized_h = vh[-1].reshape(3, 3)
    if np.linalg.cond(normalized_h) > 1e10:
        raise ValueError("unstable floor calibration")
    matrix = np.linalg.inv(ti) @ normalized_h @ tw
    matrix /= matrix[2, 2] if abs(matrix[2, 2]) > 1e-10 else np.linalg.norm(matrix)
    return matrix


def calibrate_floor(config):
    if config.get("schema_version") != 1:
        raise ValueError("unsupported calibration schema")
    dimensions = config["dimensions"]
    validate_dimensions(dimensions)
    width, depth = dimensions["width"], dimensions["depth"]
    image = config["image"]
    for key in ("width", "height"):
        number(image.get(key), f"image.{key}", positive=True)
    if not isinstance(image.get("sha256"), str) or len(image["sha256"]) != 64:
        raise ValueError("calibration must bind an exact source-image SHA-256")
    if "floor_corners_pixels" in config:
        if "correspondences" in config:
            raise ValueError("supply floor corners or general correspondences, not both")
        pixels = _points(config["floor_corners_pixels"])
        if len(pixels) != 4:
            raise ValueError("exactly four ordered room-floor corners required")
        world = np.array([[0, 0], [width, 0], [width, depth], [0, depth]], dtype=float)
    else:
        world = _points([p["floor_metres"] for p in config["correspondences"]])
        pixels = _points([p["image_pixels"] for p in config["correspondences"]])
    if (
        np.any(pixels < 0)
        or np.any(pixels[:, 0] > image["width"])
        or np.any(pixels[:, 1] > image["height"])
    ):
        raise ValueError("control point outside source image")
    if np.any(world < 0) or np.any(world[:, 0] > width) or np.any(world[:, 1] > depth):
        raise ValueError("floor control point outside the declared room")
    pixel_uncertainty = number(
        config.get("pixel_uncertainty", 5), "pixel_uncertainty", positive=True
    )
    matrix = estimate_homography(world, pixels)
    inverse = np.linalg.inv(matrix)
    corners = np.array([[0, 0], [width, 0], [width, depth], [0, depth]])
    denominators = np.column_stack((corners, np.ones(4))) @ matrix[2]
    if np.any(np.abs(denominators) < 1e-9) or np.any(
        np.sign(denominators) != np.sign(denominators[0])
    ):
        raise ValueError("perspective horizon crosses the declared room floor")
    residual = transform_points(matrix, world) - pixels
    horizon = inverse.T @ np.array([0.0, 0.0, 1.0])
    line_norm = np.linalg.norm(horizon[:2])
    horizon /= line_norm if line_norm > 1e-12 else np.linalg.norm(horizon)

    def vanishing(direction):
        point = matrix @ np.array(direction)
        return {
            "homogeneous": point.tolist(),
            "image_pixels": (point[:2] / point[2]).tolist() if abs(point[2]) > 1e-9 else None,
            "at_infinity": bool(abs(point[2]) <= 1e-9),
        }

    canonical = json.dumps(config, sort_keys=True, allow_nan=False)
    return {
        "schema_version": 1,
        "kind": "user_assisted_planar_perspective",
        "image": dict(image),
        "room_id": config.get("room_id", "provided-room"),
        "dimensions": dict(dimensions),
        "physical_accuracy_verified": False,
        "control_provenance": config.get(
            "control_provenance",
            {"status": "provided_unverified", "source": "caller-supplied correspondences"},
        ),
        "source_controls_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "floor_to_image": matrix.tolist(),
        "image_to_floor": inverse.tolist(),
        "floor_points_metres": world.tolist(),
        "image_points_pixels": pixels.tolist(),
        "pixel_uncertainty": pixel_uncertainty,
        "reprojection_rmse_pixels": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "camera_perspective": {
            "horizon_line": horizon.tolist(),
            "x_vanishing_point": vanishing([1, 0, 0]),
            "y_vanishing_point": vanishing([0, 1, 0]),
            "camera_intrinsics": None,
            "camera_pose_3d": None,
        },
        "limitations": [
            "Control points and dimensions are externally supplied, not automatically measured.",
            "A homography maps the assumed planar floor only; full camera pose/intrinsics are unknown.",
            "Four control points can fit exactly even when the physical assumptions are wrong.",
            "Lens distortion, unknown scale error and nonplanar floors are not modeled.",
        ],
    }


def project_floor_contact(calibration, pixel):
    point = np.asarray(pixel, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("contact needs two finite image coordinates")
    image = calibration["image"]
    if not 0 <= point[0] <= image["width"] or not 0 <= point[1] <= image["height"]:
        raise ValueError("contact point is outside source image")
    matrix = calibration["image_to_floor"]
    mapped = transform_points(matrix, [point])[0]
    jitter = calibration["pixel_uncertainty"]
    offsets = [[-jitter, 0], [jitter, 0], [0, -jitter], [0, jitter]]
    spread = transform_points(matrix, point + np.array(offsets))
    sensitivity = float(np.max(np.linalg.norm(spread - mapped, axis=1)))
    dimensions = calibration["dimensions"]
    inside = bool(0 <= mapped[0] <= dimensions["width"] and 0 <= mapped[1] <= dimensions["depth"])
    return {
        "position": [float(mapped[0]), float(mapped[1]), 0.0],
        "source_image_sha256": calibration["image"]["sha256"],
        "room_id": calibration["room_id"],
        "scale_provenance": calibration["dimensions"],
        "units": "m",
        "status": "estimated",
        "source": calibration["source_controls_sha256"],
        "method": "floor homography of assumed contact pixel",
        "pixel": point.tolist(),
        "uncertainty_metres": sensitivity,
        "uncertainty_scope": "local pixel-perturbation sensitivity only; excludes scale, control-point and contact-assumption error",
        "within_declared_floor": inside,
        "physical_accuracy_verified": False,
        "entity_centre_pose": None,
        "usable_as_verified_fit_input": False,
    }


def project_detection_contacts(calibration, record):
    if record.get("image_sha256") != calibration["image"]["sha256"]:
        raise ValueError("detections and floor calibration reference different image bytes")
    if [record["width"], record["height"]] != [
        calibration["image"]["width"],
        calibration["image"]["height"],
    ]:
        raise ValueError("calibration image dimensions differ from detections")
    results = []
    for detection in record["detections"]:
        box = detection["box"]
        contact = [(box[0] + box[2]) / 2, box[3]]
        result = {
            "detection_id": detection["detection_id"],
            "label": detection["label"],
            "assumption": "box bottom-centre is on the floor; may be false due to occlusion, elevated objects or fragments",
        }
        try:
            result["ground_contact_estimate"] = project_floor_contact(calibration, contact)
        except ValueError as exc:
            result["projection_error"] = str(exc)
        results.append(result)
    return {
        "schema_version": 1,
        "image_id": record["image_id"],
        "room_id": calibration["room_id"],
        "physical_accuracy_verified": False,
        "furniture_entities_created": False,
        "observations": results,
    }
