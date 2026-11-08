"""Semantic-supported observed surface patches without filling room boundaries."""

import numpy as np
from scipy.spatial import cKDTree

SURFACES = {1: "wall", 2: "floor", 3: "ceiling"}


def fit_observed_planes(cloud, min_points=150, threshold=0.035, min_views=2):
    points, labels = cloud["points"], cloud["labels"]
    rows = []
    rng = np.random.default_rng(47)
    for code, name in SURFACES.items():
        candidates = np.flatnonzero((labels == code) & (cloud["view_count"] >= min_views))
        for _ in range(6 if name == "wall" else 3):
            if len(candidates) < min_points:
                break
            xyz = points[candidates]
            sample_ids = rng.choice(len(xyz), min(5000, len(xyz)), replace=False)
            sample = xyz[sample_ids]
            best = np.zeros(len(sample), bool)
            for _ in range(200):
                a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
                n = np.cross(b - a, c - a)
                norm = np.linalg.norm(n)
                if norm < 1e-7:
                    continue
                n /= norm
                inliers = np.abs(np.sum((sample - a) * n, axis=1)) < threshold
                if inliers.sum() > best.sum():
                    best = inliers
            if best.sum() < min_points:
                break
            centre = sample[best].mean(axis=0)
            _, _, v = np.linalg.svd(sample[best] - centre, full_matrices=False)
            normal = v[-1]
            good = np.abs(np.sum((xyz - centre) * normal, axis=1)) < threshold
            if good.sum() < min_points:
                break
            ids = candidates[good]
            selected = points[ids]
            centre = selected.mean(axis=0)
            _, singular, v = np.linalg.svd(selected - centre, full_matrices=False)
            if singular[1] / np.sqrt(len(selected)) < 0.12:
                break
            normal = v[-1]
            axes = v[:2]
            uv = np.einsum("nj,ij->ni", selected - centre, axes, optimize=False)
            # Occupancy cells describe only measured support, with holes and no hull fill.
            cell = 0.08
            cells = np.floor(uv / cell).astype(int)
            unique = np.unique(cells, axis=0)
            # Exclude very isolated support before reporting an area.
            neighbours = cKDTree(unique).query_ball_point(unique, 1.5)
            unique = unique[np.array([len(n) >= 3 for n in neighbours])]
            rows.append(
                {
                    "id": f"sensor-plane-{len(rows)}",
                    "surface_hypothesis": name,
                    "normal": normal.tolist(),
                    "offset_metres": float(-normal @ centre),
                    "origin_metres": centre.tolist(),
                    "tangent_axes": axes.tolist(),
                    "support_voxel_indices": ids.tolist(),
                    "support_points": len(ids),
                    "support_minimum_views": min_views,
                    "rms_residual_metres": float(
                        np.sqrt(np.mean(np.sum((selected - centre) * normal, axis=1) ** 2))
                    ),
                    "observed_occupancy_cell_metres": cell,
                    "observed_cells_uv": unique.tolist(),
                    "observed_support_area_estimate_m2": float(len(unique) * cell**2),
                    "status": "estimated_plane_from_measured_depth_and_model_semantics",
                    "hidden_extent": "unknown",
                    "closed_boundary": None,
                    "verified_room_surface": False,
                    "classification_confidence": "heuristic_semantic_consensus_not_calibrated",
                    "units": "m",
                }
            )
            candidates = candidates[~good]
    return rows


def supported_mesh(samples, pose, planes, stride=4):
    """Image-grid triangles only across contiguous same-semantic measured support."""
    from .rgbd import transform

    points = transform(samples["points"], pose)
    pixels = samples["pixels"]
    lookup = {tuple(p): i for i, p in enumerate(pixels)}
    triangles = []
    plane_ids = []
    for plane in planes:
        code = next(k for k, v in SURFACES.items() if v == plane["surface_hypothesis"])
        on_plane = (samples["labels"] == code) & (
            np.abs(np.sum(points * np.asarray(plane["normal"]), axis=1) + plane["offset_metres"])
            < 0.035
        )
        for i, (x, y) in enumerate(pixels):
            if not on_plane[i]:
                continue
            right = lookup.get((x + stride, y))
            lower = lookup.get((x, y + stride))
            diagonal = lookup.get((x + stride, y + stride))
            for ids in ((i, right, lower), (right, diagonal, lower)):
                if any(j is None for j in ids) or not all(on_plane[j] for j in ids):
                    continue
                xyz = points[list(ids)]
                edges = np.linalg.norm(xyz - np.roll(xyz, 1, axis=0), axis=1)
                if (
                    edges.max() > 0.12
                    or np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) < 1e-8
                ):
                    continue
                triangles.append(ids)
                plane_ids.append(plane["id"])
    return points, np.array(triangles, dtype=np.uint32).reshape(-1, 3), plane_ids


def intersections(planes):
    results = []
    for index, a in enumerate(planes):
        for b in planes[index + 1 :]:
            n1, n2 = np.array(a["normal"], dtype=float), np.array(b["normal"], dtype=float)
            direction = np.cross(n1, n2)
            if np.linalg.norm(direction) < 0.3:
                continue
            direction /= np.linalg.norm(direction)
            point = np.linalg.lstsq(
                np.array([n1, n2]), -np.array([a["offset_metres"], b["offset_metres"]]), rcond=None
            )[0]
            # Plane intersection alone is not observed wall/floor boundary evidence.
            results.append(
                {
                    "planes": [a["id"], b["id"]],
                    "point_metres": point.tolist(),
                    "direction": direction.tolist(),
                    "status": "inferred_infinite_plane_intersection_not_observed_junction",
                    "supported_segment": None,
                    "usable_as_room_boundary": False,
                }
            )
    return results
