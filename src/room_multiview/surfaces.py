"""Fit supported sparse semantic plane patches, never complete hidden room geometry."""

import numpy as np


def plane_patches(landmarks, minimum=25):
    result = []
    rng = np.random.default_rng(17)
    for surface in ("floor", "wall", "ceiling"):
        rows = [p for p in landmarks if p.get("semantic_surface") == surface]
        for _ in range(3 if surface == "wall" else 1):
            if len(rows) < minimum:
                break
            points = np.array([p["position"] for p in rows])
            threshold = max(0.005, float(np.median(np.linalg.norm(points, axis=1))) * 0.005)
            best = np.zeros(len(rows), bool)
            for _ in range(200):
                a, b, c = points[rng.choice(len(points), 3, replace=False)]
                normal = np.cross(b - a, c - a)
                length = np.linalg.norm(normal)
                if length < 1e-8:
                    continue
                normal /= length
                supported = np.abs((points - a) @ normal) <= threshold
                if supported.sum() > best.sum():
                    best = supported
            if best.sum() < minimum:
                break
            selected = points[best]
            centre = selected.mean(axis=0)
            _, singular, vectors = np.linalg.svd(selected - centre, full_matrices=False)
            if singular[1] < threshold * 3:
                break
            normal = vectors[-1]
            result.append(
                {
                    "id": f"plane-{len(result)}",
                    "surface_hypothesis": surface,
                    "normal": normal.tolist(),
                    "offset": float(-normal @ centre),
                    "supporting_landmarks": [p["id"] for p, keep in zip(rows, best) if keep],
                    "rms_residual_arbitrary_units": float(
                        np.sqrt(np.mean(((selected - centre) @ normal) ** 2))
                    ),
                    "status": "estimated_sparse_patch_not_complete_wall",
                    "units": "arbitrary",
                    "boundary_polygon": None,
                    "hidden_extent": "unknown",
                    "human_verified": False,
                }
            )
            rows = [p for p, keep in zip(rows, best) if not keep]
    return result
