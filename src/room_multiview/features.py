"""CPU SIFT with mutual ratio-test correspondences. No pretrained downloads."""

import numpy as np


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Install the optional ".[multiview]" extra for OpenCV') from exc
    cv2.setNumThreads(1)
    cv2.setRNGSeed(17)
    return cv2


def extract_features(image, max_features=6000):
    cv = opencv()
    gray = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)
    keys, descriptors = cv.SIFT_create(nfeatures=max_features).detectAndCompute(gray, None)
    return {
        "pixels": np.array([k.pt for k in keys], dtype=np.float64).reshape(-1, 2),
        "descriptors": descriptors if descriptors is not None else np.empty((0, 128), np.float32),
    }


def match_features(first, second, ratio=0.75):
    cv = opencv()
    a, b = first["descriptors"], second["descriptors"]
    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv.BFMatcher(cv.NORM_L2)

    def accepted(x, y):
        return {
            m.queryIdx: m.trainIdx
            for pair in matcher.knnMatch(x, y, k=2)
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < ratio * n.distance
        }

    forward, backward = accepted(a, b), accepted(b, a)
    return np.array(
        [(i, j) for i, j in forward.items() if backward.get(j) == i], dtype=int
    ).reshape(-1, 2)
