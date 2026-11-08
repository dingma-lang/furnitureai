"""Bounded simple polygons for explicit destination plans; no image inference."""

import math

from .model import EPS, vector


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p, a, b):
    return abs(cross(a, b, p)) <= EPS and all(
        min(a[i], b[i]) - EPS <= p[i] <= max(a[i], b[i]) + EPS for i in (0, 1)
    )


def intersects(a, b, c, d, proper=False):
    values = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if values[0] * values[1] < -EPS and values[2] * values[3] < -EPS:
        return True
    return not proper and any(
        on_segment(p, s, e) for p, s, e in [(c, a, b), (d, a, b), (a, c, d), (b, c, d)]
    )


def area(poly):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(poly, poly[1:] + poly[:1])) / 2


def polygon(value):
    if not isinstance(value, list) or not 3 <= len(value) <= 100:
        raise ValueError("A room polygon requires 3–100 points")
    points = [vector(p, 2, "polygon point") for p in value]
    if max(abs(v) for p in points for v in p) > 10000:
        raise ValueError("Polygon coordinates exceed planning limits")
    n = len(points)
    for i in range(n):
        if math.dist(points[i], points[(i + 1) % n]) <= 1e-6:
            raise ValueError("Duplicate or zero-length polygon edge")
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if intersects(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                raise ValueError("Self-intersecting room polygon")
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        if (
            abs(cross(a, b, c)) <= EPS
            and (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) < 0
        ):
            raise ValueError("Polygon edge doubles back")
    if abs(area(points)) <= 1e-6:
        raise ValueError("Degenerate room polygon")
    return points


def inside(p, poly):
    result = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if on_segment(p, a, b):
            return True
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            result = not result
    return result


def contains(outer, inner):
    # Midpoints and proper edge crossings prevent concave notch corner-only errors.
    return all(inside(p, outer) for p in inner) and all(
        inside([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], outer)
        and not any(intersects(a, b, c, d, True) for c, d in zip(outer, outer[1:] + outer[:1]))
        for a, b in zip(inner, inner[1:] + inner[:1])
    )


def overlaps(first, second):
    return (
        any(
            intersects(a, b, c, d, True)
            for a, b in zip(first, first[1:] + first[:1])
            for c, d in zip(second, second[1:] + second[:1])
        )
        or any(
            inside(p, second)
            and not any(on_segment(p, a, b) for a, b in zip(second, second[1:] + second[:1]))
            for p in [
                *first,
                *[
                    [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
                    for a, b in zip(first, first[1:] + first[:1])
                ],
            ]
        )
        or any(
            inside(p, first)
            and not any(on_segment(p, a, b) for a, b in zip(first, first[1:] + first[:1]))
            for p in [
                *second,
                *[
                    [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
                    for a, b in zip(second, second[1:] + second[:1])
                ],
            ]
        )
        or (contains(first, second) and contains(second, first))
    )
