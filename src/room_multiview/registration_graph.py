"""Reference-free selection of a supported registration component."""
from collections import deque

import numpy as np


def largest_component_cameras(keys, pairs, preferred=(), use_icp=False):
    adjacency = {key: [] for key in keys}
    for pair in pairs:
        if pair["status"] == "accepted":
            a, b = pair["views"]
            adjacency[a].append((b, pair, False))
            adjacency[b].append((a, pair, True))
    remaining = set(keys)
    components = []
    for key in keys:
        if key not in remaining:
            continue
        component, queue = [], deque([key])
        remaining.remove(key)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor, _, _ in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    component = max(components, key=len)
    anchor = next((key for key in keys if key in component and key in preferred), component[0])
    cameras, parents, used = {anchor: np.eye(4)}, {}, []
    queue = deque([anchor])
    while queue:
        current = queue.popleft()
        for neighbor, pair, reverse in sorted(adjacency[current], key=lambda item: item[1]["sample_gap"]):
            if neighbor in cameras:
                continue
            transform = np.asarray(pair["icp"]["camera_b_from_a"] if use_icp and pair.get("icp", {}).get("adopted") else pair["camera_b_from_a"], float)
            cameras[neighbor] = (np.linalg.inv(transform) if reverse else transform) @ cameras[current]
            parents[neighbor] = current
            used.append(pair)
            queue.append(neighbor)
    return cameras, parents, used, anchor, sorted(map(len, components), reverse=True)
