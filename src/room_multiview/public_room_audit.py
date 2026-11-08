"""Conservative acceptance of an entire cached fragment for experimental fusion."""
import math


def approve_fragment(stats, tracking_views, primary_views):
    """Never accept a fragment from a partial or nonfinite reference audit."""
    if primary_views < 2 or tracking_views < 2:
        return False
    if stats.get("evaluated_non_anchor_views") != tracking_views - 1:
        return False
    translation = stats.get("translation_max_metres")
    rotation = stats.get("rotation_max_degrees")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (translation, rotation)):
        return False
    return translation <= 0.10 and rotation <= 3.0
