"""Bounded classical photo replacement; no model downloads or physical-fit claims.

All positions are in source image pixels. A positive rotation is clockwise.
``skew`` is a trapezoid-perspective fraction in [-.45, .45]. Optional depth is
relative inverse depth (larger = nearer), or a dict with ``values`` and an
explicit ``higher_is_nearer`` flag. It is never treated as metric geometry.
"""
from __future__ import annotations

import math
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MAX_PIXELS = 12_000_000
MAX_SIDE = 4096
MAX_MASK_FRACTION = 0.72


def _image(image, mode="RGB"):
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a decoded photograph")
    w, h = image.size
    if w < 8 or h < 8 or max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
        raise ValueError("Image must be at least 8 pixels per side and at most 4096 pixels / 12 megapixels")
    return image.convert(mode)


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _box(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("Select a furniture rectangle first")
    x1, y1, x2, y2 = [_number(v, "selection coordinate") for v in box]
    w, h = size
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Selection rectangle must have positive width and height")
    clipped = (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))),
               min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2))))
    x1, y1, x2, y2 = clipped
    if x2 - x1 < 3 or y2 - y1 < 3:
        raise ValueError("Selection is too small or outside the photograph")
    if (x2 - x1) * (y2 - y1) > w * h * 0.86:
        raise ValueError("Select an individual furniture object, not nearly the entire photograph")
    return clipped


def _bbox(binary):
    ys, xs = np.nonzero(binary)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def _downsample(rgb, max_side=800):
    scale = min(1.0, max_side / max(rgb.shape[:2]))
    if scale == 1:
        return rgb, scale
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA), scale


def make_mask(image, box, edits=None, initial_mask=None):
    """GrabCut the selected box, then apply explicit polygon corrections.

    Polygon edits are in image coordinates and are the final authority over the
    automatic mask. An unreliable empty/tiny GrabCut result uses a visibly
    reported rectangle fallback rather than claiming a successful instance mask.
    """
    started = perf_counter()
    image = _image(image)
    w, h = image.size
    x1, y1, x2, y2 = _box(box, image.size)
    rgb = np.asarray(image)
    small, scale = _downsample(rgb)
    sh, sw = small.shape[:2]
    sx1, sy1 = max(0, int(x1 * scale)), max(0, int(y1 * scale))
    sx2, sy2 = min(sw, int(math.ceil(x2 * scale))), min(sh, int(math.ceil(y2 * scale)))
    # Explicit background exists outside the target rectangle even at image edges.
    labels = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    labels[sy1:sy2, sx1:sx2] = cv2.GC_PR_FGD
    warnings = []
    method = "opencv_grabcut_box"
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)
        cv2.grabCut(small, labels, None, bg, fg, 4, cv2.GC_INIT_WITH_MASK)
        selected = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
        count = int(np.count_nonzero(selected))
        if count < max(9, (sx2 - sx1) * (sy2 - sy1) * 0.015):
            raise ValueError("GrabCut found insufficient furniture pixels")
        mask = cv2.resize(selected, (w, h), interpolation=cv2.INTER_NEAREST)
        # Remove thin original edges too, without spilling beyond the user's box.
        radius = max(1, round(max(w, h) / 900))
        mask = cv2.dilate(mask, np.ones((radius * 2 + 1,) * 2, np.uint8))
        bounded = np.zeros((h, w), np.uint8)
        bounded[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = bounded
    except (cv2.error, ValueError) as exc:
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        method = "rectangle_fallback"
        warnings.append("Automatic mask failed; the selection rectangle is used. Correct the mask before removal.")
        warnings.append(type(exc).__name__ + ": " + str(exc)[:180])
    if initial_mask is not None:
        semantic = _mask_array(initial_mask, image.size, "Initial semantic mask")
        if semantic.any():
            radius = max(1, round(max(w, h) / 900))
            mask = cv2.dilate(semantic.astype(np.uint8) * 255, np.ones((radius * 2 + 1,) * 2, np.uint8))
            bounded = np.zeros((h, w), np.uint8)
            bounded[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = bounded
            method = "provided_semantic_mask"
            warnings = ["Provided semantic furniture mask used; semantic labels do not establish separate physical instances."]
        else:
            warnings.append("Provided semantic mask was empty; classical selection mask retained.")
    edits = edits or []
    if not isinstance(edits, list) or len(edits) > 100:
        raise ValueError("At most 100 mask correction polygons are supported")
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("mode") not in ("add", "subtract"):
            raise ValueError("Mask edit mode must be add or subtract")
        points = edit.get("points")
        if not isinstance(points, list) or not 3 <= len(points) <= 500:
            raise ValueError("A mask correction polygon needs 3–500 points")
        converted = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Mask correction points need x and y")
            x, y = (_number(v, "mask correction coordinate") for v in point)
            converted.append([round(min(w - 1, max(0, x))), round(min(h - 1, max(0, y)))])
        cv2.fillPoly(mask, [np.array(converted, np.int32)], 255 if edit["mode"] == "add" else 0)
    fraction = float(np.count_nonzero(mask) / (w * h))
    if not fraction:
        raise ValueError("The corrected mask is empty; select or add furniture pixels")
    if fraction > MAX_MASK_FRACTION:
        raise ValueError("Removal mask covers too much of the photograph; narrow the selection")
    return Image.fromarray(mask), {
        "method": method, "seconds": perf_counter() - started,
        "mask_fraction": fraction, "mask_bbox": _bbox(mask),
        "manual_corrections": len(edits), "warnings": warnings,
        "semantic_instance_segmentation": False, "initial_semantic_mask_used": method == "provided_semantic_mask",
        "note": "Classical appearance mask; manual inspection is required, especially near similar colors or other furniture.",
    }


def extract_asset(image):
    """Preserve existing alpha, otherwise try classical background isolation."""
    started = perf_counter()
    rgba = _image(image, "RGBA")
    data = np.asarray(rgba).copy()
    alpha = data[:, :, 3]
    warnings = []
    if np.any(alpha < 250):
        if not np.any(alpha > 5):
            raise ValueError("Replacement image is fully transparent")
        # No erosion, thresholding or resegmentation of a supplied cutout.
        bounds = _bbox(alpha > 0)
        return rgba.crop(tuple(bounds)), {
            "method": "existing_alpha_preserved", "seconds": perf_counter() - started,
            "source_bbox": bounds, "foreground_fraction": float(np.mean(alpha > 0)),
            "warnings": warnings,
        }
    rgb = data[:, :, :3]
    small, _ = _downsample(rgb)
    h, w = small.shape[:2]
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]], axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.max(np.abs(small.astype(np.float32) - background), axis=2)
    uniform_border = float(np.mean(np.max(np.abs(border.astype(np.float32) - background), axis=1) < 28))
    labels = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    labels[:2, :] = labels[-2:, :] = labels[:, :2] = labels[:, -2:] = cv2.GC_BGD
    method = "opencv_grabcut_background"
    try:
        if uniform_border > 0.7:
            candidate = (distance < 26).astype(np.uint8)
            count, components = cv2.connectedComponents(candidate, connectivity=8)
            edge_components = np.unique(np.concatenate([components[0], components[-1], components[:, 0], components[:, -1]]))
            edge_components = edge_components[edge_components != 0]
            connected_background = np.isin(components, edge_components)
            labels[connected_background] = cv2.GC_BGD
            labels[(distance > 60) & ~connected_background] = cv2.GC_FGD
            method = "border_seeded_grabcut"
        else:
            warnings.append("Nonuniform replacement background; automatic extraction may retain background or miss furniture parts.")
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)
        cv2.grabCut(small, labels, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
        binary = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
        fraction = float(np.mean(binary > 0))
        if fraction < .005 or fraction > .97:
            raise ValueError("Foreground separation is unreliable")
        extracted_alpha = cv2.resize(binary, rgba.size, interpolation=cv2.INTER_LINEAR)
        data[:, :, 3] = extracted_alpha
        bounds = _bbox(extracted_alpha > 0)
        result = Image.fromarray(data).crop(tuple(bounds))
    except (cv2.error, ValueError) as exc:
        result = rgba
        bounds = [0, 0, rgba.width, rgba.height]
        fraction = 1.0
        method = "opaque_background_retained"
        warnings.append("Could not isolate replacement reliably. Its original background is retained; upload a transparent cutout for cleaner insertion.")
        warnings.append(type(exc).__name__ + ": " + str(exc)[:180])
    return result, {
        "method": method, "seconds": perf_counter() - started,
        "source_bbox": bounds, "foreground_fraction": fraction,
        "border_uniform_fraction": uniform_border, "warnings": warnings,
        "note": "Classical extraction is approximate; preview edges before using this asset.",
    }


def _mask_array(mask, size, name):
    if mask is None:
        return np.zeros((size[1], size[0]), bool)
    if isinstance(mask, Image.Image):
        if mask.size != size:
            raise ValueError(name + " dimensions do not match the photograph")
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
    if array.shape != (size[1], size[0]) or not np.isfinite(array).all():
        raise ValueError(name + " must be finite and match photograph dimensions")
    return array > 0


def placement_geometry(placement, asset_size, size):
    """Image-space corners shared by rasterization and its preview contract.

    Missing lock_aspect retains legacy saved placements. New UI placements lock
    by default. The floor anchor is the cutout bottom midpoint, not measured feet.
    """
    w, h = size
    aw, ah = asset_size
    if min(aw, ah) < 2 or not isinstance(placement, dict):
        raise ValueError("Invalid replacement asset or placement")
    x = _number(placement.get("x", w / 2), "replacement x")
    y = _number(placement.get("y", h / 2), "replacement y")
    pw = _number(placement.get("width", w / 3), "replacement width")
    ph = _number(placement.get("height", h / 3), "replacement height")
    locked = bool(placement.get("lock_aspect", False))
    if locked:
        ph = pw * (ah - 1) / (aw - 1)
    rotation = _number(placement.get("rotation", 0), "rotation")
    skew = _number(placement.get("skew", 0), "perspective skew")
    if not 2 <= pw <= 2 * w or not 2 <= ph <= 2 * h:
        raise ValueError("Replacement width and height must be 2 pixels to twice the photograph size")
    if not -w <= x <= 2 * w or not -h <= y <= 2 * h:
        raise ValueError("Replacement anchor is too far outside the photograph")
    if abs(skew) > .45 or abs(rotation) > 3600:
        raise ValueError("Perspective exceeds the safe ±0.45 limit; use a better-matched source view")
    stretch = (ph / (ah - 1)) / (pw / (aw - 1))
    if not .5 <= stretch <= 2:
        raise ValueError("Extreme nonuniform stretch; lock the source aspect ratio")
    anchor = placement.get("anchor", "center")
    if anchor not in ("center", "floor"):
        raise ValueError("Anchor must be center or floor")
    top_factor = 1 - abs(skew) * .55
    shift = skew * pw * .3
    corners = np.array([[-pw / 2 * top_factor + shift, -ph / 2],
                        [pw / 2 * top_factor + shift, -ph / 2],
                        [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float64)
    if anchor == "floor":
        corners[:, 1] -= ph / 2
    theta = math.radians(rotation % 360)
    matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    corners = corners @ matrix.T + [x, y]
    validate_quad(corners)
    warnings = []
    if abs(stretch - 1) > .15:
        warnings.append("Nonuniform scaling changes the furniture proportions. Enable the aspect-ratio lock.")
    if abs(skew) > .15:
        warnings.append("Large 2D perspective distortion; a different product photograph may match the room better.")
    if abs(rotation % 360 if rotation % 360 < 180 else rotation % 360 - 360) > 12:
        warnings.append("Large in-plane rotation cannot correct an incompatible 3D source view.")
    return {"x": x, "y": y, "width": pw, "height": ph, "rotation": rotation,
            "skew": skew, "flip": bool(placement.get("flip", False)), "anchor": anchor,
            "lock_aspect": locked, "vertical_to_horizontal_scale": stretch,
            "corners": corners.tolist(), "warnings": warnings}


def validate_quad(corners):
    q = np.asarray(corners, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all() or np.max(np.abs(q)) > 100000:
        raise ValueError("Perspective corners must be finite, bounded image coordinates")
    edges = np.roll(q, -1, axis=0) - q
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    lengths = np.linalg.norm(edges, axis=1)
    if not np.all(cross > 1e-4) or lengths.min() < 1 or lengths.max() / lengths.min() > 30:
        raise ValueError("Perspective quadrilateral is crossed, reversed, degenerate or excessively stretched")


def _warp_asset(asset, placement, size):
    w, h = size
    normalized = placement_geometry(placement, asset.size, size)
    corners = np.array(normalized["corners"], np.float32)
    source = np.asarray(asset.convert("RGBA"))
    if normalized["flip"]:
        source = source[:, ::-1].copy()
    ah, aw = source.shape[:2]
    source_points = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], np.float32)
    if aw < 2 or ah < 2:
        raise ValueError("Replacement cutout is too small")
    transform = cv2.getPerspectiveTransform(source_points, corners)
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("Replacement perspective transform is degenerate")
    denominators = np.c_[source_points, np.ones(4)] @ transform[2]
    if np.min(np.abs(denominators)) < 1e-8 or not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("Perspective transform crosses an infinite image plane")
    # Premultiplication prevents background-color fringes along semi-transparent edges.
    premultiplied = source.astype(np.float32) / 255
    premultiplied[:, :, :3] *= premultiplied[:, :, 3:4]
    warped = cv2.warpPerspective(premultiplied, transform, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped, normalized


def composite(room, mask, asset, placement, options=None, depth=None, protected_mask=None, *,
              layers=None, background_override=None, cache_dir=None, expansion_exclusion=None, background_override_mask=None):
    """Inpaint selected furniture and composite a transformed replacement.

    Removal/background pixels are estimated with classical Telea inpainting.
    Foreground protection is explicit or inferred heuristically from supplied
    relative depth. It is not exact 3D geometry or a photorealistic renderer.
    """
    started = perf_counter()
    room = _image(room)
    asset = _image(asset, "RGBA")
    w, h = room.size
    original = np.asarray(room).copy()
    removal = _mask_array(mask, room.size, "Removal mask")
    if not removal.any() or removal.mean() > MAX_MASK_FRACTION:
        raise ValueError("Removal mask is empty or covers too much of the photograph")
    protection = _mask_array(protected_mask, room.size, "Foreground protection mask")
    options = options or {}
    if not isinstance(options, dict):
        raise ValueError("Composite options must be an object")
    warnings = []
    depth_status = "not_requested"
    depth_protected_pixels = 0
    if options.get("depth_assist"):
        if depth is None:
            depth_status = "unavailable"
            warnings.append("Relative depth is unavailable; depth-assisted occlusion was skipped. Manual placement remains available.")
        else:
            try:
                depth_values = depth.get("values") if isinstance(depth, dict) else depth
                higher_nearer = bool(depth.get("higher_is_nearer", True)) if isinstance(depth, dict) else True
                values = np.asarray(depth_values, dtype=np.float64)
                if values.shape != (h, w):
                    raise ValueError("depth dimensions differ")
                valid = np.isfinite(values)
                if valid.mean() < .9 or (valid & removal).sum() < 10:
                    raise ValueError("insufficient valid relative depth")
                finite = values[valid]
                low, high = np.percentile(finite, [2, 98])
                if high - low <= max(1e-12, abs(high) * 1e-12):
                    raise ValueError("constant or degenerate relative depth")
                near = (values - low) / (high - low)
                if not higher_nearer:
                    near = 1 - near
                target = float(np.median(near[valid & removal]))
                nearer = valid & ~removal & (near > target + .08)
                depth_protected_pixels = int(nearer.sum())
                protection |= nearer
                depth_status = "relative_depth_occlusion_heuristic"
                warnings.append("Depth occlusion uses a heuristic relative-depth ordering around the selected object, not calibrated physical placement.")
            except (ValueError, TypeError, OverflowError) as exc:
                depth_status = "invalid_skipped"
                warnings.append("Depth assistance skipped: " + str(exc))
    removal &= ~protection
    radius = _number(options.get("inpaint_radius", 5), "inpaint radius")
    if not 1 <= radius <= 15:
        raise ValueError("Inpaint radius must be between 1 and 15 pixels")
    from .restoration import restore_background, refine_alpha, contact_shadow
    excluded = _mask_array(expansion_exclusion, room.size, "Expansion exclusion") if expansion_exclusion is not None else None
    override = np.array(_image(background_override)) if background_override is not None else None
    asset, edge_info = refine_alpha(asset, options)
    if layers is not None:layers["refined-asset"] = asset.copy()
    warped, normalized = _warp_asset(asset, placement, room.size)
    warnings.extend(normalized["warnings"])
    alpha = np.clip(warped[:, :, 3], 0, 1)
    if not np.any(alpha > .001):
        raise ValueError("Replacement is completely outside the photograph")
    alpha[protection] = 0
    override_scope = _mask_array(background_override_mask, room.size, "Background correction scope") if background_override_mask is not None else None
    background, removal, repair_info = restore_background(original, removal, protection, options, excluded, override, cache_dir, coverage_alpha=alpha, override_mask=override_scope)
    if options.get('restoration_scope') == 'visible':
        from .visibility import coverage_masks, visible_seam_metrics
        from hashlib import sha256
        target, exposed, opaque = coverage_masks(removal, protection, alpha, repair_info['visibility']['margin'])
        repair_info['visibility'].update(
            coverage_sha256=sha256(opaque.tobytes()).hexdigest(),
            target_sha256=sha256(target.tobytes()).hexdigest(),
            exposed_unrestored_pixels=int((exposed&~target).sum()),
            protected_pixel_changes=int(np.any(background!=original,axis=2)[protection].sum()),
            seam=visible_seam_metrics(background,target,alpha,protection))
    inpaint_seconds = repair_info["seconds"]
    if layers is not None:
        layers["background"] = Image.fromarray(background.copy())
        layers["unknown"] = Image.fromarray(removal.astype(np.uint8)*255)
        layers["protection"] = Image.fromarray(protection.astype(np.uint8)*255)
        if options.get('restoration_scope') == 'visible':
            from .visibility import coverage_masks
            target, exposed, opaque = coverage_masks(removal, protection, alpha, repair_info['visibility']['margin'])
            layers['restoration-mask'] = Image.fromarray(target.astype(np.uint8)*255)
            layers['exposed'] = Image.fromarray(exposed.astype(np.uint8)*255)
            layers['replacement-coverage'] = Image.fromarray(opaque.astype(np.uint8)*255)
    background = background.astype(np.float32)
    rgb = np.divide(warped[:, :, :3], warped[:, :, 3:4], out=np.zeros_like(warped[:, :, :3]), where=warped[:, :, 3:4] > 1e-5) * 255
    color_gain = 1.0
    if options.get("color_match") and np.count_nonzero(alpha > .5) > 10:
        footprint = alpha > .1
        ring = cv2.dilate(footprint.astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool) & ~footprint & ~protection
        if ring.any():
            target_luminance = float(np.median(background[ring].mean(axis=1)))
            source_luminance = float(np.median(rgb[alpha > .5].mean(axis=1)))
            if source_luminance > 5:
                color_gain = float(np.clip(target_luminance / source_luminance, .85, 1.15))
                rgb = np.clip(rgb * color_gain, 0, 255)
        warnings.append("Brightness matching is a bounded image adjustment, not recovered room lighting.")
    shadow_info = {}
    shadow_mask = np.zeros((h, w), np.float32)
    if options.get("shadow") and options.get("shadow_mode") == "contact_v2":
        shadow_mask, shadow_info = contact_shadow(warped[:, :, 3], normalized, protection, options)
        background *= 1 - shadow_mask[:, :, None]
        warnings.append("Contact shadows are editable image-space estimates, not physical lighting.")
    elif options.get("shadow"):
        shadow_image = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(shadow_image)
        if options.get("shadow_mode") == "contact":
            # Short local shadows under actual visible alpha at two bottom contacts.
            footprint = warped[:, :, 3] > .3
            ys, xs = np.nonzero(footprint)
            if len(xs):
                left, right = int(xs.min()), int(xs.max())
                for lo, hi in [(left, left + (right-left)//3), (right-(right-left)//3, right+1)]:
                    yy, xx = np.nonzero(footprint[:, lo:hi])
                    if len(xx):
                        near = yy >= np.percentile(yy, 99)
                        cx, cy = float(np.median(xx[near]) + lo), float(np.max(yy))
                        rx, ry = max(3, normalized["width"] * .025), 3
                        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=48)
            shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(3))
        else:
            corners = np.asarray(normalized["corners"])
            cx, cy = ((normalized["x"], normalized["y"] + normalized["height"] * .44)
                      if normalized["anchor"] == "center" else corners[2:4].mean(axis=0) - [0, normalized["height"] * .06])
            rx, ry = normalized["width"] * .43, max(2, normalized["height"] * .045)
            draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=45)
            shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(max(1, normalized["height"] * .02)))
        shadow_mask = np.asarray(shadow_image).astype(np.float32) / 255
        shadow_mask[protection] = 0
        background *= 1 - shadow_mask[:, :, None]
        warnings.append("Contact shadow is an optional soft 2D approximation; floor and illumination were not measured.")
    if layers is not None:
        layers["shadow"] = Image.fromarray(np.rint(shadow_mask*255).astype(np.uint8))
        layers["placed-alpha"] = Image.fromarray(np.rint(alpha*255).astype(np.uint8))
    result = np.clip(rgb * alpha[:, :, None] + background * (1 - alpha[:, :, None]), 0, 255).round().astype(np.uint8)
    result[protection] = original[protection]
    affected = removal | (alpha > 0) | (shadow_mask > 0)
    # This explicit copy prevents resampling/rounding from changing unrelated pixels.
    result[~affected] = original[~affected]
    changed = np.any(result != original, axis=2)
    warnings.insert(0, "Rough composite: removed furniture hides unobserved background; classical fill may smear textures and cannot reconstruct hidden room structure.")
    return Image.fromarray(result), {
        "method": "classical_inpaint_and_alpha_composite", "seconds": perf_counter() - started,
        "inpaint_seconds": inpaint_seconds, "inpaint_method": "opencv_" + repair_info["method"],
        "background_repair": repair_info, "edge_refinement": edge_info, "shadow_layer": shadow_info,
        "inpaint_radius": radius, "placement": normalized,
        "removed_pixels": int(removal.sum()), "foreground_protected_pixels": int(protection.sum()),
        "depth_status": depth_status, "depth_protected_pixels": depth_protected_pixels,
        "color_gain": color_gain, "shadow_enabled": bool(options.get("shadow")),
        "changed_pixels": int(changed.sum()), "changed_bbox": _bbox(changed),
        "replacement_footprint_bbox": _bbox(alpha > 0), "shadow_bbox": _bbox(shadow_mask > 0),
        "geometry": {"coordinates": "image_pixels", "physical_dimensions_verified": False,
                     "measured_room_geometry": None, "hidden_background": "unknown_estimated_fill"},
        "label": "Visual preview — physical dimensions not verified.",
        "warnings": warnings,
    }
