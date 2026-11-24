"""Opt-in v2 cache contracts: raw inference identity and supervision are separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import digest

SCHEMA = 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require_sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def file_identity(files):
    """Hash an explicit logical-role -> file map, without importing/loading models.

    Callers list all relevant weights, head deltas, tokenizer and processor files.
    Paths are not identities: moving identical files does not invalidate a cache.
    """
    if not isinstance(files, dict) or not files or any(not isinstance(k, str) or not k for k in files):
        raise ValueError("Explicit nonempty model/processor file roles required")
    hashes = {}
    for role, path in sorted(files.items()):
        path = Path(path)
        before = path.stat()
        hashes[role] = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("File changed while fingerprinting")
    return {"files": hashes, "sha256": fingerprint(hashes)}


def make_contract(*, image_id, image_sha256, model, preprocessing, categories, prompts,
                  inference_settings, supervision, query_categories=None, task="detection"):
    """Construct a self-contained contract, not a human-review or rights attestation.

    Annotations/ignore regions/taxonomy/completeness/group membership belong to
    supervision. If they determine a crop or other pixels, include that transform
    in preprocessing as well. Raw features/predictions survive label-only changes.
    """
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("Image identity required")
    require_sha(image_sha256, "image_sha256")
    if not isinstance(model, dict) or not isinstance(model.get("model_id"), str) or not model["model_id"]:
        raise ValueError("Model identity required")
    for key in ("checkpoint_sha256", "processor_sha256"):
        require_sha(model.get(key), key)
    if task not in ("detection", "image_embedding"):
        raise ValueError("Unsupported inference task")
    if task == "image_embedding":
        if categories != [] or prompts != [] or query_categories not in (None, []):
            raise ValueError("Image-only features must not claim text/category queries")
        query_categories = []
    else:
        if (
            not isinstance(categories, list)
            or not categories
            or any(not isinstance(c, str) or not c for c in categories)
            or len(set(categories)) != len(categories)
        ):
            raise ValueError("Ordered unique category strings required")
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(p, str) or not p for p in prompts)
        ):
            raise ValueError("Ordered prompts required")
        if query_categories is None:
            query_categories = list(categories)
        if (
            not isinstance(query_categories, list)
            or len(query_categories) != len(prompts)
            or any(c not in categories for c in query_categories)
        ):
            raise ValueError("Explicit prompt-to-category ordering required")
    for name, value in (("preprocessing", preprocessing), ("inference settings", inference_settings)):
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Explicit {name} required")
    required = {"annotations", "ignore_regions", "taxonomy_version", "completeness", "source_group_id"}
    if (
        not isinstance(supervision, dict)
        or not required <= supervision.keys()
        or not isinstance(supervision["source_group_id"], str)
        or not supervision["source_group_id"]
    ):
        raise ValueError("Annotation, ignored-region, taxonomy, completeness and group context required")
    if (
        not isinstance(supervision["annotations"], list)
        or not isinstance(supervision["ignore_regions"], list)
        or not isinstance(supervision["completeness"], dict)
        or not isinstance(supervision["taxonomy_version"], str)
        or not supervision["taxonomy_version"]
    ):
        raise ValueError("Malformed supervision context")
    # JSON round-trip owns the returned dependency snapshot instead of aliasing caller dictionaries.
    inference = json.loads(canonical(dict(
        task=task, image_id=image_id, image_sha256=image_sha256, model=model,
        preprocessing=preprocessing, categories=categories, prompts=prompts,
        query_categories=query_categories, settings=inference_settings,
    )))
    context = json.loads(canonical(supervision))
    result = dict(schema_version=SCHEMA, inference=inference, supervision=context)
    result["inference_key"] = fingerprint(dict(schema_version=SCHEMA, inference=inference))
    result["context_key"] = fingerprint(dict(schema_version=SCHEMA, supervision=context))
    result["full_key"] = fingerprint(dict(inference_key=result["inference_key"], context_key=result["context_key"]))
    return result


def validate_contract(contract, *, image_id=None, image_sha256=None, model=None):
    if (
        not isinstance(contract, dict)
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != SCHEMA
    ):
        raise ValueError("Unsupported cache contract schema")
    value = contract.get("inference", {})
    if not isinstance(value, dict):
        raise ValueError("Cache inference dependencies must be an object")
    rebuilt = make_contract(
        image_id=value.get("image_id"), image_sha256=value.get("image_sha256"),
        model=value.get("model"), preprocessing=value.get("preprocessing"),
        categories=value.get("categories"), prompts=value.get("prompts"),
        inference_settings=value.get("settings"), supervision=contract.get("supervision"),
        query_categories=value.get("query_categories"), task=value.get("task"),
    )
    if rebuilt != contract:
        raise ValueError("Cache contract content or keys changed")
    for key, expected in (("image_id", image_id), ("image_sha256", image_sha256), ("model", model)):
        if expected is not None and value[key] != expected:
            raise ValueError(f"Cache contract {key} mismatch")
    return rebuilt


def _prediction_payload(payload, contract):
    from .data import validate_prediction
    inference = contract["inference"]
    if inference["task"] != "detection":
        raise ValueError("Not a detection-prediction contract")
    if not isinstance(payload, dict) or payload.get("image_id") != inference["image_id"]:
        raise ValueError("Prediction payload image identity mismatch")
    if not inference["preprocessing"].get("box_frame") or payload.get("box_frame") != inference["preprocessing"]["box_frame"]:
        raise ValueError("Explicit prediction coordinate frame required")
    if not isinstance(payload.get("detections"), list):
        raise ValueError("Prediction detections must be a list")
    for prediction in payload["detections"]:
        validate_prediction(prediction)
        if prediction["label"] not in inference["categories"]:
            raise ValueError("Prediction category is outside the contract")


def write_prediction_cache(path, contract, payload):
    """Publish one new raw-prediction artifact, never overwrite old predictions."""
    from .persistence import atomic_json
    validate_contract(contract)
    _prediction_payload(payload, contract)
    atomic_json(path, dict(schema_version=SCHEMA, contract=contract, payload=payload,
                           payload_sha256=fingerprint(payload)), overwrite=False)


def read_prediction_cache(path, expected_contract):
    """Reuse identical raw inference; flag changed supervision for reevaluation."""
    validate_contract(expected_contract)
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA:
        raise ValueError("Not a v2 prediction cache; use its explicit legacy reader")
    actual = validate_contract(value.get("contract"))
    if actual["inference_key"] != expected_contract["inference_key"]:
        raise ValueError("Prediction inference dependencies changed")
    payload = value.get("payload")
    _prediction_payload(payload, actual)
    if fingerprint(payload) != value.get("payload_sha256"):
        raise ValueError("Prediction payload integrity mismatch")
    return payload, dict(schema_version=SCHEMA, inference_reused=True,
                         supervision_matches=actual["context_key"] == expected_contract["context_key"],
                         evaluation_recomputation_required=actual["context_key"] != expected_contract["context_key"])


def evaluation_fingerprint(contract, protocol):
    """Bind an evaluation artifact to both inference/context and explicit metric rules."""
    validate_contract(contract)
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("Explicit evaluation protocol required")
    return fingerprint(dict(schema_version=SCHEMA, full_key=contract["full_key"], protocol=protocol))
