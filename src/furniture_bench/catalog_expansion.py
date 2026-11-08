"""Budgeted, resumable licensed image acquisition and incremental feature storage.

This module does not change application startup or its production catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .data import digest, read_json


class BudgetExceeded(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def validate_source(source):
    if not all(source.get(k) for k in ["name", "license", "license_evidence", "attribution"]):
        raise ValueError("Source requires license evidence and attribution")
    if not all(
        source.get(k) is True
        for k in ["commercial_training", "commercial_evaluation", "retain_images"]
    ):
        raise ValueError("Explicit intended-use permissions required")
    if not source.get("allowed_hosts"):
        raise ValueError("Approved source hosts required")
    return source


class DownloadStore:
    """Single-writer ledger. Limits count failed/partial transfers as well as successes."""

    def __init__(self, root, source, max_files=4000, max_bytes=1_500_000_000, opener=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = validate_source(source)
        self.max_files, self.max_bytes = max_files, max_bytes
        self.opener = opener or urllib.request.urlopen
        self.path = self.root / "download-ledger.json"
        self.state = (
            read_json(self.path) if self.path.exists() else {"entries": {}, "transferred_bytes": 0}
        )
        if self.state["transferred_bytes"] > max_bytes or len(self.state["entries"]) > max_files:
            raise BudgetExceeded("Existing ledger exceeds supplied limits")

    def save(self):
        atomic_json(self.path, self.state)

    def fetch(self, image_id, url):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in self.source["allowed_hosts"]:
            raise ValueError("URL outside approved licensed source")
        key = hashlib.sha256(url.encode()).hexdigest()
        entries = self.state["entries"]
        if key not in entries:
            if len(entries) >= self.max_files:
                raise BudgetExceeded("Image-file limit reached")
            entries[key] = {"image_id": image_id, "url": url, "status": "pending", "attempts": 0}
            self.save()
        row = entries[key]
        if row["image_id"] != image_id:
            raise ValueError("Same source URL supplied with different image ID")
        destination = self.root / "images" / (key[:24] + Path(parsed.path).suffix)
        destination.parent.mkdir(exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        if row["status"] == "complete":
            if not destination.is_file() or digest(destination) != row["sha256"]:
                raise ValueError("Completed source file changed; refusing silent overwrite")
            return {**row, "file": str(destination.resolve()), "cached": True}
        remaining = self.max_bytes - self.state["transferred_bytes"]
        if remaining <= 0:
            raise BudgetExceeded("Image-byte limit reached")
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "FurnitureBench-LicensedCatalog/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
            if row.get("etag"):
                headers["If-Range"] = row["etag"]
        row.update(status="downloading", attempts=row["attempts"] + 1)
        self.save()
        try:
            with self.opener(urllib.request.Request(url, headers=headers), timeout=40) as response:
                code = response.getcode()
                if (
                    response.geturl()
                    and urlsplit(response.geturl()).hostname not in self.source["allowed_hosts"]
                ):
                    raise ValueError("Redirect outside approved source")
                if start and code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise ValueError("Invalid resumed byte range")
                    mode = "ab"
                    row["resumed_from_bytes"] = start
                elif code == 200:
                    mode = "wb"
                    if start:
                        row["server_restarted_full_transfer"] = True
                else:
                    raise ValueError(f"Unexpected response status {code}")
                length = response.headers.get("Content-Length")
                length = int(length) if length else None
                if length is not None and length > remaining:
                    raise BudgetExceeded("Object exceeds remaining image-byte budget")
                row["etag"] = response.headers.get("ETag")
                received = 0
                with part.open(mode) as output:
                    while length is None or received < length:
                        allowance = self.max_bytes - self.state["transferred_bytes"]
                        if allowance <= 0:
                            raise BudgetExceeded("Transfer reached image-byte budget")
                        amount = min(
                            65536, allowance, length - received if length is not None else allowance
                        )
                        # Reserve before I/O so a hard process interruption cannot refund
                        # already consumed bytes. Unknown interrupted reads are conservative.
                        self.state["transferred_bytes"] += amount
                        self.save()
                        block = response.read(amount)
                        self.state["transferred_bytes"] -= amount - len(block)
                        self.save()
                        if not block:
                            break
                        output.write(block)
                        output.flush()
                        received += len(block)
                if length is not None and received != length:
                    raise OSError("Incomplete HTTP response; partial file retained")
            with Image.open(part) as image:
                image.load()
                dimensions = list(image.size)
            sha = digest(part)
            os.replace(part, destination)
            row.update(
                status="complete",
                file=str(destination.resolve()),
                sha256=sha,
                dimensions=dimensions,
                bytes=destination.stat().st_size,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
            )
            row.pop("error", None)
            self.save()
            return {**row, "cached": False}
        except Exception as exc:
            row.update(status="partial_or_failed", error=f"{type(exc).__name__}: {exc}")
            self.save()
            raise


def validate_catalog(catalog, root="."):
    ids = [p["product_id"] for p in catalog["products"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Product identities must be unique")
    for product in catalog["products"]:
        if product.get("rights", {}).get("allowed_for_benchmark") is not True:
            raise ValueError("Product has no evaluation rights")
        if not product.get("rights", {}).get("license"):
            raise ValueError("Product license missing")
        if not product.get("images"):
            raise ValueError("Product must have reference images")
        image_ids = [i["image_id"] for i in product["images"]]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate image ID within product")
        for image in product["images"]:
            if digest(Path(root) / image["file"]) != image["sha256"]:
                raise ValueError("Reference file hash mismatch")
    return {"products": len(ids), "images": sum(len(p["images"]) for p in catalog["products"])}


def incremental_embeddings(rows, directory, model_signature, encoder, batch_size=8):
    """Per-image checkpoints bind identity, original bytes and exact model configuration."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors, reused, seconds = [], 0, 0.0
    pending = []
    paths = []
    for i, row in enumerate(rows):
        if digest(row["file"]) != row["sha256"]:
            raise ValueError("Source bytes changed before embedding")
        key = hashlib.sha256(
            json.dumps([row["image_id"], row["sha256"], model_signature], sort_keys=True).encode()
        ).hexdigest()
        path = directory / (key + ".npy")
        paths.append(path)
        vectors.append(None)
        if path.exists():
            value = np.load(path, allow_pickle=False)
            if (
                value.ndim != 1
                or not np.isfinite(value).all()
                or not np.isclose(np.linalg.norm(value), 1, atol=1e-4)
            ):
                raise ValueError("Invalid cached feature")
            vectors[i] = value
            reused += 1
        else:
            pending.append(i)
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        values, elapsed = encoder([rows[i]["file"] for i in indices])
        values = np.asarray(values, dtype=np.float32)
        if (
            values.ndim != 2
            or len(values) != len(indices)
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Encoder returned invalid or non-unit features")
        seconds += elapsed
        for i, vector in zip(indices, values):
            temporary = paths[i].with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vector)
            os.replace(temporary, paths[i])
            vectors[i] = vector
    if not rows:
        raise ValueError("No image rows")
    return np.stack(vectors), {
        "reused": reused,
        "new_encodings": len(pending),
        "inference_seconds": seconds,
    }


def validate_nested_galleries(galleries, queries):
    known = {q["product_id"] for q in queries if q["true_product_in_catalog"]}
    unknown = {q["product_id"] for q in queries if not q["true_product_in_catalog"]}
    if known & unknown:
        raise ValueError("Known/unknown identity overlap")
    query_images = {q["image_id"] for q in queries}
    query_hashes = {q["sha256"] for q in queries}
    if len(query_images) != len(queries):
        raise ValueError("Duplicate query image ID")
    previous = set()
    for size, products in sorted(galleries.items(), key=lambda pair: int(pair[0])):
        ids = {p["product_id"] for p in products}
        if len(ids) != int(size) or len(ids) != len(products):
            raise ValueError("Gallery size/identity mismatch")
        if not previous <= ids or not known <= ids or unknown & ids:
            raise ValueError("Non-nested gallery, missing known identity, or unknown identity leak")
        for product in products:
            if any(
                i["image_id"] in query_images or i["sha256"] in query_hashes
                for i in product["images"]
            ):
                raise ValueError("Query/reference image leakage")
        previous = ids
    return {"known_query_identities": len(known), "unknown_query_identities": len(unknown)}
