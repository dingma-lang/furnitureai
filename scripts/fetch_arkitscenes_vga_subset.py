"""Fetch a bounded VGA subset from the approved scene, using HTTP ZIP ranges.

No complete VGA archive, models, trajectories or reference geometry are downloaded.
"""
import argparse
import io
import json
import struct
import subprocess
import tempfile
import zipfile
import zlib
from pathlib import Path

from furniture_bench.data import digest, write_json

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42444946/"


class RangeZip:
    def __init__(self, size, cache, limit=32_000_000):
        self.size, self.cache, self.limit, self.bytes = size, Path(cache), limit, 0
        tail = self.fetch(size - 65557, size - 1)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0:
            raise ValueError("ZIP end record unavailable")
        end = struct.unpack_from("<4s4H2LH", tail, position)
        directory = self.fetch(end[6], end[6] + end[5] - 1)
        self.entries = {}
        offset = 0
        while offset < len(directory):
            fields = struct.unpack_from("<4s6H3I5H2I", directory, offset)
            if fields[0] != b"PK\x01\x02":
                raise ValueError("invalid central directory")
            name = directory[offset + 46:offset + 46 + fields[10]].decode()
            self.entries[name] = {"crc": fields[7], "compressed": fields[8], "size": fields[9], "offset": fields[16], "method": fields[4]}
            offset += 46 + fields[10] + fields[11] + fields[12]

    def fetch(self, first, last):
        count = last - first + 1
        if first < 0 or count <= 0 or self.bytes + count > self.limit:
            raise ValueError("bounded subset download limit exceeded")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bytes"
            subprocess.run(["curl", "-fsSL", "--range", f"{first}-{last}", "--max-filesize", str(count), "--max-time", "45", "-o", str(output), BASE + "vga_wide.zip"], check=True)
            data = output.read_bytes()
        if len(data) != count:
            raise ValueError("server did not honor exact byte range")
        self.bytes += len(data)
        return data

    def read(self, name):
        e = self.entries[name]
        header = self.fetch(e["offset"], e["offset"] + 29)
        values = struct.unpack("<4s5H3I2H", header)
        if values[0] != b"PK\x03\x04":
            raise ValueError("invalid local ZIP header")
        start = e["offset"] + 30 + values[9] + values[10]
        raw = self.fetch(start, start + e["compressed"] - 1)
        data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
        if e["method"] not in (0, 8) or len(data) != e["size"] or zlib.crc32(data) != e["crc"]:
            raise ValueError("ZIP entry failed method/size/CRC validation")
        return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise ValueError("preserve previous subset; select new output")
    output.mkdir(parents=True)
    archive = RangeZip(986058175, output)
    # Eight scene-spanning anchors, three predeclared temporal intervals.
    indices = sorted({i + delta for i in range(240, 5281, 720) for delta in (0, 12, 30, 180)})
    rows = {r["source_rgb_index"]: r for r in json.load(open(Path(a.data) / "tracking-usable.json"))["images"]}
    names = {float(Path(n).stem.split("_")[-1]): n for n in archive.entries if n.endswith(".png")}
    calibration = output / "vga_wide_intrinsics.zip"
    subprocess.run(["curl", "-fsSL", "--max-filesize", "600000", "--max-time", "45", "-o", str(calibration), BASE + calibration.name], check=True)
    records = []
    with zipfile.ZipFile(calibration) as intrinsics:
        k_names = {float(Path(n).stem.split("_")[-1]): n for n in intrinsics.namelist() if n.endswith(".pincam")}
        for index in indices:
            row = rows[index]
            time = min(names, key=lambda t: abs(t - row["timestamp"]))
            if abs(time - row["timestamp"]) > 0.02 or time not in k_names:
                raise ValueError("VGA synchronization/calibration unavailable")
            path = output / f"vga-{index:05d}.png"
            path.write_bytes(archive.read(names[time]))
            k_path = path.with_suffix(".pincam")
            k_path.write_bytes(intrinsics.read(k_names[time]))
            records.append({"source_rgb_index": index, "lowres_image_id": row["image_id"], "lowres_timestamp": row["timestamp"], "vga_timestamp": time, "file": path.name, "sha256": digest(path), "intrinsics_file": k_path.name})
    write_json(output / "receipt.json", {"scene_id": "42444946", "research_only": True, "source_url": BASE, "archive_bytes_not_downloaded_in_full": archive.size, "downloaded_bytes": archive.bytes + calibration.stat().st_size, "anchors": list(range(240, 5281, 720)), "gaps_raw_indices": [12, 30, 180], "records": records})
    print(f"Fetched {len(records)} frames; {archive.bytes + calibration.stat().st_size} bytes")


if __name__ == "__main__":
    main()
