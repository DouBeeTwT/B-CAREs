#!/usr/bin/env python3
"""Verify the downloaded model weights against weights/manifest.json.

    python scripts/verify_weights.py                 # check what is on disk
    python scripts/verify_weights.py --compute       # print hashes to fill the manifest
    python scripts/verify_weights.py --json          # machine-readable output

Exit code 0 = every weight file is present and matches the recorded size/hash,
1 = something is missing or differs, 2 = the manifest itself is unusable.

A `null` size or hash in the manifest means "not recorded yet"; the file is then
only checked for existence, and a warning is printed so that this cannot be
mistaken for a verified download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "weights", "manifest.json")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str) -> list:
    if not os.path.isfile(path):
        raise SystemExit("manifest not found: {}".format(path))
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("weights", payload if isinstance(payload, list) else [])
    if not entries:
        raise SystemExit("manifest contains no weights: {}".format(path))
    return entries


def check(entry: dict) -> dict:
    path = entry.get("target_path", "")
    result = {"task": entry.get("task", "?"), "path": path, "status": "ok", "message": ""}
    if not os.path.isfile(path):
        result.update(status="missing",
                      message="not found; download from {}".format(entry.get("url", "manifest")))
        return result
    size = os.path.getsize(path)
    result["size_bytes"] = size
    if entry.get("size_bytes") and size != entry["size_bytes"]:
        result.update(status="size_mismatch",
                      message="expected {} bytes, got {}".format(entry["size_bytes"], size))
        return result
    actual = sha256(path)
    result["sha256"] = actual
    expected = entry.get("sha256")
    if expected and expected.lower() != actual.lower():
        result.update(status="hash_mismatch",
                      message="expected {}, got {}".format(expected, actual))
    elif not expected:
        result.update(status="unverified", message="size/hash not recorded in the manifest")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify B-CAREs model weights.")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--compute", action="store_true",
                        help="print size and sha256 of files that exist (to fill the manifest)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    entries = load_manifest(args.manifest)
    results = [check(entry) for entry in entries]

    if args.compute:
        for entry, result in zip(entries, results):
            if result["status"] == "ok":
                result["status"] = "ok"
        bad = [r for r in results if r["status"] == "missing"]
        for result in results:
            if result["status"] != "missing":
                print('{task}: size_bytes={size_bytes} sha256={sha256}'.format(**result))
        if bad:
            print("cannot compute: {} missing file(s)".format(len(bad)), file=sys.stderr)
            return 1
        return 0

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        width = max(len(r["task"]) for r in results)
        for result in results:
            print("{:<{w}}  {:<12} {}".format(result["task"], result["status"],
                                             result["message"], w=width))

    failed = [r for r in results if r["status"] not in {"ok"}]
    warnings = [r for r in results if r["status"] == "unverified"]
    if failed or warnings:
        print("\n{} file(s) missing or mismatched, {} unverified.".format(len(failed), len(warnings)))
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
