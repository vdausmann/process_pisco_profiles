#!/usr/bin/env python3
"""Write ViT predictions onto EcoTaxa objects as automatic classifications.

Takes a ``predictions.csv`` from ``vit_classify_hdf5.py``, translates its model
labels into EcoTaxa taxon ids through a versioned mapping (see
``ecotaxa_taxonomy_map.py``), and posts them to
``POST /object_set/classify_auto_multiple``.

This is the *re-prediction* route: it works on a project that already exists,
repeatably, without re-importing. Getting predictions in during the first
import instead is the exporter's job -- it writes the annotation columns of the
TSV -- and avoids a second trip to EcoTaxa altogether. Use this one when the
objects are already there.

Labels marked ``skip`` in the mapping (the binary stage's "living" gate) are
not classifications and are never written; objects whose top-1 is such a label
are left alone.

Usage:
    ecotaxa_write_predictions.py stage2/predictions.csv --project-id 7 \\
        --taxonomy-map ecotaxa_taxonomy_map.json \\
        --url http://10.11.11.114:8088 --user X --password Y
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ecotaxa_taxonomy_map import load_map

# The API asserts scores are floats in [0, 1] and rejects the whole call
# otherwise, so clamp rather than let one rounding artefact fail the batch.
def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def api(base: str, path: str, data=None, token: str | None = None, timeout: int = 300):
    request = urllib.request.Request(
        base.rstrip("/") + "/api" + path,
        method="POST" if data is not None else "GET",
        data=json.dumps(data).encode() if data is not None else None)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise SystemExit(f"ERROR: {path} returned HTTP {exc.code}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", help="predictions.csv, or the stage's output directory")
    ap.add_argument("--project-id", type=int, required=True)
    ap.add_argument("--taxonomy-map", required=True)
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    ap.add_argument("--top-k", type=int, default=1,
                    help="ranked labels to record per object (default: 1)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="skip objects whose top-1 probability is below this")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    predictions_path = Path(args.predictions)
    if predictions_path.is_dir():
        predictions_path = predictions_path / "predictions.csv"
    if not predictions_path.exists():
        print(f"ERROR: {predictions_path} not found", file=sys.stderr)
        return 1

    # Refuses a map with unresolved labels rather than dropping those objects.
    mapping = load_map(args.taxonomy_map)
    skipped_labels = {
        label for label, entry in
        json.loads(Path(args.taxonomy_map).read_text())["labels"].items()
        if entry.get("skip")}
    print(f"taxonomy map: {len(mapping)} labels -> taxon ids"
          + (f", {len(skipped_labels)} skipped" if skipped_labels else ""))

    rows = list(csv.DictReader(open(predictions_path, newline="", encoding="utf-8")))
    print(f"predictions : {len(rows)} objects from {predictions_path.name}")

    if not args.user or not args.password:
        print("ERROR: --user/--password (or ECOTAXA_USER/ECOTAXA_PASSWORD) required",
              file=sys.stderr)
        return 1
    token = api(args.url, "/login", {"username": args.user, "password": args.password})

    # orig_id -> objid in one call. `fields` is a query parameter and the
    # matching values come back in `details`, in request order. Going through
    # the API rather than the database means this also works against an
    # instance we only have API access to.
    found = api(args.url,
                f"/object_set/{args.project_id}/query?fields=obj.orig_id",
                {"filters": {}}, token)
    by_orig: dict[str, int] = {}
    details = found.get("details") or []
    if len(details) == len(found["object_ids"]):
        for objid, detail in zip(found["object_ids"], details):
            by_orig[str(detail[0])] = objid
    else:
        # Older backends ignore `fields`; fall back to one call per object.
        for objid in found["object_ids"]:
            by_orig[api(args.url, f"/object/{objid}", None, token)["orig_id"]] = objid
    print(f"project {args.project_id}: {len(by_orig)} objects")

    targets: list[int] = []
    classifications: list[list[int]] = []
    scores: list[list[float]] = []
    n_unmapped = n_gate = n_low = n_unknown = 0

    for row in rows:
        objid = by_orig.get(row["object_id"])
        if objid is None:
            n_unknown += 1
            continue
        ids, probs = [], []
        for rank in range(1, args.top_k + 1):
            label = (row.get(f"top{rank}") or "").strip()
            if not label:
                break
            if label in skipped_labels:
                if rank == 1:
                    n_gate += 1
                break
            taxon_id = mapping.get(label)
            if taxon_id is None:
                if rank == 1:
                    n_unmapped += 1
                break
            probability = _clamp(float(row.get(f"prob{rank}") or 0.0))
            if rank == 1 and probability < args.min_score:
                n_low += 1
                break
            ids.append(taxon_id)
            probs.append(probability)
        if ids:
            targets.append(objid)
            classifications.append(ids)
            scores.append(probs)

    print(f"to write    : {len(targets)}"
          + (f"  (skipped: {n_gate} gate, {n_unmapped} unmapped, "
             f"{n_low} below min-score, {n_unknown} not in project)"
             if (n_gate or n_unmapped or n_low or n_unknown) else ""))
    if not targets:
        print("ERROR: nothing to write", file=sys.stderr)
        return 1
    if args.dry_run:
        for objid, ids, probs in list(zip(targets, classifications, scores))[:5]:
            print(f"    obj {objid} -> {ids} @ {[round(p, 3) for p in probs]}")
        print("dry run, nothing written")
        return 0

    written = 0
    for start in range(0, len(targets), args.batch_size):
        end = start + args.batch_size
        written += api(args.url, "/object_set/classify_auto_multiple", {
            "target_ids": targets[start:end],
            "classifications": classifications[start:end],
            "scores": scores[start:end],
            "keep_log": True,
        }, token) or 0
        print(f"  wrote {min(end, len(targets))}/{len(targets)}")
    print(f"updated {written} objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
