#!/usr/bin/env python3
"""Fetch validated labels out of EcoTaxa, for retraining the dual ViT.

Writes ``labels.csv`` -- one row per validated object, giving the crop's
identity and the taxon a human confirmed. That file is the only thing the
training set builder needs from EcoTaxa, which is what lets the rest of the
retraining chain run on a GPU worker that has no EcoTaxa credentials.

Runs on the EcoTaxa machine, like the import and feature tools, because
credentials live there rather than in task settings (settings are rendered
onto shared storage in plain text).

**Only validated objects are fetched** (``--status V``). Predicted objects
carry this pipeline's own output; training on those would teach the model its
own mistakes. Note that the status filter belongs at the top level of the
query body -- wrapping it in ``{"filters": ...}`` is silently ignored and
returns *every* object in the project, predictions included.

Projects are given in priority order and the **last one wins** on a duplicate
crop. EcoTaxa subsets are copies, so the same physical crop appears in a
project and in every subset drawn from it; letting the later project win means
a re-validation in a subset supersedes the original call.

Usage:
    ecotaxa_fetch_labels.py --projects 23022 23052 23196 -o out/
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ecotaxa_import is stdlib-only; reusing its client keeps one implementation of
# login, error reporting and the credentials file layout.
from ecotaxa_import import DEFAULT_CREDENTIALS, api, load_credentials  # noqa: E402

FIELDS = "obj.orig_id,img.orig_file_name,txo.display_name"


def short(taxon: str | None) -> str:
    """EcoTaxa display names carry lineage ("Copepoda<Multicrustacea")."""
    return (taxon or "").split("<")[0].strip()


def fetch(url: str, token: str, project_id: int, status: str) -> list[tuple[str, str, str]]:
    """Return [(orig_id, file_name, label)] for one project.

    ``fields`` is a query parameter and the values come back in ``details``, in
    the same order as ``object_ids`` -- one call for the whole project.
    """
    found = api(url, f"/object_set/{project_id}/query?fields={FIELDS}",
                {"statusfilter": status} if status else {}, token)
    details = found.get("details") or []
    object_ids = found.get("object_ids") or []
    if len(details) != len(object_ids):
        raise SystemExit(
            f"ERROR: project {project_id} returned {len(object_ids)} objects but "
            f"{len(details)} detail rows; this backend does not support `fields`.")
    rows = []
    for detail in details:
        orig_id, file_name, taxon = (list(detail) + [None, None, None])[:3]
        rows.append((str(orig_id or ""), str(file_name or ""), short(taxon)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--projects", type=int, nargs="+", required=True,
                    help="EcoTaxa project ids, in priority order (last wins on duplicates)")
    ap.add_argument("-o", "--output", required=True, help="directory for labels.csv")
    ap.add_argument("--status", default="V",
                    help="EcoTaxa status filter: V validated, P predicted, "
                         "D dubious, empty for all (default: V)")
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--credentials", default=os.environ.get(
        "ECOTAXA_CREDENTIALS", DEFAULT_CREDENTIALS))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    args = ap.parse_args()

    user, password = load_credentials(args.credentials, args.user, args.password)
    if not user or not password:
        print(f"ERROR: no credentials. Put {{\"user\": ..., \"password\": ...}} in "
              f"{args.credentials}, or pass --user/--password.", file=sys.stderr)
        return 1
    token = api(args.url, "/login", {"username": user, "password": password})

    # orig_id -> row. A later project overwrites an earlier one, so subsets
    # listed last supersede the project they were drawn from.
    labels: dict[str, dict] = {}
    per_project = {}
    for project_id in args.projects:
        rows = fetch(args.url, token, project_id, args.status)
        kept = 0
        for orig_id, file_name, label in rows:
            if not label:
                continue           # validated but unclassified should not exist
            key = orig_id or file_name
            if not key:
                continue
            labels[key] = {"object_id": orig_id, "file_name": file_name,
                           "label": label, "project_id": project_id}
            kept += 1
        per_project[project_id] = kept
        print(f"  project {project_id}: {len(rows)} objects with status "
              f"'{args.status or 'any'}', {kept} labelled")

    if not labels:
        print(f"ERROR: no objects with status '{args.status}' in "
              f"{args.projects}. Validate some objects in EcoTaxa first.",
              file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["object_id", "file_name", "label", "project_id"])
        writer.writeheader()
        for key in sorted(labels):
            writer.writerow(labels[key])

    counts = collections.Counter(row["label"] for row in labels.values())
    (out_dir / "labels.json").write_text(json.dumps(
        {"projects": args.projects, "status": args.status,
         "per_project": per_project, "n": len(labels),
         "classes": dict(counts.most_common())}, indent=1))

    print(f"\n{labels_path}: {len(labels)} distinct crops after deduplication")
    for label, n in counts.most_common():
        print(f"    {label:32s} {n:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
