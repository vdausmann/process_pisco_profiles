#!/usr/bin/env python3
"""Run an EcoTaxa prediction for its deep features, then put our own labels back.

EcoTaxa's "sort by similarity" uses deep features that only the server can
compute, with the project's CNN network. Nothing uploads feature vectors. The
features are computed by a prediction job, and that job overwrites every
predicted (P) object with EcoTaxa's own random-forest label, so the ViT's
labels go. This keeps the features and brings the labels back:

    # 1. record what is there now (do this BEFORE predicting)
    ecotaxa_restore_predictions.py snapshot --project-id 23286 -o snap.tsv

    # 2. predict P objects, learning from the project's own validated ones;
    #    waits for the job, which computes the deep features as a side effect
    ecotaxa_restore_predictions.py predict --project-id 23286

    # 3. write the snapshot's labels back as automatic classifications
    ecotaxa_restore_predictions.py restore snap.tsv --project-id 23286 --dry-run
    ecotaxa_restore_predictions.py restore snap.tsv --project-id 23286

`restore` only touches objects that were P in the snapshot *and* are still P
now. It never touches anything validated, marked dubious, or validated in the
meantime. The written score is the ViT probability (the `prob_1` free column by
default), not the 1.0 that the TSV import put in.

Credentials come from --user/--password, ECOTAXA_USER/ECOTAXA_PASSWORD, or the
"ecotaxa" block of the pipeline config (--config).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("process_pisco_profiles.config.json")
DEFAULT_URL = "https://ecotaxa.obs-vlfr.fr/api"
SNAPSHOT_COLUMNS = ["objid", "orig_id", "classif_qual", "classif_id", "score"]
# Features handed to EcoTaxa's random forest. Its labels are thrown away, so
# these only need to exist in the project.
DEFAULT_FEATURES = ["area", "major_axis_len", "minor_axis_len", "mean_intensity",
                    "circularity", "elongation"]


class Api:
    def __init__(self, base: str, user: str, password: str):
        self.base = base.rstrip("/")
        if not self.base.endswith("/api"):
            self.base += "/api"
        self.token = None
        self.token = self.call("/login", {"username": user, "password": password})

    def call(self, path: str, data=None, method: str | None = None, timeout: int = 600):
        request = urllib.request.Request(
            self.base + path,
            method=method or ("POST" if data is not None else "GET"),
            data=json.dumps(data).encode() if data is not None else None)
        request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"ERROR: {path} returned HTTP {exc.code}: "
                             f"{exc.read().decode()[:800]}")


def connect(args) -> Api:
    url, user, password = args.url, args.user, args.password
    config = Path(args.config)
    if config.exists():
        block = json.loads(config.read_text()).get("ecotaxa", {})
        url = url or block.get("host")
        user = user or block.get("username")
        password = password or block.get("password")
    if not (user and password):
        raise SystemExit("ERROR: no credentials (--user/--password, ECOTAXA_USER/"
                         f"ECOTAXA_PASSWORD, or an \"ecotaxa\" block in {config})")
    return Api(url or DEFAULT_URL, user, password)


def query(api: Api, project_id: int, fields: list[str], filters: dict | None = None,
          window: int = 20000) -> list[tuple[int, list]]:
    """(objid, [field values...]) for every object matching the filters."""
    out: list[tuple[int, list]] = []
    start = 0
    while True:
        page = api.call(f"/object_set/{project_id}/query?fields={','.join(fields)}"
                        f"&window_start={start}&window_size={window}",
                        {"filters": filters or {}})
        out.extend(zip(page["object_ids"], page["details"]))
        start += len(page["object_ids"])
        if not page["object_ids"] or start >= page["total_ids"]:
            return out


def cmd_snapshot(api: Api, args) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"ERROR: {out} exists; a second snapshot taken after a prediction "
              "would record EcoTaxa's labels, not yours. --force to overwrite.",
              file=sys.stderr)
        return 1
    rows = query(api, args.project_id, ["obj.orig_id", "obj.classif_qual",
                                        "obj.classif_id", args.score_field])
    counts: dict = {}
    missing_score = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(SNAPSHOT_COLUMNS)
        for objid, (orig_id, qual, classif_id, score) in rows:
            counts[qual] = counts.get(qual, 0) + 1
            if qual == "P" and score is None:
                missing_score += 1
            writer.writerow([objid, orig_id, qual or "",
                             "" if classif_id is None else classif_id,
                             "" if score is None else score])
    print(f"snapshot: {len(rows)} objects -> {out}  "
          + "  ".join(f"{k or 'unclassified'}={v}" for k, v in sorted(counts.items(), key=str)))
    if missing_score:
        print(f"WARNING: {missing_score} P objects have no {args.score_field}; "
              "they will be restored with score 0")
    return 0


def cmd_predict(api: Api, args) -> int:
    project = api.call(f"/projects/{args.project_id}")
    if not project.get("cnn_network_id"):
        print("ERROR: the project has no deep feature extractor (CNN network) set; "
              "choose one in its settings first", file=sys.stderr)
        return 1
    free = project.get("obj_free_cols") or {}
    missing = [f for f in args.features if f not in free]
    if missing:
        print(f"ERROR: not free columns of the project: {missing}", file=sys.stderr)
        return 1
    sources = args.source_project_ids or [args.project_id]
    categories = sorted({cid for src in sources
                         for _, (cid,) in query(api, src, ["obj.classif_id"],
                                                {"statusfilter": "V"})
                         if cid is not None})
    if not categories:
        print("ERROR: no validated objects in the source project(s) to learn from",
              file=sys.stderr)
        return 1
    print(f"predicting status-P objects of {args.project_id} with CNN "
          f"'{project['cnn_network_id']}', learning from {sources} "
          f"({len(categories)} validated categories)")
    response = api.call("/object_set/predict", {
        "filters": {"statusfilter": "P"},
        "request": {
            "project_id": args.project_id,
            "source_project_ids": sources,
            "features": [f"fre.{f}" for f in args.features],
            "categories": categories,
            "use_scn": True,
            "pre_mapping": {},
        }})
    for warning in response.get("warnings") or []:
        print(f"  warning: {warning}")
    if response.get("errors"):
        print("ERROR: " + "; ".join(response["errors"]), file=sys.stderr)
        return 1
    job_id = response["job_id"]
    print(f"job {job_id} started")
    last = None
    while True:
        job = api.call(f"/jobs/{job_id}/")
        line = f"  {job.get('state')} {job.get('progress_pct')}% {job.get('progress_msg') or ''}"
        if line != last:
            print(line, flush=True)
            last = line
        if job.get("state") in ("F", "E", "A"):
            break
        time.sleep(15)
    if job["state"] != "F":
        print(f"ERROR: job ended in state {job['state']}; see /jobs/{job_id}/log",
              file=sys.stderr)
        return 1
    return 0


def cmd_restore(api: Api, args) -> int:
    with open(args.snapshot, newline="", encoding="utf-8") as fh:
        snapshot = [row for row in csv.DictReader(fh, delimiter="\t")
                    if row["classif_qual"] == "P" and row["classif_id"]]
    now = {objid: qual for objid, (qual,) in
           query(api, args.project_id, ["obj.classif_qual"])}
    targets, classifications, scores = [], [], []
    gone = changed = 0
    for row in snapshot:
        objid = int(row["objid"])
        if objid not in now:
            gone += 1
            continue
        if now[objid] != "P":
            changed += 1
            continue
        targets.append(objid)
        classifications.append([int(row["classif_id"])])
        score = float(row["score"]) if row["score"] else 0.0
        scores.append([max(0.0, min(1.0, score))])
    print(f"restore: {len(targets)} of {len(snapshot)} snapshot P objects"
          + (f"  (skipped: {changed} no longer P, {gone} no longer in project)"
             if changed or gone else ""))
    if args.dry_run:
        for objid, cid, score in list(zip(targets, classifications, scores))[:5]:
            print(f"    obj {objid} -> {cid} @ {score}")
        print("dry run, nothing written")
        return 0
    written = 0
    for start in range(0, len(targets), args.batch_size):
        end = start + args.batch_size
        written += api.call("/object_set/classify_auto_multiple", {
            "target_ids": targets[start:end],
            "classifications": classifications[start:end],
            "scores": scores[start:end],
            "keep_log": True,
        }) or 0
        print(f"  wrote {min(end, len(targets))}/{len(targets)}", flush=True)
    print(f"updated {written} objects")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL"))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    ap.add_argument("--config", default=os.environ.get("PISCO_CONFIG", DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("snapshot", help="record every object's current label and score")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--score-field", default="fre.prob_1",
                   help="field holding the classifier's probability (default: fre.prob_1)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("predict", help="run an EcoTaxa prediction on P objects and wait")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--source-project-ids", type=int, nargs="+",
                   help="projects to learn from (default: the project itself)")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)

    p = sub.add_parser("restore", help="write a snapshot's P labels back")
    p.add_argument("snapshot")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    api = connect(args)
    return {"snapshot": cmd_snapshot, "predict": cmd_predict,
            "restore": cmd_restore}[args.command](api, args)


if __name__ == "__main__":
    sys.exit(main())
