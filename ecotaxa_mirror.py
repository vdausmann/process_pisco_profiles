#!/usr/bin/env python3
"""Mirror projects from one EcoTaxa instance into another.

For each selected project on the source: run a backup export, download the
zip onto the shared storage, and import it into the target as a new project.
Backup exports are re-importable by design and carry annotation *status*, so
a project's validated labels arrive as validated -- which is what makes a
mirrored project usable as training data.

Selection is by minimum validated percentage and (optionally) instrument, or
an explicit id list. Subsets are copied as well as their parents on purpose:
EcoTaxa subsets are copies, so validation done inside one does not propagate
to the parent, and the subset may be where the annotation work actually lives.

Resumable: a zip already on disk is not downloaded again, and a project
recorded in ``mirror.json`` is not imported again.

Usage:
    ecotaxa_mirror.py --source-credentials /opt/pisco/etc/ecotaxa-online.json \\
        --staging /mnt/pisco/ecotaxa-imports/online \\
        --min-validated 1 --instrument "Other camera"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from ecotaxa_import import (DEFAULT_CREDENTIALS, DEFAULT_LOAD_AREA_PREFIX,
                            DEFAULT_SHARED_ROOT, api, load_credentials,
                            to_load_area_path, wait_for_job)


def download(base, token, job_id, dest: Path, chunk=8 * 1024 * 1024):
    request = urllib.request.Request(f"{base.rstrip('/')}/api/jobs/{job_id}/file")
    request.add_header("Authorization", "Bearer " + token)
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=3600) as response, open(part, "wb") as fh:
        while True:
            block = response.read(chunk)
            if not block:
                break
            fh.write(block)
    part.rename(dest)
    return dest.stat().st_size


def select_projects(base, token, args) -> list[dict]:
    projects = api(base, "/projects/search?not_granted=false&also_others=false&title_filter=",
                   None, token)
    if args.project_ids:
        wanted = {int(x) for x in args.project_ids.split(",") if x.strip()}
        chosen = [p for p in projects if p["projid"] in wanted]
    else:
        chosen = [p for p in projects
                  if (p.get("pctvalidated") or 0) > args.min_validated
                  and (not args.instrument or p.get("instrument") == args.instrument)]
    # Smallest first: early feedback, and a failure late in the run costs less.
    return sorted(chosen, key=lambda p: p.get("objcount") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-credentials", required=True,
                    help='JSON with "host", "username"/"user", "password" of the source')
    ap.add_argument("--target-url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--target-credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--staging", required=True,
                    help="directory on the share for the downloaded zips and mirror.json")
    ap.add_argument("--min-validated", type=float, default=1.0,
                    help="copy projects with more than this percent validated")
    ap.add_argument("--instrument", default=None,
                    help="only projects with this instrument, e.g. 'Other camera'")
    ap.add_argument("--project-ids", default=None,
                    help="comma-separated ids; overrides the filters")
    ap.add_argument("--instrument-on-target", default="Other camera")
    ap.add_argument("--shared-root", default=DEFAULT_SHARED_ROOT)
    ap.add_argument("--load-area-prefix", default=DEFAULT_LOAD_AREA_PREFIX)
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied")
    args = ap.parse_args()

    src = json.loads(Path(args.source_credentials).read_text())
    src_base = src["host"].replace("/api", "").rstrip("/")
    src_user = src.get("username") or src.get("user")
    src_token = api(src_base, "/login", {"username": src_user, "password": src["password"]})
    print(f"source: {src_base} as {src_user}")

    projects = select_projects(src_base, src_token, args)
    total = sum(int(p.get("objcount") or 0) for p in projects)
    print(f"{len(projects)} projects selected, {total:,} objects "
          f"(~{total * 2 / 1e6:.1f} GB at 2 KB/object)")
    for p in projects:
        print(f"  {p['projid']:>6}  {int(p.get('objcount') or 0):>8}  "
              f"{p.get('pctvalidated') or 0:>5.1f}%  {p.get('title')}")
    if args.dry_run:
        return 0

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    state_path = staging / "mirror.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    tgt_user, tgt_password = load_credentials(args.target_credentials, None, None)
    tgt_token = api(args.target_url, "/login", {"username": tgt_user, "password": tgt_password})
    print(f"target: {args.target_url} as {tgt_user}\n")

    failures = 0
    for p in projects:
        pid = str(p["projid"])
        title = p.get("title") or pid
        record = state.get(pid, {})
        if record.get("target_project_id"):
            print(f"[{pid}] already mirrored as target project {record['target_project_id']}, skipping")
            continue
        print(f"[{pid}] {title}  ({int(p.get('objcount') or 0):,} objects)")
        zip_path = staging / f"{pid}.zip"
        try:
            if zip_path.exists():
                print(f"  zip present ({zip_path.stat().st_size / 1e6:.1f} MB), not re-exporting")
            else:
                t0 = time.time()
                rsp = api(src_base, "/object_set/export/backup",
                          {"filters": {}, "request": {"project_id": int(pid), "out_to_ftp": False}},
                          src_token)
                job = wait_for_job(src_base, src_token, rsp["job_id"])
                if job.get("state") != "F":
                    raise RuntimeError(f"export job ended in state {job.get('state')}: "
                                       f"{job.get('progress_msg')}")
                size = download(src_base, src_token, rsp["job_id"], zip_path)
                print(f"  exported and downloaded {size / 1e6:.1f} MB in {time.time() - t0:.0f}s")

            # Re-login: a long export can outlive the token.
            tgt_token = api(args.target_url, "/login",
                            {"username": tgt_user, "password": tgt_password})
            target_title = f"[online {pid}] {title}"
            target_id = api(args.target_url, "/projects/create", {
                "title": target_title, "instrument": args.instrument_on_target, "visible": True,
            }, tgt_token)
            source_path = to_load_area_path(zip_path, args.shared_root, args.load_area_prefix)
            rsp = api(args.target_url, f"/file_import/{target_id}", {
                "source_path": source_path, "skip_loaded_files": False,
                "skip_existing_objects": False, "update_mode": "",
            }, tgt_token)
            if rsp.get("errors"):
                raise RuntimeError(f"import refused: {rsp['errors']}")
            job = wait_for_job(args.target_url, tgt_token, rsp["job_id"], answer_questions=True)
            if job.get("state") != "F":
                detail = job.get("question") or job.get("errors") or job.get("progress_msg")
                raise RuntimeError(f"import job ended in state {job.get('state')}: {detail}")
            rows = (job.get("result") or {}).get("rowcount") or 0
            state[pid] = {"title": title, "target_project_id": target_id,
                          "target_title": target_title, "imported_rows": rows,
                          "source_objcount": p.get("objcount"),
                          "source_pctvalidated": p.get("pctvalidated")}
            print(f"  imported {rows:,} objects as target project {target_id}")
        except Exception as exc:  # keep going; the state file says what is left
            failures += 1
            state[pid] = {"title": title, "error": str(exc)[:500]}
            print(f"  FAILED: {exc}", file=sys.stderr)
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    done = sum(1 for r in state.values() if r.get("target_project_id"))
    print(f"\n{done} mirrored, {failures} failed. State in {state_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
