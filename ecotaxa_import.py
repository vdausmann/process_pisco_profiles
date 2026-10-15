#!/usr/bin/env python3
"""Import an exported archive into EcoTaxa, creating the project if needed.

Creates a new project by default and imports into it; pass ``--project-id`` to
add to one that already exists. Several archives (a split export) go into the
same project, in order.

Nothing is copied. The Hub's shared storage is mounted inside EcoTaxa as a
subdirectory of its server load area, so an archive sitting in a workflow's
output directory is already visible to the importer -- this only has to
translate the path (``/mnt/pisco/x`` -> ``/pisco/x``).

The project id is written to ``import.json`` in the output directory, so a
following task can attach features to the project this one just created
without a human carrying the number across.

Credentials come from a file on the EcoTaxa machine rather than from task
settings, which are rendered onto shared storage in plain text.

Usage:
    ecotaxa_import.py /mnt/pisco/.../export -o out/ --project-title "M202 028-02"
    ecotaxa_import.py /mnt/pisco/.../x.zip  -o out/ --project-id 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CREDENTIALS = "/opt/pisco/etc/ecotaxa.json"

# Where the Hub's shared storage appears inside EcoTaxa. The importer resolves
# source_path against its server load area, and the share is mounted there.
DEFAULT_SHARED_ROOT = "/mnt/pisco"
DEFAULT_LOAD_AREA_PREFIX = "/pisco"

# EcoTaxa's project access levels. "open" lets every logged-in user annotate
# without being granted anything, so accounts made later are covered too; note
# that upstream also lets them change the project's settings.
ACCESS_LEVELS = {"private": "0", "public": "1", "open": "2"}

# Job states: Finished, Error, Asking (a question the import needs answered).
TERMINAL_STATES = {"F", "E", "A"}


def api(base, path, data=None, token=None, timeout=600, method=None):
    request = urllib.request.Request(
        base.rstrip("/") + "/api" + path,
        method=method or ("POST" if data is not None else "GET"),
        data=json.dumps(data).encode() if data is not None else None)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"ERROR: {path} returned HTTP {exc.code}: "
                         f"{exc.read().decode()[:500]}")


def load_credentials(path, user, password):
    if user and password:
        return user, password
    candidate = Path(path)
    if candidate.exists():
        data = json.loads(candidate.read_text())
        return (user or data.get("user") or data.get("username"),
                password or data.get("password"))
    return user, password


def to_load_area_path(archive: Path, shared_root: str, prefix: str) -> str:
    """Translate a path on the share into one the importer understands."""
    archive = archive.resolve()
    root = Path(shared_root).resolve()
    try:
        relative = archive.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"ERROR: {archive} is not under {root}. EcoTaxa can only import "
            f"from its server load area; put the archive on the share, or set "
            f"--shared-root/--load-area-prefix to match your mounts.")
    return f"{prefix.rstrip('/')}/{relative}"


# Shown in the hover popover of the classification screen.
POPOVER_FIELDS = ["esd", "area", "prediction_score", "pressure"]


def sorting_tools(project: dict) -> tuple[str, str]:
    """The project's 'sorting tools' and popover lists, from its own mapping.

    An import maps every numeric column the export carries into a numeric
    free column (n01, n02, ...), which makes it sortable through the API -
    but the GUI's sort dropdown only offers what the project's
    classiffieldlist names, and an import leaves that empty. Both lists are
    newline-separated ``name=label`` lines keyed by the free column's *name*
    (``esd=esd``); the frontend resolves the name to its nXX column itself
    and silently drops keys it does not know. The GUI shows them sorted by
    label, so order here does not matter.
    """
    mapping = project.get("obj_free_cols") or {}
    numeric = sorted(name for name, col in mapping.items() if str(col).startswith("n"))
    sort_lines = ["depth_min=depth"] + [f"{n}={n}" for n in numeric]
    pop_lines = [f"{n}={n}" for n in POPOVER_FIELDS if n in mapping] + ["depth_min=depth"]
    return "\n".join(sort_lines), "\n".join(pop_lines)


def configure_project(base, token, project_id) -> int:
    """Make every numeric column the project holds available in the GUI."""
    project = api(base, f"/projects/{project_id}", None, token)
    sort_list, pop_list = sorting_tools(project)
    project["classiffieldlist"] = sort_list
    project["popoverfieldlist"] = pop_list
    api(base, f"/projects/{project_id}", project, token, method="PUT")
    n = sort_list.count("\n") + 1
    print(f"project {project_id}: {n} fields offered as sorting tools, "
          f"{pop_list.count(chr(10)) + 1} in the popover")
    return n


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "." for c in name.strip().lower()).strip(".")


def resolve_missing_users(base, token, names: list[str]) -> dict[str, int]:
    """Map annotator names from an export onto local accounts, creating them.

    An EcoTaxa backup carries who validated each object by *name*. Mapping
    every unknown name to the administrator would import fine but erase that
    provenance; a per-name account keeps it. The accounts cannot be logged
    into (random password nobody is told) -- they exist for attribution.
    """
    import secrets

    existing = {u["name"].strip().lower(): u["id"] for u in (api(base, "/users", None, token) or [])}
    mapping: dict[str, int] = {}
    for name in names:
        key = name.strip().lower()
        if key not in existing:
            api(base, "/users/create", {
                "id": -1,
                "name": name,
                "email": f"{_slug(name)}@mirror.local",
                "organisation": "mirrored annotator",
                "password": secrets.token_urlsafe(24),
                "usercreationreason": "created to preserve annotation provenance on import",
            }, token)
            existing = {u["name"].strip().lower(): u["id"]
                        for u in (api(base, "/users", None, token) or [])}
            print(f"    created local account for annotator {name!r} (id {existing[key]})")
        mapping[name] = existing[key]
    return mapping


def wait_for_job(base, token, job_id, poll=4, timeout=7200, answer_questions=False):
    """Block until the job ends. Returns the final job record.

    With ``answer_questions``, an import that stops to ask about unknown
    annotators is answered by creating matching local accounts, and the wait
    continues. Unknown *taxa* are not guessed at: the job is returned in its
    asking state for the caller to report.
    """
    deadline = time.time() + timeout
    last = None
    answered = set()
    while time.time() < deadline:
        job = api(base, f"/jobs/{job_id}/", None, token)
        state = job.get("state")
        progress = (state, job.get("progress_pct"), job.get("progress_msg"))
        if progress != last:
            print(f"    job {job_id}: {state} {job.get('progress_pct')}% "
                  f"{job.get('progress_msg') or ''}")
            last = progress
        if state == "A" and answer_questions:
            question = job.get("question") or {}
            missing_users = question.get("missing_users") or []
            missing_taxa = question.get("missing_taxa") or []
            key = (tuple(missing_users), tuple(missing_taxa))
            if missing_taxa or key in answered:
                return job          # not ours to guess, or we already tried
            answered.add(key)
            reply = {"users": resolve_missing_users(base, token, missing_users), "taxa": {}}
            api(base, f"/jobs/{job_id}/answer", reply, token)
            print(f"    answered: {len(missing_users)} annotator(s) mapped")
            time.sleep(poll)
            continue
        if state in TERMINAL_STATES:
            return job
        time.sleep(poll)
    raise SystemExit(f"ERROR: job {job_id} did not finish within {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive", nargs="?", default=None,
                    help="a .zip, or a directory of them (a split export)")
    ap.add_argument("-o", "--output", required=True,
                    help="directory for import.json, which records the project id")
    ap.add_argument("--project-id", type=int, default=None,
                    help="import into this existing project instead of creating one")
    ap.add_argument("--project-title", default=None,
                    help="title for the new project (default: the archive's name)")
    ap.add_argument("--instrument", default="Other camera",
                    help="must be one EcoTaxa knows; 'Other camera' is the safe generic")
    ap.add_argument("--visible", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--access", choices=sorted(ACCESS_LEVELS), default=None,
                    help="'open' lets every user annotate, 'public' lets them "
                         "view, 'private' only those granted. Default: public, "
                         "or private with --no-visible. Also applied to an "
                         "existing --project-id when given.")
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--credentials", default=os.environ.get(
        "ECOTAXA_CREDENTIALS", DEFAULT_CREDENTIALS))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    ap.add_argument("--shared-root", default=DEFAULT_SHARED_ROOT)
    ap.add_argument("--load-area-prefix", default=DEFAULT_LOAD_AREA_PREFIX)
    ap.add_argument("--update-mode", default="",
                    help="'Yes' updates data, 'Cla' also updates classification")
    ap.add_argument("--skip-existing-objects", action="store_true")
    ap.add_argument("--configure-only", action="store_true",
                    help="only (re)set the sorting tools of --project-id; import nothing")
    args = ap.parse_args()

    if args.configure_only:
        if args.project_id is None:
            print("ERROR: --configure-only needs --project-id", file=sys.stderr)
            return 1
        user, password = load_credentials(args.credentials, args.user, args.password)
        token = api(args.url, "/login", {"username": user, "password": password})
        configure_project(args.url, token, args.project_id)
        return 0

    if args.archive is None:
        ap.error("archive is required unless --configure-only")
    source = Path(args.archive)
    if source.is_dir():
        archives = sorted(source.glob("*.zip"))
        if not archives:
            print(f"ERROR: no .zip under {source}", file=sys.stderr)
            return 1
    elif source.exists():
        archives = [source]
    else:
        print(f"ERROR: {source} not found", file=sys.stderr)
        return 1
    print(f"{len(archives)} archive(s) to import")

    user, password = load_credentials(args.credentials, args.user, args.password)
    if not user or not password:
        print(f"ERROR: no credentials. Put {{\"user\": ..., \"password\": ...}} in "
              f"{args.credentials}, or pass --user/--password.", file=sys.stderr)
        return 1

    token = api(args.url, "/login", {"username": user, "password": password})

    project_id = args.project_id
    created = False
    if project_id is None:
        title = args.project_title or archives[0].stem
        access = args.access or ("public" if args.visible else "private")
        project_id = api(args.url, "/projects/create", {
            "title": title, "instrument": args.instrument,
            "visible": access != "private", "access": ACCESS_LEVELS[access],
        }, token)
        created = True
        print(f"created project {project_id}: {title!r} ({access})")
    else:
        project = api(args.url, f"/projects/{project_id}", None, token)
        print(f"importing into existing project {project_id}: "
              f"{project.get('title')!r} ({project.get('objcount') or 0} objects)")
        if args.access and project.get("access") != ACCESS_LEVELS[args.access]:
            project["access"] = ACCESS_LEVELS[args.access]
            api(args.url, f"/projects/{project_id}", project, token, method="PUT")
            print(f"  access set to {args.access}")

    total_rows = 0
    for archive in archives:
        source_path = to_load_area_path(archive, args.shared_root,
                                        args.load_area_prefix)
        print(f"  {archive.name} -> {source_path}")
        response = api(args.url, f"/file_import/{project_id}", {
            "source_path": source_path,
            "skip_loaded_files": False,
            "skip_existing_objects": args.skip_existing_objects,
            "update_mode": args.update_mode,
        }, token)
        if response.get("errors"):
            print(f"ERROR: import refused: {response['errors']}", file=sys.stderr)
            return 1
        job = wait_for_job(args.url, token, response["job_id"], answer_questions=True)
        if job.get("state") != "F":
            # "Asking" means the import wants a question answered, which needs a
            # human; say which rather than leaving a job parked invisibly.
            print(f"ERROR: import job {response['job_id']} ended in state "
                  f"{job.get('state')}: {job.get('progress_msg') or ''}",
                  file=sys.stderr)
            if job.get("question"):
                print(f"  it is asking: {json.dumps(job['question'])[:400]}",
                      file=sys.stderr)
            if job.get("errors"):
                print(f"  errors: {job['errors'][:5]}", file=sys.stderr)
            return 1
        rows = (job.get("result") or {}).get("rowcount") or 0
        total_rows += rows
        print(f"    imported {rows} objects")

    configure_project(args.url, token, project_id)
    project = api(args.url, f"/projects/{project_id}", None, token)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The Hub chains tasks by path, not by value, so the project id is written
    # where a following task can read it.
    (out_dir / "import.json").write_text(json.dumps({
        "project_id": project_id,
        "title": project.get("title"),
        "created": created,
        "imported_rows": total_rows,
        "objcount": project.get("objcount"),
        "archives": [a.name for a in archives],
    }, indent=2) + "\n")

    print(f"project {project_id} now holds {project.get('objcount')} objects")
    print(f"wrote {out_dir / 'import.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
