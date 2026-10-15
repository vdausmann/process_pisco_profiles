#!/usr/bin/env python3
"""Make every EcoTaxa user an annotator of a cruise's projects.

EcoTaxa's GUI only goes by explicit grants: a project at access level OPEN can
be annotated through the API, but the project list and the classification page
still show it as view-only. So the grants are written per user, and because an
account made tomorrow has none, this is meant to run on a schedule (cron on
the EcoTaxa machine) and is idempotent -- a project already right is left
alone.

Users already managers or annotators keep their role; viewers are raised to
annotators. Accounts that cannot log in are skipped: inactive ones, and the
``@mirror.local`` stand-ins ecotaxa_mirror.py creates to keep who validated
what.

Usage:
    ecotaxa_grant_annotators.py --title-prefix MSM147_
    ecotaxa_grant_annotators.py --project-id 38 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

from ecotaxa_import import DEFAULT_CREDENTIALS, api, load_credentials

SKIP_DOMAINS = ("@mirror.local",)

# What a project's member lists hold (the API's MinUserModel).
MIN_USER_FIELDS = ("id", "email", "name", "organisation")


def real_users(base, token) -> list[dict]:
    users = api(base, "/users", None, token)
    return [u for u in users
            if u.get("status") == 1
            and not u.get("email", "").endswith(SKIP_DOMAINS)]


def grant(base, token, project: dict, users: list[dict], dry_run: bool) -> int:
    """Add the missing annotators to one project; return how many were added."""
    managers = {u["id"] for u in project.get("managers") or []}
    annotators = {u["id"] for u in project.get("annotators") or []}
    missing = [u for u in users if u["id"] not in managers | annotators]
    if not missing:
        return 0
    names = ", ".join(u.get("name") or u["email"] for u in missing)
    print(f"project {project['projid']} {project['title']!r}: + {names}")
    if dry_run:
        return len(missing)
    added = {u["id"] for u in missing}
    project["viewers"] = [u for u in project.get("viewers") or []
                          if u["id"] not in added]
    project["annotators"] = list(project.get("annotators") or []) + [
        {k: u.get(k) for k in MIN_USER_FIELDS} for u in missing]
    api(base, f"/projects/{project['projid']}", project, token, method="PUT")
    return len(missing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--title-prefix", action="append",
                       help="every project whose title starts with this "
                            "(repeatable), e.g. MSM147_")
    which.add_argument("--project-id", type=int, action="append")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--credentials", default=os.environ.get(
        "ECOTAXA_CREDENTIALS", DEFAULT_CREDENTIALS))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    args = ap.parse_args()

    user, password = load_credentials(args.credentials, args.user, args.password)
    if not user or not password:
        print(f"ERROR: no credentials in {args.credentials}", file=sys.stderr)
        return 1
    token = api(args.url, "/login", {"username": user, "password": password})

    if args.project_id:
        ids = args.project_id
    else:
        # not_granted=false lists what the login has rights on, which for the
        # application administrator is everything.
        found = api(args.url, "/projects/search?not_granted=false", None, token)
        ids = [p["projid"] for p in found
               if p["title"].startswith(tuple(args.title_prefix))]

    users = real_users(args.url, token)
    added = 0
    for pid in ids:
        project = api(args.url, f"/projects/{pid}", None, token)
        added += grant(args.url, token, project, users, args.dry_run)
    if added:
        print(f"{added} grant(s) over {len(ids)} project(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
