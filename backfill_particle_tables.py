#!/usr/bin/env python3
"""Write the particle table for Hub runs exported before it existed.

Finds every ``pisco_ecotaxa_export`` task under a Hub's workflows directory
and runs the exporter in ``--table-only`` mode on that task's own settings, so
the table is measured with exactly the options the export used: same
predictions, same pixel size, and the same subsample, which is what makes its
``object_exported`` flags agree with what went to EcoTaxa. The flags are then
checked against the exported TSV.

The settings go through ``run.sh``, as the Hub runs them, on a copy with
``tableOnly = 1`` added; the task's own settings file is not touched, and
nothing but ``particles_<profile>.h5`` is written to its export directory.

Usage:
    backfill_particle_tables.py /media/veit/T710_data/hub-data/workflows --dry-run
    backfill_particle_tables.py /media/veit/T710_data/hub-data/workflows
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ini_value(text: str, key: str) -> str:
    """The same reading as run.sh's cfg(): first match, inline comments dropped."""
    m = re.search(rf"^\s*{key}\s*=(.*)$", text, re.IGNORECASE | re.MULTILINE)
    return re.sub(r"\s*#.*", "", m.group(1)).strip() if m else ""


def tsv_rows(path: Path) -> int:
    with open(path, "rb") as f:
        lines = sum(1 for _ in f)
    return max(lines - 2, 0)       # header and the [t]/[f] type row


def exported_count(table: Path) -> int:
    import h5py
    with h5py.File(table, "r") as f:
        return int(f["objects/object_exported"][()].sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workflows", help="the Hub's <DATA_ROOT>/workflows directory")
    ap.add_argument("--dry-run", action="store_true", help="list what would be done")
    ap.add_argument("--force", action="store_true", help="rewrite tables that exist")
    ap.add_argument("--python", default=None,
                    help="interpreter for the exporter (sets PISCO_PYTHON for run.sh)")
    args = ap.parse_args()

    tasks = sorted(Path(args.workflows).glob("*/*_pisco_ecotaxa_export/settings.ini"))
    print(f"{len(tasks)} export tasks under {args.workflows}")
    env = dict(os.environ)
    if args.python:
        env["PISCO_PYTHON"] = args.python

    done = skipped = failed = 0
    for settings in tasks:
        text = settings.read_text()
        target = Path(ini_value(text, "targetPath"))
        source = Path(ini_value(text, "sourcePath"))
        label = f"{settings.parent.parent.name[:8]} {settings.parent.name}"
        tsvs = sorted(target.glob("ecotaxa_*.tsv"))
        tables = sorted(target.glob("particles_*.h5"))
        if not tsvs:
            print(f"  skip  {label}: no exported TSV in {target}")
            skipped += 1
            continue
        if tables and not args.force:
            print(f"  have  {label}: {tables[0].name}")
            skipped += 1
            continue
        result = source / "result.h5" if source.is_dir() else source
        if not result.exists() and not sorted(source.glob("result*.h5")):
            print(f"  skip  {label}: no result.h5 in {source}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  would {label}: {tsvs[0].name}, {tsv_rows(tsvs[0])} exported rows")
            continue

        body = "\n".join(line for line in text.splitlines()
                         if not re.match(r"\s*(tableOnly|particleTable)\s*=", line, re.I))
        with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as tmp:
            tmp.write(body + "\ntableOnly = 1\n")
        try:
            run = subprocess.run(["bash", str(HERE / "run.sh"), tmp.name], env=env,
                                 capture_output=True, text=True)
        finally:
            os.unlink(tmp.name)
        if run.returncode != 0:
            print(f"  FAIL  {label}: exit {run.returncode}\n{run.stderr[-2000:]}")
            failed += 1
            continue
        tables = sorted(target.glob("particles_*.h5"))
        if not tables:
            print(f"  FAIL  {label}: the exporter wrote no table\n{run.stdout[-2000:]}")
            failed += 1
            continue
        # The check that the subsample was redrawn the same: the objects the
        # table marks as exported are as many as the TSV holds.
        n_tsv = sum(tsv_rows(t) for t in tsvs)
        n_exp = exported_count(tables[0])
        if n_exp != n_tsv:
            print(f"  WARN  {label}: table marks {n_exp} exported, the TSV has {n_tsv}")
            failed += 1
            continue
        summary = next((l.strip() for l in run.stdout.splitlines()
                        if "particle table:" in l), tables[0].name)
        print(f"  done  {label}: {summary}")
        done += 1

    print(f"{done} written, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
