#!/usr/bin/env python3
"""Build a dual-ViT training set from validated EcoTaxa objects.

Labels come from EcoTaxa (what a human validated); the images come from the
run that produced the objects. Two sources are supported, and both keep the
crops exactly as the classifier sees them:

* **``result.h5``** -- the segmenter's own output, read through
  ``vit_classify_hdf5.iter_crops``, i.e. the identical code path inference
  uses. Neighbour isolation is applied here, at build time, and must match the
  ``isolateCrops`` setting the model will be run with.
* **``deconv_crops*.zip``** -- the old python pipeline's exported crops,
  matched on the filename EcoTaxa stores as ``img.orig_file_name``. This is
  how already-validated pre-Hub projects can still be trained from.

The EcoTaxa *export* zips (``*_ecotaxa.zip``) are deliberately **not** a
source: their PNGs carry a scale bar, which no model was trained with.

Two sets are written, both as imagefolders (``<class>/<crop>.png``):

  binary/      living vs not-living, for the first-stage classifier
  multiclass/  the living taxa, for the second stage

Labels are read from ``labels.csv`` (see ``ecotaxa_fetch_labels.py``), which
is how this runs on a GPU worker with no EcoTaxa credentials. Passing
``--projects`` instead fetches them inline, for standalone use on a machine
that does have credentials.

Example (the ATAIIR2604_NorthSea run):

    build_trainset.py \\
        --labels /mnt/pisco/.../labels/labels.csv \\
        --crops-root /media/veit/T710_data/pisco_processed/ATAIR-BSH_v5alucyd_iso \\
        --out /home/veit/Documents/ATAIIR2604_NorthSea_trainset \\
        --isolate-crops \\
        --nonliving Unknowns not-living bubble t001 \\
        --exclude-multiclass "multiple species" \\
        --min-per-class 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The one implementation of "what a crop looks like". Training crops that do
# not come from this function are not the crops inference will produce.
from vit_classify_hdf5 import iter_crops  # noqa: E402


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Read labels.csv into {key: row}, keyed by object_id and by file name.

    Both keys are kept because the two crop sources are matched differently:
    HDF5 by ``object_id``, zips by the file name EcoTaxa recorded.
    """
    if labels_path.is_dir():
        labels_path = labels_path / "labels.csv"
    if not labels_path.exists():
        raise SystemExit(f"ERROR: {labels_path} not found")
    rows = {}
    with open(labels_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip()
            if label:
                rows[row["object_id"]] = {"object_id": row["object_id"],
                                          "file_name": row.get("file_name") or "",
                                          "label": label}
    return rows


def fetch_labels_inline(projects, url, credentials, user, password) -> dict[str, dict]:
    from ecotaxa_fetch_labels import fetch
    from ecotaxa_import import api, load_credentials
    user, password = load_credentials(credentials, user, password)
    if not user or not password:
        raise SystemExit("ERROR: no EcoTaxa credentials for --projects")
    token = api(url, "/login", {"username": user, "password": password})
    rows = {}
    for project_id in projects:
        n = 0
        for object_id, file_name, label in fetch(url, token, project_id, "V"):
            if label:
                rows[object_id] = {"object_id": object_id, "file_name": file_name,
                                   "label": label}
                n += 1
        print(f"  project {project_id}: {n} validated")
    return rows


def index_zips(roots: list[Path], wanted: set[str]) -> dict[str, tuple[Path, str]]:
    """Map crop file name -> (zip path, member) for the crops we want."""
    index: dict[str, tuple[Path, str]] = {}
    for root in roots:
        for zip_path in sorted(root.rglob("deconv_crops*.zip")):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    for member in zf.namelist():
                        name = os.path.basename(member)
                        if name in wanted and name not in index:
                            index[name] = (zip_path, member)
            except zipfile.BadZipFile:
                print(f"  WARNING: {zip_path} is not a readable zip, skipped")
    return index


def find_h5(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".h5":
            found.append(root)
        else:
            found.extend(sorted(root.rglob("result*.h5")))
    return found


def write_png(data_or_array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data_or_array, bytes):
        path.write_bytes(data_or_array)
    else:
        from PIL import Image
        Image.fromarray(data_or_array).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.csv from ecotaxa_fetch_labels.py, or its directory")
    ap.add_argument("--projects", type=int, nargs="+",
                    help="fetch labels inline instead, from these EcoTaxa projects "
                         "(priority order, last wins); needs credentials on this machine")
    ap.add_argument("--crops-root", nargs="+", required=True,
                    help="one or more roots holding result.h5 files and/or deconv_crops zips")
    ap.add_argument("--out", required=True, help="destination trainset directory")
    ap.add_argument("--isolate-crops", action="store_true",
                    help="mask neighbours out to white when reading from HDF5. Must "
                         "match the isolateCrops the model will be run with.")
    ap.add_argument("--nonliving", nargs="*", default=["Unknowns", "not-living"],
                    help="classes forming the binary not-living side")
    ap.add_argument("--exclude-multiclass", nargs="*", default=[],
                    help="living classes to leave out of the multiclass set")
    ap.add_argument("--min-per-class", type=int, default=20,
                    help="drop a multiclass class with fewer crops than this: it "
                         "cannot support a train/val/test split")
    ap.add_argument("--url", default=os.environ.get("ECOTAXA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--credentials", default=os.environ.get(
        "ECOTAXA_CREDENTIALS", "/opt/pisco/etc/ecotaxa.json"))
    ap.add_argument("--user", default=os.environ.get("ECOTAXA_USER"))
    ap.add_argument("--password", default=os.environ.get("ECOTAXA_PASSWORD"))
    a = ap.parse_args()

    if not a.labels and not a.projects:
        print("ERROR: one of --labels or --projects is required", file=sys.stderr)
        return 1

    print("Labels...")
    if a.labels:
        rows = load_labels(Path(a.labels))
    else:
        rows = fetch_labels_inline(a.projects, a.url, a.credentials, a.user, a.password)
    if not rows:
        print("ERROR: no labelled crops", file=sys.stderr)
        return 1
    print(f"  {len(rows)} labelled crops")

    nonliving = set(a.nonliving)
    excluded = set(a.exclude_multiclass)
    counts = collections.Counter(row["label"] for row in rows.values())
    mc_classes = {c for c, n in counts.items()
                  if c not in nonliving and c not in excluded and n >= a.min_per_class}
    dropped = {c: n for c, n in counts.items()
               if c not in nonliving and c not in mc_classes}
    print(f"\nbinary not-living : {sorted(nonliving)}")
    print(f"multiclass classes: {sorted(mc_classes)}")
    print(f"multiclass dropped: {dropped}")
    unseen = nonliving - set(counts)
    if unseen:
        print(f"  note: no validated crops for {sorted(unseen)}")

    roots = [Path(r) for r in a.crops_root]
    for root in roots:
        if not root.exists():
            print(f"ERROR: {root} does not exist", file=sys.stderr)
            return 1

    out = Path(a.out)
    written = collections.Counter()
    by_source = collections.Counter()
    resolved: set[str] = set()

    def emit(key: str, image) -> None:
        row = rows[key]
        cls = row["label"]
        name = row["file_name"] or f"{row['object_id']}.png"
        side = "not-living" if cls in nonliving else "living"
        write_png(image, out / "binary" / side / name)
        written["binary/" + side] += 1
        if cls in mc_classes:
            write_png(image, out / "multiclass" / cls.replace(" ", "_") / name)
            written["multiclass/" + cls] += 1
        resolved.add(key)

    print("\nReading crops from HDF5...")
    h5_files = find_h5(roots)
    print(f"  {len(h5_files)} result.h5 file(s)")
    keep = set(rows)
    for h5_path in h5_files:
        n = 0
        # iter_crops is the inference path: inverted, optionally isolated.
        for object_id, crop in iter_crops(h5_path, a.isolate_crops, keep):
            if object_id in resolved:
                continue
            emit(object_id, crop)
            n += 1
        by_source[str(h5_path)] = n
        print(f"  {h5_path}: {n}")

    missing = {rows[k]["file_name"]: k for k in rows if k not in resolved
               and rows[k]["file_name"]}
    if missing:
        print(f"\nLooking for the remaining {len(missing)} crops in deconv_crops zips...")
        index = index_zips(roots, set(missing))
        by_zip = collections.defaultdict(list)
        for name, (zip_path, member) in index.items():
            by_zip[zip_path].append((name, member))
        for zip_path, items in by_zip.items():
            with zipfile.ZipFile(zip_path) as zf:
                for name, member in items:
                    emit(missing[name], zf.read(member))
            by_source[str(zip_path)] = len(items)
            print(f"  {zip_path}: {len(items)}")

    unresolved = [k for k in rows if k not in resolved]
    if unresolved:
        print(f"\nWARNING: {len(unresolved)} labelled crops were not found in "
              f"{[str(r) for r in roots]} and are skipped "
              f"(e.g. {unresolved[:3]})")
    if not resolved:
        print("ERROR: no labelled crop could be matched to an image. Check that "
              "--crops-root points at the run these objects were exported from.",
              file=sys.stderr)
        return 1

    sources = {k: v for k, v in by_source.items() if v}
    if len({Path(s).suffix for s in sources}) > 1:
        print("\nNOTE: crops came from both HDF5 and zips. Those are different "
              "export routes; mixing them mixes appearance regimes.")

    provenance = {
        "n_labelled": len(rows), "n_written": len(resolved),
        "n_missing": len(unresolved),
        "isolate_crops": a.isolate_crops,
        "crops_root": [str(r) for r in roots],
        "sources": sources,
        "nonliving": sorted(nonliving),
        "excluded_multiclass": sorted(excluded),
        "min_per_class": a.min_per_class,
        "multiclass_classes": sorted(mc_classes),
        "dropped": dropped,
        "written": dict(written),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "trainset.json").write_text(json.dumps(provenance, indent=1))

    print("\nwritten:")
    for k, v in sorted(written.items(), key=lambda x: -x[1]):
        print(f"   {k:34} {v:>6}")
    print(f"\n{out}/trainset.json records how this set was built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
