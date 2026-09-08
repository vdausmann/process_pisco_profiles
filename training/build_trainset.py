#!/usr/bin/env python3
"""Build a dual-ViT training set from validated EcoTaxa objects.

Labels come from EcoTaxa (what a human validated); the images come from the
`deconv_crops` zips the run already produced, matched on the crop filename that
EcoTaxa stores as `img.orig_file_name`. Nothing is exported from EcoTaxa, so the
crops keep exactly the appearance the pipeline classified.

Two sets are written, both as imagefolders (`<class>/<crop>.png`):

  binary/      living vs not-living, for the first-stage classifier
  multiclass/  the living taxa, for the second stage

Deduplication matters: EcoTaxa subsets are *copies*, so the same physical crop
appears in a project and in every subset drawn from it. Projects are given in
priority order and the last one wins, which is normally what you want -- a
re-validation in a subset supersedes the original call.

Example (the ATAIIR2604_NorthSea run):

    build_trainset.py \\
        --projects 23022 23052 23196 \\
        --crops-root /media/veit/T710_data/pisco_processed/ATAIR-BSH_v5alucyd_iso \\
        --out /home/veit/Documents/ATAIIR2604_NorthSea_trainset \\
        --nonliving Unknowns not-living bubble t001 \\
        --exclude-multiclass "multiple species" \\
        --min-per-class 20
"""
import argparse
import collections
import glob
import json
import os
import zipfile


def short(taxon):
    """EcoTaxa display names carry lineage ("Copepoda<Multicrustacea")."""
    return (taxon or "?").split("<")[0]


def fetch_labels(projects, config_path):
    """Return {crop_filename: label}; later projects override earlier ones."""
    from ecotaxa_py_client import (Configuration, ApiClient, AuthentificationApi,
                                   LoginReq, ObjectsApi, ProjectFilters)
    cfg = json.load(open(config_path))["ecotaxa"]
    conf = Configuration(host=cfg["host"])
    with ApiClient(conf) as ac:
        conf.access_token = AuthentificationApi(ac).login(
            LoginReq(username=cfg["username"], password=cfg["password"]))
    labels = {}
    for pid in projects:
        with ApiClient(conf) as ac:
            res = ObjectsApi(ac).get_object_set(
                pid, ProjectFilters(statusfilter="V"),
                fields="obj.objid,img.orig_file_name,txo.display_name")
        n = 0
        for _objid, fn, taxon in res.details:
            if fn:
                labels[fn] = short(taxon)
                n += 1
        print(f"  project {pid}: {n} validated")
    return labels


def index_crops(crops_root):
    """Map crop filename -> (zip path, member name) across a run's exports."""
    index = {}
    for z in glob.glob(os.path.join(crops_root, "*", "*_Results", "EcoTaxa", "*",
                                    "deconv_crops*.zip")):
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if member.endswith(".png"):
                    index[os.path.basename(member)] = (z, member)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", type=int, nargs="+", required=True,
                    help="EcoTaxa project ids, in priority order (last wins on duplicates)")
    ap.add_argument("--crops-root", required=True,
                    help="Pipeline output root holding the deconv_crops zips")
    ap.add_argument("--out", required=True, help="Destination trainset directory")
    ap.add_argument("--config", default="process_pisco_profiles.config.example.json")
    ap.add_argument("--nonliving", nargs="*", default=["Unknowns", "not-living"],
                    help="Classes forming the binary not-living side")
    ap.add_argument("--exclude-multiclass", nargs="*", default=[],
                    help="Living classes to leave out of the multiclass set")
    ap.add_argument("--min-per-class", type=int, default=20,
                    help="Drop a multiclass class with fewer crops than this: it "
                         "cannot support a train/val/test split")
    a = ap.parse_args()

    nonliving = set(a.nonliving)
    excluded = set(a.exclude_multiclass)

    print("Fetching validated labels...")
    labels = fetch_labels(a.projects, a.config)
    print(f"  {len(labels)} distinct crops after deduplication")

    print("Indexing crops on disk...")
    index = index_crops(a.crops_root)
    missing = [f for f in labels if f not in index]
    if missing:
        print(f"  WARNING: {len(missing)} labelled crops not found "
              f"(e.g. {missing[:3]}) - they are skipped")
        for f in missing:
            del labels[f]
    print(f"  {len(index)} crops indexed, {len(labels)} usable")

    counts = collections.Counter(labels.values())
    mc_classes = {c for c, n in counts.items()
                  if c not in nonliving and c not in excluded and n >= a.min_per_class}
    dropped = {c: n for c, n in counts.items()
               if c not in nonliving and c not in mc_classes}
    print(f"\nbinary not-living : {sorted(nonliving)}")
    print(f"multiclass classes: {sorted(mc_classes)}")
    print(f"multiclass dropped: {dropped}")

    # group by source zip so each archive is opened once
    by_zip = collections.defaultdict(list)
    for fn in labels:
        z, member = index[fn]
        by_zip[z].append((fn, member))

    written = collections.Counter()
    for z, items in by_zip.items():
        with zipfile.ZipFile(z) as zf:
            for fn, member in items:
                cls = labels[fn]
                data = zf.read(member)
                side = "not-living" if cls in nonliving else "living"
                d = os.path.join(a.out, "binary", side)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, fn), "wb") as fh:
                    fh.write(data)
                written["binary/" + side] += 1
                if cls in mc_classes:
                    d2 = os.path.join(a.out, "multiclass", cls.replace(" ", "_"))
                    os.makedirs(d2, exist_ok=True)
                    with open(os.path.join(d2, fn), "wb") as fh:
                        fh.write(data)
                    written["multiclass/" + cls] += 1

    print("\nwritten:")
    for k, v in sorted(written.items(), key=lambda x: -x[1]):
        print(f"   {k:34} {v:>6}")


if __name__ == "__main__":
    main()
