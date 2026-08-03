#!/usr/bin/env python3
"""
Regenerate deconvolved crops for already-processed profiles and (optionally)
build the EcoTaxa ZIPs — for when the crop dirs were deleted but the Data CSVs
and EcoTaxa TSVs were kept (e.g. after --cleanup-intermediate).

For each profile it reuses the real segmenter (reader -> bg-correction ->
deconvolution -> detection) in a deconv-only mode so the crops are byte-faithful
and their filenames match `object_full_path` in the preserved TSV:
  - force_all=True         : re-run every frame even though Data CSVs exist
  - save_raw_crops/masks/data = False : write ONLY Deconv_crops, keep Data/EcoTaxa
  - mask_radius_override    : taken from the profile's settings.csv (exact match)

Then, per profile (to keep inodes bounded): build the EcoTaxa ZIP from the
regenerated Deconv_crops + preserved TSV, and delete the loose Deconv_crops.

Run one instance per GPU for parallelism, e.g.:
  HIP_VISIBLE_DEVICES=0 python regenerate_deconv_crops.py --output <root> --profiles-file listA.txt
  HIP_VISIBLE_DEVICES=1 python regenerate_deconv_crops.py --output <root> --profiles-file listB.txt
"""

import argparse
import csv
import importlib
import os
import shutil
import sys
import traceback
from datetime import datetime

# --- locate the segmenter (same fallback names as process_pisco_profiles.py) ---
run_segmenter = None
for _mod in ("segmenter", "pisco_segmenter"):
    try:
        _m = importlib.import_module(_mod)
        run_segmenter = getattr(_m, "run_segmenter", None)
        if run_segmenter is not None:
            break
    except ImportError:
        continue
if run_segmenter is None:
    sys.exit("ERROR: could not import run_segmenter from segmenter/pisco_segmenter")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_ecotaxa_zips import export_profile_zips  # noqa: E402


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def read_settings(results_folder):
    """Return dict from a profile's settings.csv (Field Name,Value)."""
    path = os.path.join(results_folder, "settings.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "Field Name":
                out[row[0]] = row[1]
    return out


def tsv_crop_basenames(ecotaxa_dir):
    """Set of crop image basenames referenced by object_full_path in the TSV."""
    import pandas as pd
    tsvs = [os.path.join(ecotaxa_dir, n) for n in os.listdir(ecotaxa_dir)] if os.path.isdir(ecotaxa_dir) else []
    tsvs = [t for t in tsvs if t.endswith("_ecotaxa.tsv")]
    if not tsvs:
        return set()
    try:
        df = pd.read_csv(tsvs[0], sep="\t", header=[0, 1], low_memory=False)
        col = [c for c in df.columns if (c[0] if isinstance(c, tuple) else c) == "object_full_path"]
        if not col:
            return set()
        return set(os.path.basename(str(p)) for p in df[col[0]].dropna())
    except Exception as e:
        log(f"  (coverage) could not parse TSV: {e}")
        return set()


def load_profiles(args):
    if args.profiles_file:
        with open(args.profiles_file) as f:
            return [ln.split("#", 1)[0].strip() for ln in f if ln.split("#", 1)[0].strip()]
    if args.profiles:
        return args.profiles
    # discover: any profile dir under output with a _Results/EcoTaxa/*.tsv
    out = []
    for d in sorted(os.listdir(args.output)):
        rf = os.path.join(args.output, d, f"{d}_Results")
        if os.path.isdir(os.path.join(rf, "EcoTaxa")):
            out.append(d)
    return out


def process_profile(profile, args):
    results_folder = os.path.join(args.output, profile, f"{profile}_Results")
    if not os.path.isdir(results_folder):
        log(f"SKIP {profile}: no _Results dir")
        return "skip"

    settings = read_settings(results_folder)
    src = settings.get("data source")
    if not src or not os.path.isdir(src):
        log(f"SKIP {profile}: source images not found ({src})")
        return "skip"

    mask_radius = settings.get("mask_radius")
    mask_radius = int(float(mask_radius)) if mask_radius else None

    deconv_dir = os.path.join(results_folder, "Deconv_crops")
    ecotaxa_dir = os.path.join(results_folder, "EcoTaxa")

    # resume: if a deconv_crops zip already exists in EcoTaxa, assume done
    if os.path.isdir(ecotaxa_dir) and any(
        n.endswith(".zip") and "deconv" in n.lower() for n in os.listdir(ecotaxa_dir)
    ):
        log(f"SKIP {profile}: EcoTaxa deconv zip already present")
        return "skip"

    log(f"RECROP {profile}  (src={src}, mask_radius={mask_radius})")
    run_segmenter(
        src, results_folder, True,
        force_all=True, save_raw_crops=False, save_masks=False, save_data=False,
        mask_radius_override=mask_radius,
    )
    n_crops = len(os.listdir(deconv_dir)) if os.path.isdir(deconv_dir) else 0
    log(f"  regenerated {n_crops} deconv crops")
    if n_crops == 0:
        log(f"  WARNING {profile}: no crops regenerated — leaving as-is")
        return "fail"

    # Faithfulness check: every crop the TSV references must now exist on disk.
    present = set(os.listdir(deconv_dir))
    expected = tsv_crop_basenames(ecotaxa_dir)
    if expected:
        missing = expected - present
        cov = 100.0 * (len(expected) - len(missing)) / len(expected)
        log(f"  TSV coverage: {cov:.3f}%  ({len(expected)-len(missing)}/{len(expected)}; {len(missing)} missing)")
        if missing:
            log(f"  WARNING {profile}: {len(missing)} TSV-referenced crops missing "
                f"(e.g. {sorted(missing)[:3]}) — NOT exporting/cleaning up.")
            return "fail"
    else:
        log(f"  WARNING {profile}: could not read TSV crop list — skipping coverage check")

    if args.export:
        log(f"  building EcoTaxa zip for {profile}")
        ok = export_profile_zips(results_folder, max_zip_size_mb=args.max_zip_size)
        if not ok:
            log(f"  WARNING {profile}: zip export failed — keeping Deconv_crops")
            return "fail"
        if args.cleanup:
            shutil.rmtree(deconv_dir, ignore_errors=True)
            log(f"  cleaned loose Deconv_crops for {profile}")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description="Regenerate deconv crops + EcoTaxa zips for processed profiles.")
    ap.add_argument("--output", required=True, help="Results root (parent of the profile dirs)")
    ap.add_argument("--profiles-file", help="Newline-delimited profile dir names")
    ap.add_argument("--profiles", nargs="+", help="Explicit profile dir names")
    ap.add_argument("--no-export", dest="export", action="store_false",
                    help="Only regenerate Deconv_crops; do not build zips")
    ap.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                    help="Keep the regenerated Deconv_crops after zipping")
    ap.add_argument("--max-zip-size", type=int, default=500, help="Max zip size MB (default 500)")
    args = ap.parse_args()

    profiles = load_profiles(args)
    if not profiles:
        log("No profiles to process.")
        return 1
    log(f"Profiles: {len(profiles)}   export={args.export}  cleanup={args.cleanup}")

    results = {"ok": [], "skip": [], "fail": []}
    for p in profiles:
        try:
            results[process_profile(p, args)].append(p)
        except Exception as e:
            log(f"ERROR {p}: {e}")
            traceback.print_exc()
            results["fail"].append(p)

    log("=" * 50)
    log(f"DONE  ok={len(results['ok'])}  skipped={len(results['skip'])}  failed={len(results['fail'])}")
    for p in results["fail"]:
        log(f"  FAILED: {p}")
    return 1 if results["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
