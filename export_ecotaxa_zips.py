#!/usr/bin/env python3
"""
Export EcoTaxa ZIP files from existing processed profiles.

Usage:
  python export_ecotaxa_zips.py /path/to/results_root
  python export_ecotaxa_zips.py /path/to/results_root --max-zip-size 1000
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Try to import utils from local or as package
try:
    from utils import create_ecotaxa_zips
except ImportError:
    try:
        from process_pisco_profiles.utils import create_ecotaxa_zips
    except ImportError:
        print("ERROR: Could not import utils module. Make sure it's in the same directory.")
        sys.exit(1)


def find_profile_results(results_root):
    """Find all profile result folders containing EcoTaxa TSV files."""
    results_root = Path(results_root)
    profile_dirs = []
    
    # Check if this is a single profile folder
    if (results_root / "EcoTaxa").exists():
        profile_dirs.append(results_root)
    else:
        # Search for nested profile folders
        for item in results_root.rglob("*"):
            if item.is_dir() and item.name == "EcoTaxa":
                # Found EcoTaxa folder, parent should be the Results folder
                results_dir = item.parent
                if (
                    results_dir.name.endswith("_Results")
                    and (results_dir / "Data").exists()
                    and (results_dir / "Deconv_crops").exists()
                ):
                    profile_dirs.append(results_dir)
    
    return sorted(set(profile_dirs))


def export_profile_zips(results_folder, max_zip_size_mb=500):
    """Export ZIPs for a single profile."""
    results_folder = Path(results_folder)
    profile_name = results_folder.parent.name if results_folder.parent.name else results_folder.name
    
    ecotaxa_dir = results_folder / "EcoTaxa"
    if not ecotaxa_dir.exists():
        print(f"[WARN] No EcoTaxa folder in {results_folder}")
        return False
    
    # Find EcoTaxa TSV file
    tsv_files = list(ecotaxa_dir.glob("*_ecotaxa.tsv"))
    if not tsv_files:
        print(f"[WARN] No EcoTaxa TSV found in {ecotaxa_dir}")
        return False
    
    tsv_path = tsv_files[0]
    print(f"Processing {profile_name}...")
    print(f"  Reading TSV: {tsv_path}")
    
    try:
        # EcoTaxa TSVs saved by process_pisco_profiles.py use two header rows:
        # level 0 = column name, level 1 = type marker ([t], [f], ...).
        df = pd.read_csv(tsv_path, sep="\t", header=[0, 1], low_memory=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=["header", "type"])
        print(f"  Loaded {len(df)} rows")
    except Exception as e:
        print(f"[ERROR] Could not read TSV: {e}")
        return False

    # Fix pressure column: strip any unit suffix and ensure numeric type marker
    for col in df.columns:
        col_name = col[0] if isinstance(col, tuple) else col
        if col_name == 'object_pressure':
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'\s*d?bar\s*', '', regex=True),
                errors='coerce'
            )
            if isinstance(col, tuple):
                df.columns = pd.MultiIndex.from_tuples(
                    [(c[0], '[f]') if c[0] == 'object_pressure' else c for c in df.columns],
                    names=df.columns.names
                )
            break

    # Remap model class names to current EcoTaxa taxonomy names
    ecotaxa_taxon_map = {
        'copepoda': 'Copepoda<Multicrustacea',
        'appendicularia': 'Appendicularia<Tunicata',
        'cnidaria<metazoa': 'Cnidaria<Animalia',
        'chaetognatha': 'Chaetognatha<Animalia',
        'ctenophora_metazoa': 'Ctenophora<Animalia',
    }
    annotation_col_names = [
        'object_annotation_category', 'object_annotation_category_2',
        'object_annotation_category_3', 'object_annotation_category_4',
        'object_annotation_category_5',
    ]
    for col in df.columns:
        col_name = col[0] if isinstance(col, tuple) else col
        if col_name in annotation_col_names:
            df[col] = df[col].replace(ecotaxa_taxon_map)

    # Create ZIPs
    try:
        create_ecotaxa_zips(
            output_folder=str(results_folder),
            df=df,
            profile_name=profile_name,
            max_zip_size_mb=max_zip_size_mb,
            compression_ratio=1.0,
            copy_images=True
        )
        print(f"  OK: ZIPs created")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create ZIPs: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export EcoTaxa ZIP files from existing processed profiles."
    )
    parser.add_argument(
        "results_roots",
        nargs="+",
        help="One or more paths to results roots or individual profile Results folders"
    )
    parser.add_argument(
        "--max-zip-size",
        type=int,
        default=500,
        help="Maximum ZIP file size in MB (default: 500)"
    )

    args = parser.parse_args()

    profile_dirs = []
    for root in args.results_roots:
        if not os.path.exists(root):
            print(f"ERROR: {root} not found")
            sys.exit(1)
        found = find_profile_results(root)
        if not found:
            print(f"[WARN] No profile Results folders found in {root}")
        profile_dirs.extend(found)

    # Deduplicate while preserving order
    seen = set()
    profile_dirs = [p for p in profile_dirs if not (p in seen or seen.add(p))]

    if not profile_dirs:
        print("ERROR: No profile Results folders found.")
        sys.exit(1)

    print(f"Found {len(profile_dirs)} profile(s)")

    success_count = 0
    for profile_dir in profile_dirs:
        if export_profile_zips(profile_dir, max_zip_size_mb=args.max_zip_size):
            success_count += 1

    print(f"\nComplete: {success_count}/{len(profile_dirs)} profiles exported")
    sys.exit(0 if success_count == len(profile_dirs) else 1)


if __name__ == "__main__":
    main()
