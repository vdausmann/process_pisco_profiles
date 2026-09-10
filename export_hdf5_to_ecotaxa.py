#!/usr/bin/env python3
"""Export EcoTaxa import archives from the C++ segmenter's HDF5 results.

The old Python pipeline writes per-frame CSVs plus crop PNGs on disk, which
``process_pisco_profiles.py`` turns into an EcoTaxa archive. The C++ segmenter
route produces neither: object crops, their masks and their bounding boxes all
live inside ``result.h5``. This script is the missing half — it reads that file
and emits exactly the same archive shape the EcoTaxa importer already accepts,
so ``upload_to_ecotaxa.py`` works on the result unchanged.

Input is ``result.h5``, written by ``Mode::OBJECT_EXTRACTION``:

    /                      @imageHeight @imageWidth @sourcePath
    /<frame name>/         one group per source frame
        boundingBoxX/Y/W/H (N,) int32   position and size in the full frame
        offsets            (N,) int32   start of each crop in the flat arrays
        cropPixelData      (M,) uint8   all crops concatenated, row-major
        maskPixelData      (M,) uint8   segmentation masks, same layout

Crop *i* is ``cropPixelData[offsets[i] : offsets[i] + W*H].reshape(H, W)``.

Two things make this better than exporting from the old pipeline's PNGs:

* If ``pisco_deconvolve`` ran before extraction, the crops in ``result.h5`` are
  already deconvolved — deconvolution rewrites the correction checkpoints in
  place and extraction reads those.
* ``maskPixelData`` is the segmenter's own mask, so morphometrics are computed
  from what the segmenter actually detected. The old pipeline had to re-derive a
  mask by re-thresholding the saved crop and keeping the largest contour.

Crops carry a reversible scale bar (``scale_bar.py``): the strip is appended
below the image and the original size recorded in the PNG, so the crop can be
restored byte-for-byte before it is used as classifier input.

Crops are inverted on the way out: the segmenter stores objects bright on a
dark ground, while EcoTaxa and the old pipeline's crops are dark on light. The
inversion happens before morphometrics too, so intensity features match the
other route.

Deliberately free of pandas and of ``utils.py`` (which imports torch and
transformers at module scope): an export job should not drag a GPU stack in.
The few shared constants below are duplicated for that reason.

With ``--link-base`` every object also gets an ``object_link`` pointing back at
the Hub, which resolves it to the HDF profile viewer opened on that object in
its frame. That is what connects a classified crop in EcoTaxa to the raw data
it came from.

Usage:
    python export_hdf5_to_ecotaxa.py /path/to/results/result.h5 -o /path/to/out
    python export_hdf5_to_ecotaxa.py /path/to/results -o /out \\
        --link-base http://10.11.11.110:5000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
# The repo's reversible scale bar. It lives in its own module because
# utils.py imports torch and transformers at module scope, and the
# processing workers have neither.
from scale_bar import PIXEL_SIZE_UM, SCALE_BAR_STRIP_H, save_crop_with_scale_bar
from skimage import measure

# --- constants shared with utils.py, duplicated to keep this import-light ----

# Mapping from model output class names to current EcoTaxa taxonomy names.
ECOTAXA_TAXON_MAP = {
    "Copepoda": "Copepoda<Multicrustacea",
    "Appendicularia": "Appendicularia<Tunicata",
    "Cnidaria<Metazoa": "Cnidaria<Animalia",
    "Chaetognatha": "Chaetognatha<Animalia",
    "Ctenophora_Metazoa": "Ctenophora<Animalia",
}

# Standard atmosphere in dbar, subtracted to turn absolute pressure into depth.
ATM_DBAR = 10.1325


def area_to_esd(area_px: float) -> float:
    """Equivalent spherical diameter in µm from an area in pixels."""
    return 2.0 * np.sqrt(area_px * PIXEL_SIZE_UM**2 / np.pi)


# --- frame name parsing -----------------------------------------------------

# e.g. M202_028-02_PISCO2_0009.94dbar-38.58N-027.32W-22.18C_20240725-05400026
_FRAME_RE = re.compile(
    r"^(?P<cruise>[^_]+)_(?P<dship_id>[^_]+)_(?P<instrument>[^_]+)_"
    r"(?P<meta>[^_]+)_(?P<datetime>\d{8}-\d+)$"
)
_META_RE = re.compile(
    r"^(?P<pressure>[\d.]+)(?P<punit>d?bar)"
    r"(?:-(?P<lat>[\d.]+)(?P<latdir>[NS])-(?P<lon>[\d.]+)(?P<londir>[EW]))?"
    r"(?:-(?P<temp>[\d.]+)C)?$"
)


def parse_frame_name(name: str) -> dict:
    """Pull acquisition metadata out of a frame group name.

    The segmenter names groups after the source frame, which encodes pressure,
    position, temperature and timestamp. Anything that does not match is
    returned with empty fields rather than raising — a profile with an
    unexpected naming scheme should still export its images.
    """
    out = {
        "cruise": "", "dship_id": "", "instrument": "",
        "pressure": None, "lat": None, "lon": None, "temperature": None,
        "date": "", "time": "",
    }
    m = _FRAME_RE.match(name)
    if not m:
        return out

    out["cruise"] = m.group("cruise")
    out["dship_id"] = m.group("dship_id")
    out["instrument"] = m.group("instrument")

    mm = _META_RE.match(m.group("meta"))
    if mm:
        pressure = float(mm.group("pressure"))
        if mm.group("punit") == "bar":  # some cruises write bar, not dbar
            pressure *= 10.0
        out["pressure"] = pressure
        if mm.group("lat"):
            lat = float(mm.group("lat"))
            lon = float(mm.group("lon"))
            out["lat"] = -lat if mm.group("latdir") == "S" else lat
            out["lon"] = -lon if mm.group("londir") == "W" else lon
        if mm.group("temp"):
            out["temperature"] = float(mm.group("temp"))

    date_part, _, time_part = m.group("datetime").partition("-")
    out["date"] = date_part
    # The tail is HHMMSS plus a frame counter; EcoTaxa wants exactly HHMMSS.
    out["time"] = time_part[:6].zfill(6)
    return out


# --- predictions -----------------------------------------------------------

def load_predictions(predictions_path: Path, taxonomy_map_path: Path):
    """object_id -> (taxon_id, display_name, score), plus the model's name.

    The TSV carries ``object_annotation_category_id``, the numeric EcoTaxa
    taxon id, rather than a name: names are ambiguous (``Ctenophora`` is both a
    comb jelly and a diatom) and the id is what the importer validates.

    Labels marked ``skip`` in the map are the binary stage's gate, not
    classifications; those objects are exported unannotated.
    """
    from ecotaxa_taxonomy_map import load_map

    if predictions_path.is_dir():
        predictions_path = predictions_path / "predictions.csv"
    mapping = load_map(taxonomy_map_path)          # raises if any label is unmapped
    raw = json.loads(Path(taxonomy_map_path).read_text())
    skipped = {label for label, e in raw["labels"].items() if e.get("skip")}
    display = {label: e.get("display_name", "") for label, e in raw["labels"].items()}
    model_name = raw.get("source_model", "")

    out: dict[str, tuple] = {}
    n_gate = n_unmapped = 0
    with open(predictions_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("top1") or "").strip()
            if not label:
                continue
            if label in skipped:
                n_gate += 1
                continue
            taxon_id = mapping.get(label)
            if taxon_id is None:
                n_unmapped += 1
                continue
            score = row.get("prob1") or ""
            out[row["object_id"]] = (taxon_id, display.get(label, label), score)
    return out, model_name, n_gate, n_unmapped


# --- back-links into the Hub ----------------------------------------------

# Hub outputs live under .../workflows/<uuid>/<n>_<tool>/<output>/
_WORKFLOW_IN_PATH = re.compile(r"/workflows/([0-9a-fA-F-]{36})/")

# EcoTaxa's object_link column. Longer values are rejected by the import, so
# we drop the link rather than let the row fail.
OBJECT_LINK_MAX = 255


def workflow_id_from_path(path: Path) -> str | None:
    """The Hub workflow a results directory belongs to, if it is one."""
    match = _WORKFLOW_IN_PATH.search(str(path))
    return match.group(1) if match else None


# --- morphometrics ----------------------------------------------------------

# Feature names follow the old pipeline's EcoTaxa columns (see gen_crop_df's
# `headers` in utils.py), so exports from both routes have the same schema.
def region_features(mask: np.ndarray, crop: np.ndarray) -> dict:
    """Zooprocess-style features from the segmenter's mask.

    Mirrors regionprop2zooprocess() in utils.py. Where a mask has several
    disconnected blobs we keep the largest, matching the old pipeline's
    "largest contour" behaviour.
    """
    labels = measure.label(mask > 0)
    props = measure.regionprops(labels, intensity_image=crop)
    if not props:
        return {}
    prop = max(props, key=lambda p: p.area)

    perim = float(prop.perimeter) or float("nan")
    major = float(prop.axis_major_length) or float("nan")
    filled = float(prop.area_filled)
    area = float(prop.area)

    return {
        "object_bound_box_w": prop.bbox[3] - prop.bbox[1],
        "object_bound_box_h": prop.bbox[2] - prop.bbox[0],
        "object_circularity": (4 * np.pi * filled) / perim**2,
        "object_area_exc": area,
        "object_area_rprops": filled,
        "object_%area": 1 - (area / filled) if filled else 0.0,
        "object_major_axis_len": major,
        "object_minor_axis_len": float(prop.axis_minor_length),
        "object_centroid_y": float(prop.centroid[0]),
        "object_centroid_x": float(prop.centroid[1]),
        "object_convex_area": float(prop.area_convex),
        "object_min_intensity": float(prop.intensity_min),
        "object_max_intensity": float(prop.intensity_max),
        "object_mean_intensity": float(prop.intensity_mean),
        "object_int_density": filled * float(prop.intensity_mean),
        "object_perimeter": perim,
        "object_elongation": major / float(prop.axis_minor_length)
        if prop.axis_minor_length else float("nan"),
        "object_range": float(prop.intensity_max) - float(prop.intensity_min),
        "object_perim_area_excl": perim / area if area else float("nan"),
        "object_perim_major": perim / major,
        "object_circularity_area_excl": (4 * np.pi * area) / perim**2,
        "object_angle": float(prop.orientation) / np.pi * 180 + 90,
        "object_boundbox_area": float(prop.area_bbox),
        "object_eccentricity": float(prop.eccentricity),
        "object_equivalent_diameter": float(prop.equivalent_diameter_area),
        "object_euler_nr": float(prop.euler_number),
        "object_extent": float(prop.extent),
        "object_local_centroid_col": float(prop.centroid_local[1]),
        "object_local_centroid_row": float(prop.centroid_local[0]),
        "object_solidity": float(prop.solidity),
    }


# --- column order -----------------------------------------------------------

# Order mirrors a real EcoTaxa export from the old pipeline. img_file_name and
# object_id are the only columns EcoTaxa strictly requires.
BASE_COLUMNS = [
    "img_file_name", "img_rank", "object_id", "object_link",
    "object_date", "object_time",
    "object_lat", "object_lon",
    "object_pressure", "object_depth_min", "object_depth_max",
    "object_temperature",
    "object_x", "object_y", "object_width", "object_height",
    "object_area", "object_esd",
    "object_annotation_category", "object_annotation_category_id",
    "object_annotation_status", "object_prediction_score",
    "object_prediction_model",
]
FEATURE_COLUMNS = [
    "object_bound_box_w", "object_bound_box_h", "object_circularity",
    "object_area_exc", "object_area_rprops", "object_%area",
    "object_major_axis_len", "object_minor_axis_len",
    "object_centroid_y", "object_centroid_x", "object_convex_area",
    "object_min_intensity", "object_max_intensity", "object_mean_intensity",
    "object_int_density", "object_perimeter", "object_elongation",
    "object_range", "object_perim_area_excl", "object_perim_major",
    "object_circularity_area_excl", "object_angle", "object_boundbox_area",
    "object_eccentricity", "object_equivalent_diameter", "object_euler_nr",
    "object_extent", "object_local_centroid_col", "object_local_centroid_row",
    "object_solidity",
]
SAMPLE_COLUMNS = ["sample_id", "sample_cruise", "sample_dship_id", "sample_instrument"]
ALL_COLUMNS = BASE_COLUMNS + FEATURE_COLUMNS + SAMPLE_COLUMNS

# EcoTaxa's type row: [t] text, [f] float. Everything not listed here is text.
TEXT_COLUMNS = {
    "img_file_name", "object_id", "object_link", "object_date", "object_time",
    "object_annotation_category", "object_annotation_status",
    "object_prediction_model",
    "sample_id", "sample_cruise", "sample_dship_id", "sample_instrument",
}


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


# --- export -----------------------------------------------------------------

def export(
    h5_path: Path,
    out_dir: Path,
    profile_name: str | None = None,
    min_size: int = 0,
    absolute_pressure: bool = True,
    image_ext: str = "png",
    link_base: str | None = None,
    workflow_id: str | None = None,
    scale_bar: bool = True,
    pixel_size_um: float = PIXEL_SIZE_UM,
    predictions: dict | None = None,
    prediction_model: str = "",
) -> tuple[Path, Path, int]:
    """Write crops and an EcoTaxa TSV. Returns (crop_dir, tsv_path, n_objects)."""
    profile = profile_name or h5_path.parent.name
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_frames = 0
    n_skipped = 0
    n_links = 0
    n_links_too_long = 0
    n_predicted = 0

    if link_base and not workflow_id:
        workflow_id = workflow_id_from_path(h5_path)
    link_base = link_base.rstrip("/") if link_base else None

    with h5py.File(h5_path, "r") as f:
        for rank, frame in enumerate(sorted(f.keys()), start=1):
            group = f[frame]
            if not isinstance(group, h5py.Group) or "boundingBoxW" not in group:
                continue
            n_frames += 1
            meta = parse_frame_name(frame)

            bb_x = np.asarray(group["boundingBoxX"])
            bb_y = np.asarray(group["boundingBoxY"])
            bb_w = np.asarray(group["boundingBoxW"])
            bb_h = np.asarray(group["boundingBoxH"])
            offsets = np.asarray(group["offsets"])
            crops = np.asarray(group["cropPixelData"])
            masks = np.asarray(group["maskPixelData"]) if "maskPixelData" in group else None

            for i in range(len(offsets)):
                w, h = int(bb_w[i]), int(bb_h[i])
                if w <= 0 or h <= 0:
                    n_skipped += 1
                    continue
                if min_size and w <= min_size and h <= min_size:
                    n_skipped += 1
                    continue

                off = int(offsets[i])
                n_px = w * h
                if off + n_px > crops.size:
                    n_skipped += 1
                    continue

                # The segmenter stores objects bright on a dark ground, the way
                # the sensor sees them. EcoTaxa — and the old Python pipeline's
                # saved crops — want them dark on light. Invert before both the
                # PNG and the features, because the old route measures its
                # intensity morphometrics on the dark-on-light image; skipping
                # this would flip min/max/mean_intensity, int_density and range
                # relative to exports from the other route.
                crop = (255 - crops[off:off + n_px].reshape(h, w)).astype(np.uint8)
                object_id = f"{frame}_{i + 1}"
                rel_img = f"{object_id}.{image_ext}"
                if scale_bar:
                    # Records the original size in a PNG text chunk, so
                    # strip_scale_bar() can undo the strip and any right
                    # padding exactly before the crop reaches a classifier.
                    save_crop_with_scale_bar(Image.fromarray(crop),
                                             crop_dir / rel_img,
                                             pixel_resolution=pixel_size_um)
                else:
                    Image.fromarray(crop).save(crop_dir / rel_img)

                row = {c: "" for c in ALL_COLUMNS}
                row.update({
                    "img_file_name": rel_img,
                    "img_rank": rank,
                    "object_id": object_id,
                    "object_date": meta["date"],
                    "object_time": meta["time"],
                    "object_lat": meta["lat"],
                    "object_lon": meta["lon"],
                    "object_pressure": meta["pressure"],
                    "object_temperature": meta["temperature"],
                    "object_x": int(bb_x[i]),
                    "object_y": int(bb_y[i]),
                    "object_width": w,
                    "object_height": h,
                    "sample_id": profile,
                    "sample_cruise": meta["cruise"],
                    "sample_dship_id": meta["dship_id"],
                    "sample_instrument": meta["instrument"],
                })

                if predictions:
                    predicted = predictions.get(object_id)
                    if predicted:
                        taxon_id, display_name, score = predicted
                        # The id is what EcoTaxa validates; the name rides
                        # along so the TSV is readable by a human.
                        row["object_annotation_category_id"] = taxon_id
                        row["object_annotation_category"] = display_name
                        row["object_annotation_status"] = "predicted"
                        row["object_prediction_score"] = score
                        row["object_prediction_model"] = prediction_model
                        n_predicted += 1

                if link_base and workflow_id:
                    # Points at the Hub, which resolves the workflow to its
                    # result.h5 and opens the viewer on this object. The
                    # object number matches the one in object_id, counting
                    # from 1.
                    link = f"{link_base}/v/{workflow_id}/{frame}/{i + 1}"
                    if len(link) <= OBJECT_LINK_MAX:
                        row["object_link"] = link
                        n_links += 1
                    else:
                        n_links_too_long += 1

                if meta["pressure"] is not None:
                    depth = meta["pressure"] - (ATM_DBAR if absolute_pressure else 0.0)
                    row["object_depth_min"] = round(depth, 3)
                    row["object_depth_max"] = round(depth, 3)

                if masks is not None:
                    mask = masks[off:off + n_px].reshape(h, w)
                    feats = region_features(mask, crop)
                    row.update(feats)
                    if "object_area_exc" in feats:
                        row["object_area"] = feats["object_area_exc"]
                        row["object_esd"] = round(
                            float(area_to_esd(feats["object_area_exc"])), 2)

                rows.append(row)

    tsv_path = out_dir / f"ecotaxa_{profile}.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(ALL_COLUMNS)
        writer.writerow(["[t]" if c in TEXT_COLUMNS else "[f]" for c in ALL_COLUMNS])
        for row in rows:
            writer.writerow([_fmt(row.get(c, "")) for c in ALL_COLUMNS])

    print(f"  frames: {n_frames}  objects: {len(rows)}  skipped: {n_skipped}")
    if predictions is not None:
        print(f"  predictions: {n_predicted} of {len(rows)} objects annotated"
              + (f" ({prediction_model})" if prediction_model else ""))
    if scale_bar:
        print(f"  scale bar: reversible, at {pixel_size_um} um/px "
              f"({SCALE_BAR_STRIP_H} px strip below each crop)")
    if link_base:
        if workflow_id:
            print(f"  object_link: {n_links} written"
                  + (f", {n_links_too_long} over {OBJECT_LINK_MAX} chars and dropped"
                     if n_links_too_long else ""))
        else:
            print("  object_link: skipped - no workflow id in the path, "
                  "pass --workflow-id to link anyway")
    return crop_dir, tsv_path, len(rows)


def make_zips(crop_dir: Path, tsv_path: Path, out_dir: Path, profile: str,
              max_zip_size_mb: int = 500) -> list[Path]:
    """Package crops + TSV into one or more EcoTaxa import archives.

    EcoTaxa wants the TSV and the images it references in the same archive, so
    a split archive gets its own TSV holding only that part's rows.
    """
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        type_row = next(reader)
        data_rows = list(reader)

    name_idx = header.index("img_file_name")
    max_bytes = max_zip_size_mb * 1024 * 1024

    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_size = 0
    for row in data_rows:
        img = crop_dir / row[name_idx]
        size = img.stat().st_size if img.exists() else 0
        if current and current_size + size > max_bytes:
            groups.append(current)
            current, current_size = [], 0
        current.append(row)
        current_size += size
    if current:
        groups.append(current)

    written = []
    for n, group in enumerate(groups, start=1):
        suffix = "" if len(groups) == 1 else f"_part{n}"
        zip_path = out_dir / f"{profile}{suffix}_ecotaxa.zip"
        tsv_name = f"ecotaxa_{profile}.tsv"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            buf = ["\t".join(header), "\t".join(type_row)]
            buf += ["\t".join(r) for r in group]
            zf.writestr(tsv_name, "\n".join(buf) + "\n")
            for row in group:
                img = crop_dir / row[name_idx]
                if img.exists():
                    zf.write(img, row[name_idx])
        written.append(zip_path)
        print(f"  {zip_path.name}  {len(group)} objects  "
              f"{zip_path.stat().st_size / 1e6:.1f} MB")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export an EcoTaxa import archive from the C++ segmenter's "
                    "result.h5 (see README_ECOTAXA.md for the upload step).")
    ap.add_argument("input", help="result.h5, or a results directory containing one")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    ap.add_argument("--profile-name", default=None,
                    help="sample_id and archive name (default: results dir name)")
    ap.add_argument("--max-zip-size", type=int, default=500,
                    help="split archives above this size in MB (default: 500)")
    ap.add_argument("--min-size", type=int, default=0,
                    help="drop objects whose width and height are both <= this")
    ap.add_argument("--gauge-pressure", action="store_true",
                    help="treat filename pressure as gauge; by default it is "
                         "absolute and 10.1325 dbar is subtracted for depth")
    ap.add_argument("--scale-bar", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Draw a scale bar under each crop (default: on). "
                         "Reversible: strip_scale_bar() in scale_bar.py "
                         "restores the original crop exactly.")
    ap.add_argument("--pixel-size", type=float, default=PIXEL_SIZE_UM,
                    help=f"Micrometres per pixel for the scale bar "
                         f"(default: {PIXEL_SIZE_UM}, the value area_to_esd "
                         f"uses, so the bar and object_esd agree).")
    ap.add_argument("--predictions", default=None,
                    help="predictions.csv from vit_classify_hdf5.py, or that "
                         "stage's output directory. Annotations ride the import "
                         "TSV, so no second trip to EcoTaxa is needed.")
    ap.add_argument("--taxonomy-map", default=None,
                    help="ecotaxa_taxonomy_map.json. Required with --predictions: "
                         "the TSV carries numeric taxon ids, not label names.")
    ap.add_argument("--link-base", default=None,
                    help="Hub base URL, e.g. http://10.11.11.110:5000. Adds an "
                         "object_link per object that opens the HDF profile "
                         "viewer on that object.")
    ap.add_argument("--workflow-id", default=None,
                    help="Workflow the results came from. Read from the path "
                         "when it is a Hub output; only needed otherwise.")
    ap.add_argument("--no-zip", action="store_true",
                    help="write crops and TSV but skip archiving")
    args = ap.parse_args()

    src = Path(args.input)
    if src.is_dir():
        candidate = src / "result.h5"
        if not candidate.exists():
            found = sorted(src.glob("result*.h5"))
            if not found:
                print(f"ERROR: no result.h5 under {src}", file=sys.stderr)
                return 1
            candidate = found[0]
        src = candidate
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = args.profile_name or src.parent.name

    predictions = None
    prediction_model = ""
    if args.predictions:
        if not args.taxonomy_map:
            print("ERROR: --predictions needs --taxonomy-map", file=sys.stderr)
            return 1
        predictions, prediction_model, n_gate, n_unmapped = load_predictions(
            Path(args.predictions), Path(args.taxonomy_map))
        print(f"predictions: {len(predictions)} usable"
              + (f", {n_gate} gate-only, {n_unmapped} unmapped"
                 if (n_gate or n_unmapped) else ""))

    print(f"Exporting {src}")
    crop_dir, tsv_path, n = export(
        src, out_dir, profile_name=profile, min_size=args.min_size,
        absolute_pressure=not args.gauge_pressure,
        link_base=args.link_base, workflow_id=args.workflow_id,
        scale_bar=args.scale_bar, pixel_size_um=args.pixel_size,
        predictions=predictions, prediction_model=prediction_model)

    if n == 0:
        print("ERROR: no objects exported", file=sys.stderr)
        return 1

    if not args.no_zip:
        make_zips(crop_dir, tsv_path, out_dir, profile, args.max_zip_size)

    print(f"Done: {n} objects -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
