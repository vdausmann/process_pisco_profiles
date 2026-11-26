#!/usr/bin/env python3
"""Cut validated EcoTaxa objects out of the RAW frames as transmittance patches.

For every row of a ``labels.csv`` (from ``ecotaxa_fetch_labels.py``) this
finds the object's raw source frame and bounding box, divides the frame by its
background, and stores the padded patch

    t = frame / (background * gain)          (1 = clear water, <1 = shadow)

plus an instance mask. Shadowgraph objects attenuate light, so ``t`` composes
multiplicatively: a synthetic scene is ``background * prod(t_i) + noise``, and
overlapping objects come out the way real overlapping shadows do. The halo of
a defocused object is kept (the patch is padded), so pasted objects have no
cut edge.

Where the boxes come from: the old python route writes one table per frame
(``<profile>_Results/Data/<frame>.csv``) whose rows are
``c, crop_name, ..., area, x, y, w, h, saved, ...`` with x, y the top-left
corner in the 2560x2560 frame. The EcoTaxa ``file_name`` is that ``crop_name``.

Background: per block of ``--block`` consecutive water frames, the C++
segmenter's estimator (max over consecutive pairs of their per-pixel min),
scaled to each frame by matching the 98th percentile of the central field.
Use this on CLEAN profiles only -- in dense ones the background keeps the haze.

Mask: ``t`` smoothed (sigma 1); strong shadow (``1 - 8*noise``, noise = MAD
of t in the frame centre) connected to the object's box, plus weak shadow
(``4*noise``) within 6 px of it, holes filled. Padding starts at
max(--pad, size/4) and doubles while the mask still reaches the patch edge. It is a label, not a cut line: the generator pastes the whole patch
and only fades it out away from the mask.

Output: ``patches/<object_id>.npz`` (t float16, mask uint8) and ``index.csv``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deconv_fullframes import CENTRE, cpp_background, in_water, list_frames, read  # noqa: E402

DEFAULT_DATA_ROOT = "/media/veit/T710_data/pisco_processed/ATAIR-BSH_v5alucyd_iso"
DEFAULT_RAW_ROOT = "/mnt/filer/202604_ATAIR-BSH/202604_ATAIR-BSB_PISCO"


def frame_of(crop_name: str) -> str:
    """'20260423-17190414_005.662bar_09.45C_15.png' -> '20260423-17190414_005.662bar_09.45C'"""
    return crop_name.rsplit("_", 1)[0]


def index_data_tables(data_root: str) -> dict[str, str]:
    """frame stem -> path of its per-frame object table."""
    out = {}
    for prof in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, prof, f"{prof}_Results", "Data")
        if os.path.isdir(d):
            for f in os.listdir(d):
                out[f[:-4]] = os.path.join(d, f)
    return out


def boxes_of(table: str) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    with open(table) as fh:
        for row in csv.reader(fh):
            if len(row) > 11 and row[11] == "1":
                out[row[1]] = tuple(int(float(v)) for v in row[7:11])
    return out


def make_mask(t: np.ndarray, core: tuple[int, int, int, int], noise: float,
              k_strong: float = 8.0, k_weak: float = 4.0, grow: int = 6) -> np.ndarray:
    """Two-level mask: strong shadow connected to the object's box, plus weak
    shadow only within ``grow`` px of it (keeps the near halo, does not bleed
    into faint out-of-focus structure next to the object)."""
    ts = cv.GaussianBlur(t, (0, 0), 1.0)
    x0, y0, x1, y1 = core
    strong = ts < 1.0 - max(0.05, k_strong * noise)
    weak = ts < 1.0 - max(0.02, k_weak * noise)
    faint = ts < 1.0 - max(0.015, 2.5 * noise)
    core_m = np.zeros(t.shape, bool)
    core_m[y0:y1, x0:x1] = True
    seeds = None
    for level in (strong, weak, faint):
        lab, _ = ndi.label(level, structure=np.ones((3, 3)))
        ids = np.unique(lab[core_m])
        ids = ids[ids > 0]
        if len(ids):
            seeds = np.isin(lab, ids)
            seeds_level = level
            if level is not strong:    # faint object: do not wander off its box
                seeds &= ndi.binary_dilation(core_m, iterations=grow)
            break
    if seeds is None or not seeds.any():
        return np.zeros(t.shape, np.uint8)
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    near = cv.dilate(seeds.astype(np.uint8), disk).astype(bool)
    m = seeds | ((weak if seeds_level is not faint else faint) & near)
    lab, _ = ndi.label(m, structure=np.ones((3, 3)))
    ids = np.unique(lab[seeds])
    m = np.isin(lab, ids[ids > 0])
    return ndi.binary_fill_holes(m).astype(np.uint8)


def touches_border(mask: np.ndarray, frame_box) -> bool:
    """Mask on a patch edge that is not also the frame edge."""
    X0, Y0, X1, Y1 = frame_box
    return ((mask[0].any() and Y0 > 0) or (mask[-1].any() and Y1 < 2560)
            or (mask[:, 0].any() and X0 > 0) or (mask[:, -1].any() and X1 < 2560))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--pad", type=int, default=30, help="minimum padding; grows to 1/4 of the object size")
    ap.add_argument("--max-pad", type=int, default=240)
    ap.add_argument("--block", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "patches"), exist_ok=True)
    tables = index_data_tables(a.data_root)
    with open(a.labels) as fh:
        labels = list(csv.DictReader(fh))

    # group labelled crops by profile and frame
    todo = defaultdict(lambda: defaultdict(list))
    missing = []
    for r in labels:
        fr = frame_of(r["file_name"])
        if fr not in tables:
            missing.append((r["object_id"], "no data table"))
            continue
        prof = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(tables[fr]))))
        todo[prof][fr].append(r)
    print(f"{len(labels)} labels, {sum(len(v) for p in todo.values() for v in p.values())} "
          f"matched to {len(todo)} profiles", flush=True)

    rows_out = []
    for prof, frames_needed in todo.items():
        raw_dir = os.path.join(a.raw_root, prof.split("-")[0], prof)
        files = list_frames(raw_dir)
        stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
        with ThreadPoolExecutor(16) as ex:
            imgs = list(ex.map(read, files))
        water = [i for i, im in enumerate(imgs) if in_water(im)]
        print(f"{prof}: {len(files)} frames, {len(water)} in water, "
              f"{len(frames_needed)} frames with labelled objects", flush=True)
        blocks = [water[i:i + a.block] for i in range(0, len(water), a.block)]
        if len(blocks) > 1 and len(blocks[-1]) < a.block // 2:
            tail = blocks.pop()
            blocks[-1] = blocks[-1] + tail
        for blk in blocks:
            bg = cpp_background([imgs[i] for i in blk]).astype(np.float32)
            bg_ref = np.percentile(bg[CENTRE], 98)
            for i in blk:
                fr = stems[i]
                if fr not in frames_needed:
                    continue
                img = imgs[i].astype(np.float32)
                gain = np.percentile(img[CENTRE], 98) / bg_ref
                t = img / np.maximum(bg * gain, 1.0)
                c = t[900:1660, 900:1660]
                noise = 1.4826 * np.median(np.abs(c - np.median(c)))
                boxes = boxes_of(tables[fr])
                for r in frames_needed.pop(fr):
                    if r["file_name"] not in boxes:
                        missing.append((r["object_id"], "crop not in data table"))
                        continue
                    x, y, w, h = boxes[r["file_name"]]
                    pad = max(a.pad, max(w, h) // 4)
                    while True:
                        X0, Y0 = max(0, x - pad), max(0, y - pad)
                        X1, Y1 = min(2560, x + w + pad), min(2560, y + h + pad)
                        patch = t[Y0:Y1, X0:X1]
                        mask = make_mask(patch, (x - X0, y - Y0, x - X0 + w, y - Y0 + h), noise)
                        edge = touches_border(mask, (X0, Y0, X1, Y1))
                        if not edge or pad >= a.max_pad:
                            break
                        pad = min(2 * pad, a.max_pad)
                    if not mask.any():
                        missing.append((r["object_id"], "empty mask"))
                        continue
                    oid = r["object_id"].replace(".png", "")
                    np.savez_compressed(os.path.join(a.out, "patches", oid + ".npz"),
                                        t=np.clip(patch, 0, 2).astype(np.float16), mask=mask)
                    rows_out.append(dict(object_id=oid, label=r["label"], project_id=r["project_id"],
                                         profile=prof, frame=fr, x=x, y=y, w=w, h=h,
                                         patch_x=X0, patch_y=Y0, patch_w=X1 - X0, patch_h=Y1 - Y0,
                                         mask_area=int(mask.sum()), pad=pad, mask_on_edge=int(edge), min_t=float(patch.min()),
                                         noise=float(noise), gain=float(gain)))
        for fr, rs in frames_needed.items():
            for r in rs:
                missing.append((r["object_id"], "frame not in water / not found"))

    with open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        wr.writeheader()
        wr.writerows(rows_out)
    with open(os.path.join(a.out, "missing.csv"), "w") as fh:
        fh.write("object_id,reason\n" + "".join(f"{o},{why}\n" for o, why in missing))
    print(f"wrote {len(rows_out)} patches; {len(missing)} not extracted (see missing.csv)")


if __name__ == "__main__":
    main()
