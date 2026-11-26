#!/usr/bin/env python3
"""Generate synthetic raw PIScO frames for instance-segmentation training.

Builds 2560x2560 8-bit frames in the RAW domain (no background correction, no
deconvolution) from transmittance patches made by ``extract_transmittance.py``:

    frame = background * gain * prod_i t_i + noise

* background: a clean whole-profile background (e.g. 20260423-1705 frames
  470:, from ``deconv_fullframes.py --background-only``), scaled so the
  finished scene's central median hits a random --brightness (real ATAIR:
  1750 at 115-137, 1705 at 148-175).
* haze (unlabelled): dirt patches (Unknowns / unidentifiable / not-living)
  rescaled 0.8-2.5x, blurred and reduced in contrast, 6-14k per dirty scene,
  for the material that fills dense profiles. The defaults were fitted to
  20260426-1750: per-scale structure (std of the log image between Gaussian
  scales 1/3/8/24/64 px) 1.35/0.80/1.28/3.66/6.01 % synthetic vs
  1.37/0.80/1.40/3.61/5.28 % real. Most scenes are dirty, some are clean.
* dirt (labelled ``not-living``): the same dirt patches, sharp.
* living objects (labelled with their class), sampled with probability
  ~ count**0.5 per class so rare classes are not drowned out. A share is
  placed overlapping a previously placed object, so "multiples" occur; each
  keeps its own full mask (shadows multiply, both stay visible).
* noise: Gaussian, sigma 1-2 grey levels (fitted with the haze).

Each patch is faded out away from its mask (dilate 8 px, blur sigma 3), so the
source frame's noise and neighbours do not come along, and rotated by a random
angle (telecentric optics: no rescaling of labelled objects).

Train/val split is by source object (stable hash of object_id), so validation
scenes contain only objects never seen in training.

Output:  <out>/{train,val}/images/*.png, <out>/{train,val}/annotations.json
(COCO, polygons), <out>/previews/*.jpg with outlines for a few scenes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from multiprocessing import Pool

import cv2 as cv
import numpy as np

DEFAULT_CLASSES = {
    "Copepoda": "Copepoda", "pluteus": "pluteus", "Appendicularia": "Appendicularia",
    "Cnidaria": "Cnidaria", "gelatinous": "gelatinous", "Chaetognatha": "Chaetognatha",
    "Ctenophora": "Ctenophora",
    "Unknowns": "not-living", "unidentifiable": "not-living", "not-living": "not-living",
    "bubble": "not-living",
}
DIRT_LABELS = {"Unknowns", "unidentifiable", "not-living"}

G = {}   # filled before the pool forks: patches, background, settings


def is_val(object_id: str, frac: float) -> bool:
    h = int(hashlib.md5(object_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000 < frac


def load(objects_dir: str, classes: dict, min_area: int):
    rows = list(csv.DictReader(open(os.path.join(objects_dir, "index.csv"))))
    objs = []
    for r in rows:
        if r["label"] not in classes and r["label"] not in DIRT_LABELS:
            continue
        z = np.load(os.path.join(objects_dir, "patches", r["object_id"] + ".npz"))
        t = z["t"].astype(np.float32)
        m = z["mask"].astype(np.uint8)
        # object cut by its patch or by the source frame's edge: pasted, it
        # would show a straight edge
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (17, 17))
        w = cv.GaussianBlur(cv.dilate(m, disk).astype(np.float32), (0, 0), 3)
        w = np.maximum(w, m)
        t = 1 + (t - 1) * w                      # fade out away from the object
        objs.append(dict(id=r["object_id"], label=r["label"], cls=classes.get(r["label"]),
                         t=t, mask=m, area=int(m.sum()),
                         labelled=int(m.sum()) >= min_area))
    return objs


def rotate(t, m, angle, scale=1.0):
    h, w = t.shape
    d = int(np.ceil(np.hypot(h, w) * scale)) + 2
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += d / 2 - w / 2
    M[1, 2] += d / 2 - h / 2
    tr = cv.warpAffine(t, M, (d, d), flags=cv.INTER_LINEAR, borderValue=1.0)
    mr = cv.warpAffine(m, M, (d, d), flags=cv.INTER_NEAREST, borderValue=0)
    return tr, mr


def paste(T, t, cx, cy):
    """Multiply t into T centred at (cx, cy); returns the canvas slice used."""
    h, w = t.shape
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(T.shape[1], x0 + w), min(T.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return None
    T[Y0:Y1, X0:X1] *= t[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return (x0, y0, X0, Y0, X1, Y1)


def random_point(rng, R, cxy):
    r = R * np.sqrt(rng.random())
    a = rng.random() * 2 * np.pi
    return cxy[0] + r * np.cos(a), cxy[1] + r * np.sin(a)


def scene(args):
    idx, split, seed = args
    s = G["settings"]
    rng = np.random.default_rng(seed)
    pools = G["pools"][split]
    H = W = 2560
    T = np.ones((H, W), np.float32)
    R, cxy = G["fov_r"] - 40, G["fov_c"]

    # haze: blurred, enlarged, low-contrast dirt; unlabelled
    dirty = rng.random() < s.p_dirty
    n_haze = int(rng.integers(*s.n_haze_dirty)) if dirty else int(rng.integers(*s.n_haze_clean))
    for _ in range(n_haze):
        o = pools["dirt"][rng.integers(len(pools["dirt"]))]
        sc = rng.uniform(*s.haze_scale)
        tr, _ = rotate(o["t"], o["mask"], rng.uniform(0, 360), sc)
        sig = rng.uniform(*s.haze_blur) * sc
        p = int(3 * sig) + 1                      # room for the blur, else a straight edge
        tr = cv.copyMakeBorder(tr, p, p, p, p, cv.BORDER_CONSTANT, value=1.0)
        tr = cv.GaussianBlur(tr, (0, 0), sig)
        a = rng.uniform(*s.haze_contrast)
        paste(T, 1 - a * (1 - tr), *random_point(rng, R + 100, cxy))

    instances = []

    def place(o, near=None):
        tr, mr = rotate(o["t"], o["mask"], rng.uniform(0, 360))
        if rng.random() < 0.5:
            tr, mr = tr[:, ::-1].copy(), mr[:, ::-1].copy()
        if near is not None:
            ys, xs = np.nonzero(mr)
            size = max(np.ptp(ys), np.ptp(xs), 8)
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.25, 0.8) * (size + near["size"]) / 2
            cx, cy = near["cx"] + dist * np.cos(ang), near["cy"] + dist * np.sin(ang)
        else:
            cx, cy = random_point(rng, R, cxy)
        box = paste(T, tr, cx, cy)
        if box is None or not o["labelled"]:
            return
        x0, y0, X0, Y0, X1, Y1 = box
        full = np.zeros((H, W), np.uint8)
        full[Y0:Y1, X0:X1] = mr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        full &= G["fov_mask"]
        if full.sum() < s.min_area:
            return
        ys, xs = np.nonzero(mr)
        instances.append(dict(cls=o["cls"], label=o["label"], src=o["id"], mask=full,
                              cx=cx, cy=cy, size=max(np.ptp(ys), np.ptp(xs), 8)))

    for _ in range(int(rng.integers(*s.n_dirt))):
        place(pools["dirt"][rng.integers(len(pools["dirt"]))])
    n_living = int(rng.integers(*s.n_living))
    labels, probs = pools["living_labels"], pools["living_p"]
    for _ in range(n_living):
        lab = labels[rng.choice(len(labels), p=probs)]
        cand = pools["living"][lab]
        o = cand[rng.integers(len(cand))]
        near = None
        if instances and rng.random() < s.p_overlap:
            near = instances[rng.integers(len(instances))]
        place(o, near)

    lit = G["bg"] * T
    gain = rng.uniform(*s.brightness) / float(np.median(lit[1000:1560, 1000:1560]))
    img = lit * gain + rng.normal(0, rng.uniform(*s.noise), (H, W)).astype(np.float32)
    img = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    name = f"{split}_{idx:06d}.png"
    cv.imwrite(os.path.join(s.out, split, "images", name), img)

    anns = []
    for k, ins in enumerate(instances):
        cs, _ = cv.findContours(ins["mask"], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        polys = [c.reshape(-1).tolist() for c in cs if len(c) >= 3]
        if not polys:
            continue
        x, y, w, h = cv.boundingRect(ins["mask"])
        anns.append(dict(category=ins["cls"], segmentation=polys, bbox=[x, y, w, h],
                         area=int(ins["mask"].sum()), iscrowd=0, source_object=ins["src"],
                         source_label=ins["label"]))
    if idx < s.previews:
        prev = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for a_ in anns:
            col = (0, 0, 255) if a_["category"] == "not-living" else (0, 200, 0)
            for p in a_["segmentation"]:
                cv.polylines(prev, [np.array(p).reshape(-1, 1, 2)], True, col, 2)
        cv.imwrite(os.path.join(s.out, "previews", f"{split}_{idx:06d}.jpg"), prev,
                   [cv.IMWRITE_JPEG_QUALITY, 85])
    return dict(file_name=name, width=W, height=H, gain=gain, dirty=bool(dirty),
                n_haze=n_haze), anns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--objects", required=True, help="output dir of extract_transmittance.py")
    ap.add_argument("--background", required=True, help="clean whole-profile background PNG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of source objects held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--previews", type=int, default=6)
    ap.add_argument("--classes", help="JSON {ecotaxa_label: class}; default: 7 living + not-living")
    ap.add_argument("--min-area", type=int, default=30, help="smallest labelled mask (px)")
    ap.add_argument("--brightness", type=float, nargs=2, default=(110, 180),
                    help="target median grey of the central field (1750: 115-137, 1705: 148-175)")
    ap.add_argument("--noise", type=float, nargs=2, default=(1.0, 2.0))
    ap.add_argument("--p-dirty", type=float, default=0.75)
    ap.add_argument("--n-haze-dirty", type=int, nargs=2, default=(6000, 14000))
    ap.add_argument("--n-haze-clean", type=int, nargs=2, default=(0, 300))
    ap.add_argument("--haze-scale", type=float, nargs=2, default=(0.8, 2.5))
    ap.add_argument("--haze-blur", type=float, nargs=2, default=(0.5, 3.0), help="sigma per unit scale")
    ap.add_argument("--haze-contrast", type=float, nargs=2, default=(0.3, 1.0))
    ap.add_argument("--n-dirt", type=int, nargs=2, default=(20, 200))
    ap.add_argument("--n-living", type=int, nargs=2, default=(5, 60))
    ap.add_argument("--p-overlap", type=float, default=0.25)
    s = ap.parse_args()

    classes = json.load(open(s.classes)) if s.classes else DEFAULT_CLASSES
    objs = load(s.objects, classes, s.min_area)
    bg = cv.imread(s.background, cv.IMREAD_GRAYSCALE).astype(np.float32)

    # field of view: the bright disc of the background
    fov = (bg > 0.6 * np.median(bg[1000:1560, 1000:1560])).astype(np.uint8)
    n, lab, stats, cents = cv.connectedComponentsWithStats(fov)
    big = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    fov = (lab == big).astype(np.uint8)
    G.update(bg=bg, fov_mask=fov, fov_c=tuple(cents[big]),
             fov_r=float(np.sqrt(stats[big, cv.CC_STAT_AREA] / np.pi)), settings=s)

    cats = sorted(set(classes.values()))
    pools = {}
    for split in ("train", "val"):
        mine = [o for o in objs if is_val(o["id"], s.val_frac) == (split == "val")]
        living = {}
        for o in mine:
            if o["cls"] and o["cls"] != "not-living":
                living.setdefault(o["label"], []).append(o)
        labels = sorted(living)
        p = np.array([len(living[k]) ** 0.5 for k in labels])
        pools[split] = dict(dirt=[o for o in mine if o["label"] in DIRT_LABELS],
                            living=living, living_labels=labels, living_p=p / p.sum())
        print(f"{split}: {len(pools[split]['dirt'])} dirt, "
              + ", ".join(f"{k} {len(living[k])}" for k in labels), flush=True)
    G["pools"] = pools
    print(f"FOV radius {G['fov_r']:.0f} px, centre ({G['fov_c'][0]:.0f}, {G['fov_c'][1]:.0f})")

    os.makedirs(os.path.join(s.out, "previews"), exist_ok=True)
    for split, n in (("train", s.n_train), ("val", s.n_val)):
        os.makedirs(os.path.join(s.out, split, "images"), exist_ok=True)
        jobs = [(i, split, s.seed * 1_000_003 + i + (0 if split == "train" else 10**8)) for i in range(n)]
        with Pool(s.workers) as pool:
            res = pool.map(scene, jobs, chunksize=4)
        coco = dict(images=[], annotations=[],
                    categories=[dict(id=i + 1, name=c) for i, c in enumerate(cats)],
                    info=dict(generator="detector/synth_scenes.py", settings=vars(s)))
        cid = {c: i + 1 for i, c in enumerate(cats)}
        aid = 1
        for img_id, (im, anns) in enumerate(res, 1):
            coco["images"].append(dict(id=img_id, **im))
            for a_ in anns:
                coco["annotations"].append(dict(id=aid, image_id=img_id,
                                                category_id=cid[a_.pop("category")], **a_))
                aid += 1
        with open(os.path.join(s.out, split, "annotations.json"), "w") as fh:
            json.dump(coco, fh)
        per = {}
        for a_ in coco["annotations"]:
            per[a_["category_id"]] = per.get(a_["category_id"], 0) + 1
        print(f"{split}: {n} scenes, {len(coco['annotations'])} instances: "
              + ", ".join(f"{c} {per.get(cid[c], 0)}" for c in cats), flush=True)


if __name__ == "__main__":
    main()
