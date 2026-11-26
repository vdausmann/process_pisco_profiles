#!/usr/bin/env python3
"""Deconvolve the full frames of one profile with LUCYD -- no segmentation.

Writes one PNG per frame (2560x2560, 8 bit, the model's direct output; the
245 white clip the crops get happens later, at cropping). Useful for looking
at whole scenes, e.g. as backgrounds for a synthetic object-detection set.

Every mode computes  cleaned = 255 - |frame - background|  and feeds
cleaned/255 to LUCYD, which is what both segmenter routes do. They differ
only in the background:

  --bg-mode python   pisco_segmenter at b805b9a (pre-hub-background): batches
                     of 200, per-frame rolling window of 12 neighbours, max.
  --bg-mode cpp      pisco_segmenter_cpp: the profile split over --ranks MPI
                     ranks, then consecutive blocks of --stack frames (short
                     tail folded into the last block), background = max over
                     consecutive pairs of their per-pixel min. --stack 0 uses
                     the whole profile as one block.
  --bg-file PATH     one fixed background, e.g. from a cleaner profile, scaled
                     per frame so its --gain-percentile matches the frame's.

Dense profiles (e.g. ATAIR 20260426-1750) have no pair of consecutive frames
that is clear at every pixel, so their own background keeps the haze. There,
build the background on a clean profile from the same days and pass it in:

    # 1. background from a clean reference profile (water frames only)
    deconv_fullframes.py --background-only bg_1705.png \\
        /mnt/filer/.../20260423/20260423-1705
    # 2. deconvolve the dense profile against it
    deconv_fullframes.py --water-only --bg-file bg_1705.png \\
        --model .../lucyd-edof-plankton_v5a_20260807.pth \\
        /mnt/filer/.../20260426/20260426-1750 \\
        /mnt/filer/.../20260426/20260426-1750_deconv_v5a_bg1705

Check that the reference has no brightness step (20260423-1705 jumps from
~150 to ~175 at frame ~467; use --frames to take one side of it), and that
the window dirt matches: same cruise days, same instrument state.

Run in the pisco conda env (ROCm torch). The LUCYD class is imported from
pisco_segmenter; set PISCO_SEGMENTER_DIR if it is not at the default path.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from PIL import Image

PISCO_SEGMENTER_DIR = os.environ.get(
    "PISCO_SEGMENTER_DIR", os.path.expanduser("~/PIScO_dev/pisco_segmenter/pisco_segmenter"))

N_BG = 12        # python route: bg_size 6, doubled inside correct_img
BATCH = 200      # python route: frames per segmenter batch
CENTRE = (slice(640, 1920, 2), slice(640, 1920, 2))   # central field, subsampled


def time_key(path):
    return float(os.path.basename(path).split("_")[0].split("-")[1])


def list_frames(src):
    files = [os.path.join(src, f) for f in os.listdir(src)
             if f.endswith((".png", ".jpg", ".tif"))]
    return sorted(files, key=time_key)


def read(path):
    with Image.open(path) as im:
        img = np.array(im.convert("L"))
    return cv.resize(img, (2560, 2560))


def in_water(img):
    """False for air and surface-transition frames: saturated (p99>=250),
    dark (mean<=85) or half-and-half (std>=40) in the central field.
    Water frames on ATAIR sit at mean 100-180, p99<=190, std<=35."""
    c = img[640:1920:4, 640:1920:4]
    return np.percentile(c, 99) < 250 and c.mean() > 85 and c.std() < 40


def python_batches(files):
    n = len(files)
    batches = [files[i * BATCH:(i + 1) * BATCH] for i in range(n // BATCH - 1)]
    batches.append(files[BATCH * (n // BATCH - 1):] if n // BATCH >= 1 else files)
    return batches


def python_window(i, n):
    half = N_BG // 2
    start, end = i - half, i + half + 1
    if start < 0:
        end += -start
        start = 0
    if end > n:
        start = max(0, start - (end - n))
        end = n
    return [j for j in range(start, end) if j != i]


def cpp_blocks(n, ranks=4, stack=12):
    """Frame index blocks exactly as pisco_segmenter_cpp's HDF5 reader forms them."""
    active = min(ranks, n // 2)
    base, rem = n // active, n % active
    blocks = []
    for r in range(active):
        start = r * base + min(r, rem)
        count = base + (1 if r < rem else 0)
        nxt = 0
        while nxt < count:
            remaining = count - nxt
            c = min(stack, remaining)
            if remaining > stack and remaining - stack < stack:
                c = remaining          # fold a short tail into this stack
            blocks.append(list(range(start + nxt, start + nxt + c)))
            nxt += c
    return blocks


def cpp_background(stack):
    """max over consecutive pairs of their per-pixel min (minMaxBackgroundComputation)."""
    bg = np.minimum(stack[0], stack[1])
    for i in range(1, len(stack) - 1):
        np.maximum(bg, np.minimum(stack[i], stack[i + 1]), out=bg)
    return bg


def parse_frames(spec, n):
    """'470:' or '1:467' -> range of frame indices (python slice syntax)."""
    if not spec:
        return range(n)
    parts = [int(p) if p else None for p in spec.split(":")]
    return range(n)[slice(*parts)]


def background_only(a):
    files = list_frames(a.src)
    files = [files[i] for i in parse_frames(a.frames, len(files))]
    with ThreadPoolExecutor(16) as ex:
        imgs = list(ex.map(read, files))
    keep = [im for im in imgs if in_water(im)]
    print(f"{len(files)} frames, {len(keep)} in water used for the background")
    cv.imwrite(a.background_only, cpp_background(keep))
    print("wrote", a.background_only)


def deconvolve(a):
    import torch
    sys.path.insert(0, PISCO_SEGMENTER_DIR)
    from lucyd import LUCYD

    os.makedirs(a.dst, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    model = LUCYD(num_res=1)
    model.load_state_dict(torch.load(a.model, map_location=device))
    model.to(device).eval()

    files = list_frames(a.src)
    files = [files[i] for i in parse_frames(a.frames, len(files))]
    print(f"{len(files)} frames", flush=True)
    if a.water_only:
        flags = [in_water(read(f)) for f in files]
        dropped = [os.path.basename(f) for f, ok in zip(files, flags) if not ok]
        files = [f for f, ok in zip(files, flags) if ok]
        print(f"water-only: dropped {len(dropped)} frames: {dropped}", flush=True)
        with open(os.path.join(a.dst, "_dropped_frames.txt"), "w") as fh:
            fh.write("\n".join(dropped) + "\n")

    ext_bg = None
    if a.bg_file:
        ext_bg = cv.imread(a.bg_file, cv.IMREAD_GRAYSCALE).astype(np.float32)
        ext_ref = np.percentile(ext_bg[CENTRE], a.gain_percentile)
        gains = []
        batches = [files[i:i + BATCH] for i in range(0, len(files), BATCH)]  # no window needed
    elif a.bg_mode == "cpp":
        blocks = [list(range(len(files)))] if a.stack == 0 else cpp_blocks(len(files), a.ranks, a.stack)
        batches = [[files[i] for i in blk] for blk in blocks]
    else:
        batches = python_batches(files)

    done = skipped = 0
    pending = []

    def flush():
        nonlocal done
        if not pending:
            return
        x = torch.cat([torch.from_numpy(c / 255).unsqueeze(0).unsqueeze(0)
                       for c, _ in pending]).float().to(device)
        with torch.no_grad():
            y, _, _ = model(x)
        y = y.detach().cpu().numpy()
        for k, (_, fn) in enumerate(pending):
            out = (y[k, 0] * 255).astype(np.uint8)
            name = os.path.splitext(os.path.basename(fn))[0] + ".png"
            cv.imwrite(os.path.join(a.dst, name), out)
            done += 1
        pending.clear()

    for b, batch in enumerate(batches, 1):
        imgs = [read(f) for f in batch]
        block_bg = cpp_background(imgs) if ext_bg is None and a.bg_mode == "cpp" else None
        if block_bg is not None and len(batches) == 1:
            cv.imwrite(os.path.join(a.dst, "_background.png"), block_bg)
        for i, (img, fn) in enumerate(zip(imgs, batch)):
            if np.std(img) <= 2:
                print("skip (low stdev):", os.path.basename(fn), flush=True)
                skipped += 1
                continue
            if ext_bg is not None:
                g = np.percentile(img[CENTRE], a.gain_percentile) / ext_ref
                gains.append((os.path.basename(fn), g))
                bg = np.clip(ext_bg * g + 0.5, 0, 255).astype(np.uint8)
            elif block_bg is not None:
                bg = block_bg
            else:
                bg = np.max([imgs[j] for j in python_window(i, len(imgs))
                             if imgs[j].shape == img.shape], axis=0)
            pending.append((cv.bitwise_not(cv.absdiff(img, bg)), fn))
            if len(pending) == a.gpu_batch:
                flush()
        flush()
        print(f"batch {b}/{len(batches)}: {done} written, {skipped} skipped", flush=True)

    if ext_bg is not None:
        with open(os.path.join(a.dst, "_gains.csv"), "w") as fh:
            fh.write("frame,gain\n" + "".join(f"{n},{g:.4f}\n" for n, g in gains))
        gv = np.array([g for _, g in gains])
        print(f"gain min {gv.min():.3f} median {np.median(gv):.3f} max {gv.max():.3f}")
    print("done", done, "written,", skipped, "skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="profile folder with the raw frames")
    ap.add_argument("dst", nargs="?", help="output folder for the deconvolved PNGs")
    ap.add_argument("--model", help="LUCYD .pth (ATAIR production: lucyd-edof-plankton_v5a_20260807.pth)")
    ap.add_argument("--gpu-batch", type=int, default=4)
    ap.add_argument("--bg-mode", choices=["python", "cpp"], default="cpp")
    ap.add_argument("--ranks", type=int, default=4, help="cpp mode: MPI ranks the profile is split over")
    ap.add_argument("--stack", type=int, default=12,
                    help="cpp mode: frames per background block; 0 = whole profile in one block")
    ap.add_argument("--bg-file", help="fixed background image, scaled per frame (overrides --bg-mode)")
    ap.add_argument("--gain-percentile", type=float, default=98.0,
                    help="percentile of the central field matched between frame and --bg-file")
    ap.add_argument("--water-only", action="store_true",
                    help="drop air/surface-transition frames from background and output")
    ap.add_argument("--frames", help="python-slice of the time-sorted frames to use, e.g. '470:'")
    ap.add_argument("--background-only", metavar="OUT_PNG",
                    help="only compute the whole-profile cpp background of the water frames and save it")
    a = ap.parse_args()

    if a.background_only:
        background_only(a)
    else:
        if not (a.dst and a.model):
            ap.error("dst and --model are required unless --background-only is given")
        deconvolve(a)


if __name__ == "__main__":
    main()
