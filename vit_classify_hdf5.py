#!/usr/bin/env python3
"""Run a ViT over the crops in the C++ segmenter's result.h5.

One stage of the dual-ViT setup. Chain two of these -- a binary
living/not-living pass over everything, then a taxon pass restricted to the
living ones -- rather than baking the two-stage structure into one tool:

    vit_classify_hdf5.py results/ -o stage1/ --model .../binary/best_model \\
        --emit-embeddings
    vit_classify_hdf5.py results/ -o stage2/ --model .../living/best_model \\
        --restrict-to stage1/predictions.csv --restrict-labels living

Crops are read from ``result.h5`` rather than from exported PNGs. That skips
the scale-bar strip and the pad/unpad round trip entirely, and keeps neighbour
isolation a choice made at inference time rather than a property baked into
files on disk.

**Preprocessing mirrors ``utils.custom_image_processor`` exactly** -- longest
edge to 224 via ``torchvision.transforms.functional.resize``, centre-pad to
224x224 with white, normalise to [-1, 1]. A classifier is only valid for crops
that look like its training data, so this must not drift; see
``training/README.md`` for what a mismatch has cost before.

``--emit-embeddings`` writes the encoder's CLS vector -- the exact tensor the
classification head sees -- for every crop the stage looked at. Those feed
EcoTaxa's similarity search. Take them from the **binary** stage: it is the
only one that sees every object, and EcoTaxa disables similarity search
entirely for a project where any object lacks a feature vector.

This module deliberately does not import ``utils``: that pulls in pandas,
matplotlib, umap and datasets at module scope, none of which are installed on
the processing workers.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

from crops import tight_masked
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import ViTForImageClassification

# Matches utils.custom_image_processor. Do not change without retraining.
TARGET_SIZE = 224
PADDING_COLOUR = 255


def preprocess(crop: np.ndarray) -> torch.Tensor:
    """One crop -> a normalised 3x224x224 tensor.

    Mirrors utils.custom_image_processor: RGB, longest edge to 224, centre-pad
    with white, scale to [0,1] then normalise with mean=std=0.5 to land in
    [-1, 1]. Returns None only for an empty crop.

    The short edge is kept at least one pixel: a streak more than 224 times
    longer than wide (5 x 1298 on MSM147_002-11) would otherwise shrink to
    nothing and drop out, and EcoTaxa refuses similarity search for a whole
    project when a single object has no feature vector.
    """
    if crop.size == 0:
        return None
    image = Image.fromarray(crop).convert("RGB")

    width, height = image.size
    scale = TARGET_SIZE / max(width, height)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = TF.resize(image, (new_h, new_w))

    pad_left = (TARGET_SIZE - new_w) // 2
    pad_top = (TARGET_SIZE - new_h) // 2
    padded = TF.pad(
        resized,
        [pad_left, pad_top, TARGET_SIZE - new_w - pad_left, TARGET_SIZE - new_h - pad_top],
        fill=PADDING_COLOUR,
    )

    tensor = TF.to_tensor(padded)
    return TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def wanted_by_frame(keep: set[str] | None) -> dict[str, set[int]] | None:
    """Group object ids by their frame, as {frame: {row index, ...}}.

    ``object_id`` is ``<frame>_<n>`` with n counting from 1, and frame names
    themselves contain underscores, so the split is from the right. Lets a
    restricted pass skip whole frames without reading their pixel data.
    """
    if keep is None:
        return None
    grouped: dict[str, set[int]] = {}
    for object_id in keep:
        frame, _, index = object_id.rpartition("_")
        if frame and index.isdigit():
            grouped.setdefault(frame, set()).add(int(index) - 1)
    return grouped


def iter_crops(h5_path: Path, isolate: bool, keep: set[str] | None):
    """Yield (object_id, crop) for every object, in frame order.

    Crops are stored bright on a dark ground; inverted here so the model sees
    what it was trained on. ``isolate`` cuts each crop to its mask's bounding
    box and whites out everything outside the mask - the crop the python route
    saved and the ViTs were trained on. Off, the model sees the stored box
    as-is, with the segmenter's padding and any neighbours in it. One of the
    axes a classifier is bound to, so it has to match the training regime.
    """
    wanted = wanted_by_frame(keep)
    with h5py.File(h5_path, "r") as f:
        for frame in sorted(f.keys()):
            if wanted is not None and frame not in wanted:
                continue           # nothing wanted here; do not read its pixels
            group = f[frame]
            if not isinstance(group, h5py.Group) or "boundingBoxW" not in group:
                continue
            widths = np.asarray(group["boundingBoxW"])
            heights = np.asarray(group["boundingBoxH"])
            offsets = np.asarray(group["offsets"])
            pixels = np.asarray(group["cropPixelData"])
            masks = np.asarray(group["maskPixelData"]) if isolate and "maskPixelData" in group else None

            for i in range(len(offsets)):
                if wanted is not None and i not in wanted[frame]:
                    continue
                object_id = f"{frame}_{i + 1}"
                w, h, off = int(widths[i]), int(heights[i]), int(offsets[i])
                n_px = w * h
                if w <= 0 or h <= 0 or off + n_px > pixels.size:
                    continue
                crop = (255 - pixels[off:off + n_px].reshape(h, w)).astype(np.uint8)
                if masks is not None:
                    mask = masks[off:off + n_px].reshape(h, w)
                    crop, _, _, _ = tight_masked(crop, mask)
                yield object_id, crop


def load_restriction(path: Path, labels: set[str]) -> tuple[set[str], list[dict]]:
    """Split a previous stage's rows into (ids to classify, rows to carry).

    The carried rows are the ones this stage will not look at -- typically the
    binary pass's not-living verdicts. Keeping them means this stage's output
    describes every object, so a downstream task needs one reference to it
    rather than one per stage. That matters because the Hub only resolves
    {"$from": ...} inside dicts, not inside lists, so two references cannot be
    chained into a single setting.
    """
    keep: set[str] = set()
    carried: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("top1") or "").strip().lower() in labels:
                keep.add(row["object_id"])
            else:
                carried.append(row)
    return keep, carried


def index_crops(h5_path: Path, keep: set[str] | None) -> list[tuple[str, int, int, int, int]]:
    """Every object as (frame, index, w, h, offset), in the order iter_crops
    would yield them -- read from the bounding-box tables alone, so the
    pixels are only touched by whoever loads the crop."""
    wanted = wanted_by_frame(keep)
    items = []
    with h5py.File(h5_path, "r") as f:
        for frame in sorted(f.keys()):
            if wanted is not None and frame not in wanted:
                continue
            group = f[frame]
            if not isinstance(group, h5py.Group) or "boundingBoxW" not in group:
                continue
            widths = np.asarray(group["boundingBoxW"])
            heights = np.asarray(group["boundingBoxH"])
            offsets = np.asarray(group["offsets"])
            n_pixels = group["cropPixelData"].shape[0]
            for i in range(len(offsets)):
                if wanted is not None and i not in wanted[frame]:
                    continue
                w, h, off = int(widths[i]), int(heights[i]), int(offsets[i])
                if w <= 0 or h <= 0 or off + w * h > n_pixels:
                    continue
                items.append((frame, i, w, h, off))
    return items


class CropDataset(torch.utils.data.Dataset):
    """Crops as model-ready tensors, loadable from DataLoader workers.

    Each worker opens the HDF5 file itself and keeps the last frame's pixel
    arrays, so consecutive crops of one frame cost one read. The crop is
    inverted (and isolated if asked) exactly as iter_crops does it.
    """

    def __init__(self, h5_path: Path, items, isolate: bool):
        self.h5_path, self.items, self.isolate = h5_path, items, isolate
        self._file = None
        self._frame = None
        self._pixels = None
        self._masks = None

    def __len__(self):
        return len(self.items)

    def _arrays(self, frame):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        if frame != self._frame:
            group = self._file[frame]
            self._pixels = np.asarray(group["cropPixelData"])
            self._masks = (np.asarray(group["maskPixelData"])
                           if self.isolate and "maskPixelData" in group else None)
            self._frame = frame
        return self._pixels, self._masks

    def __getitem__(self, idx):
        frame, i, w, h, off = self.items[idx]
        pixels, masks = self._arrays(frame)
        n_px = w * h
        crop = (255 - pixels[off:off + n_px].reshape(h, w)).astype(np.uint8)
        if masks is not None:
            mask = masks[off:off + n_px].reshape(h, w)
            crop, _, _, _ = tight_masked(crop, mask)
        tensor = preprocess(crop)
        return f"{frame}_{i + 1}", tensor


def _collate(samples):
    """Drop crops preprocess() refused; count them."""
    kept = [(oid, t) for oid, t in samples if t is not None]
    skipped = len(samples) - len(kept)
    if not kept:
        return [], None, skipped
    return [oid for oid, _ in kept], torch.stack([t for _, t in kept]), skipped


def choose_precision(precision: str, device: str) -> torch.dtype | None:
    """fp16 autocast where it pays: tensor cores (compute capability 7.0+).
    A Pascal card runs fp16 slower than fp32, so 'auto' keeps fp32 there."""
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability()
    return torch.float16 if major >= 7 else None


def classify(h5_path: Path, out_dir: Path, model_path: str, *,
             isolate: bool = False, batch_size: int = 64, top_k: int = 5,
             emit_embeddings: bool = False, keep: set[str] | None = None,
             carried: list[dict] | None = None, workers: int | None = None,
             precision: str = "auto") -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = choose_precision(precision, device)
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    print(f"  device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
          + f", {'fp16 autocast' if dtype else 'fp32'}, {workers} loader worker(s)")
    model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
    id2label = model.config.id2label
    k = min(top_k, len(id2label))
    print(f"  model : {model_path}  ({len(id2label)} classes)")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    n_skipped = 0

    # Preprocessing runs in worker processes while the GPU is busy with the
    # previous batch; the main process only stacks and forwards.
    items = index_crops(h5_path, keep)
    loader = torch.utils.data.DataLoader(
        CropDataset(h5_path, items, isolate), batch_size=batch_size,
        num_workers=workers, pin_memory=(device == "cuda"), collate_fn=_collate,
        persistent_workers=False)
    autocast = (torch.autocast("cuda", dtype=dtype) if dtype
                else contextlib.nullcontext())
    done = 0
    started = time.perf_counter()
    for batch_ids, pixel_values, skipped in loader:
        n_skipped += skipped
        if pixel_values is None:
            continue
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.inference_mode(), autocast:
            # Call the encoder directly so the embedding is exactly the tensor
            # the classification head consumes, from a single forward pass.
            encoded = model.vit(pixel_values).last_hidden_state[:, 0]
            logits = model.classifier(encoded)
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, k, dim=-1)
        top_indices, top_values = top.indices.cpu(), top.values.cpu()
        for j, object_id in enumerate(batch_ids):
            row = {"object_id": object_id}
            for rank in range(k):
                row[f"top{rank + 1}"] = id2label[int(top_indices[j, rank])]
                row[f"prob{rank + 1}"] = f"{float(top_values[j, rank]):.6f}"
            rows.append(row)
        if emit_embeddings:
            ids.extend(batch_ids)
            embeddings.append(encoded.float().cpu().numpy())
        done += len(batch_ids)
        if done % 2000 < len(batch_ids) or done == len(items):
            rate = done / (time.perf_counter() - started)
            print(f"  {done}/{len(items)} crops ({rate:.0f}/s)", flush=True)

    if carried:
        # Rows this stage skipped, kept verbatim so the file covers every
        # object. Written after, then sorted, so order does not depend on which
        # stage produced a row.
        rows.extend(carried)
        rows.sort(key=lambda r: r["object_id"])
        print(f"  carried forward: {len(carried)} rows from the earlier stage")

    columns = ["object_id"] + [f"top{i + 1}" for i in range(k)] + [f"prob{i + 1}" for i in range(k)]
    for row in rows:                       # a carried row may have more columns
        for column in row:
            if column not in columns:
                columns.append(column)
    predictions_path = out_dir / "predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {predictions_path.name}: {len(rows)} objects"
          + (f", {n_skipped} unprocessable" if n_skipped else ""))

    if emit_embeddings and embeddings:
        matrix = np.concatenate(embeddings, axis=0)
        np.savez_compressed(out_dir / "embeddings.npz",
                            object_ids=np.array(ids), embeddings=matrix)
        print(f"  embeddings.npz: {matrix.shape[0]} x {matrix.shape[1]}")

    # A label histogram is the cheapest way to notice an out-of-domain run:
    # a collapsed distribution is the signature of crops the model never saw.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["top1"]] = counts.get(row["top1"], 0) + 1
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {label:32s} {n:6d}  ({n / max(len(rows), 1) * 100:.1f}%)")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="result.h5, or a directory containing one")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    ap.add_argument("--model", required=True,
                    help="model directory, or a Hugging Face repo id")
    ap.add_argument("--isolate-crops", action=argparse.BooleanOptionalAction, default=True,
                    help="Cut each crop to its mask and white out the rest, as the python "
                         "route saved crops and the ViTs were trained. --no-isolate-crops "
                         "feeds the stored box as-is.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes preprocessing crops while the GPU runs "
                         "(default: up to 8, one less than the cores)")
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto uses fp16 only on GPUs with tensor cores")
    ap.add_argument("--emit-embeddings", action="store_true",
                    help="write the encoder CLS vectors, for EcoTaxa "
                         "similarity search. Use on the stage that sees every "
                         "object, i.e. the binary one.")
    ap.add_argument("--restrict-to", default=None,
                    help="predictions.csv from an earlier stage, or that stage's output directory")
    ap.add_argument("--carry-forward", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="with --restrict-to, keep the earlier stage's rows for "
                         "objects this stage skips, so the output covers every "
                         "object and one reference to it is enough downstream "
                         "(default: on)")
    ap.add_argument("--restrict-labels", default="living",
                    help="comma-separated top1 labels to keep (default: living)")
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

    # Chaining hands over the producing task's directory. pisco_vit_train writes
    # its checkpoint to <output>/best_model beside the evaluation report, so
    # accept either that directory or the model itself.
    model_path = args.model
    candidate = Path(model_path)
    if candidate.is_dir() and not (candidate / "config.json").exists() \
            and (candidate / "best_model" / "config.json").exists():
        model_path = str(candidate / "best_model")
        print(f"Using {model_path}")

    keep = None
    carried = None
    if args.restrict_to:
        labels = {s.strip().lower() for s in args.restrict_labels.split(",") if s.strip()}
        restrict_path = Path(args.restrict_to)
        # Accept the earlier stage's output *directory* as well as the file, so
        # a workflow can chain with {"$from": {...}} and not have to spell out
        # the filename the Hub allocated.
        if restrict_path.is_dir():
            restrict_path = restrict_path / "predictions.csv"
        if not restrict_path.exists():
            print(f"ERROR: {restrict_path} not found", file=sys.stderr)
            return 1
        keep, carried = load_restriction(restrict_path, labels)
        if not args.carry_forward:
            carried = []
        print(f"Restricted to {len(keep)} objects labelled {sorted(labels)}"
              + (f"; carrying {len(carried)} others forward" if carried else ""))
        if not keep:
            print("ERROR: restriction matched no objects", file=sys.stderr)
            return 1

    print(f"Classifying {src}")
    n = classify(src, Path(args.output), model_path,
                 isolate=args.isolate_crops, batch_size=args.batch_size,
                 top_k=args.top_k, emit_embeddings=args.emit_embeddings,
                 keep=keep, carried=carried, workers=args.workers,
                 precision=args.precision)
    if n == 0:
        print("ERROR: nothing classified", file=sys.stderr)
        return 1
    print(f"Done: {n} objects -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
