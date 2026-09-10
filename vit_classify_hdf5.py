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
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
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
    [-1, 1]. Returns None for a crop that cannot be resized.
    """
    image = Image.fromarray(crop).convert("RGB")

    width, height = image.size
    scale = TARGET_SIZE / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    if new_w < 1 or new_h < 1:
        return None
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


def iter_crops(h5_path: Path, isolate: bool, keep: set[str] | None):
    """Yield (object_id, crop) for every object, in frame order.

    Crops are stored bright on a dark ground; inverted here so the model sees
    what it was trained on. ``isolate`` masks neighbours out to white using the
    segmenter's own mask -- one of the axes a classifier is bound to, so it has
    to match the training regime.
    """
    with h5py.File(h5_path, "r") as f:
        for frame in sorted(f.keys()):
            group = f[frame]
            if not isinstance(group, h5py.Group) or "boundingBoxW" not in group:
                continue
            widths = np.asarray(group["boundingBoxW"])
            heights = np.asarray(group["boundingBoxH"])
            offsets = np.asarray(group["offsets"])
            pixels = np.asarray(group["cropPixelData"])
            masks = np.asarray(group["maskPixelData"]) if isolate and "maskPixelData" in group else None

            for i in range(len(offsets)):
                object_id = f"{frame}_{i + 1}"
                if keep is not None and object_id not in keep:
                    continue
                w, h, off = int(widths[i]), int(heights[i]), int(offsets[i])
                n_px = w * h
                if w <= 0 or h <= 0 or off + n_px > pixels.size:
                    continue
                crop = (255 - pixels[off:off + n_px].reshape(h, w)).astype(np.uint8)
                if masks is not None:
                    mask = masks[off:off + n_px].reshape(h, w)
                    crop = np.where(mask > 0, crop, 255).astype(np.uint8)
                yield object_id, crop


def load_restriction(path: Path, labels: set[str]) -> set[str]:
    """Object ids from a previous stage whose top-1 label is in `labels`."""
    keep = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("top1") or "").strip().lower() in labels:
                keep.add(row["object_id"])
    return keep


def classify(h5_path: Path, out_dir: Path, model_path: str, *,
             isolate: bool = False, batch_size: int = 64, top_k: int = 5,
             emit_embeddings: bool = False, keep: set[str] | None = None) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
    id2label = model.config.id2label
    k = min(top_k, len(id2label))
    print(f"  model : {model_path}  ({len(id2label)} classes)")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch: list[torch.Tensor] = []
    n_skipped = 0

    def flush():
        if not batch:
            return
        pixel_values = torch.stack(batch).to(device)
        with torch.no_grad():
            # Call the encoder directly so the embedding is exactly the tensor
            # the classification head consumes, from a single forward pass.
            encoded = model.vit(pixel_values).last_hidden_state[:, 0]
            logits = model.classifier(encoded)
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, k, dim=-1)
        for j, object_id in enumerate(batch_ids):
            row = {"object_id": object_id}
            for rank in range(k):
                row[f"top{rank + 1}"] = id2label[int(top.indices[j, rank])]
                row[f"prob{rank + 1}"] = f"{float(top.values[j, rank]):.6f}"
            rows.append(row)
        if emit_embeddings:
            ids.extend(batch_ids)
            embeddings.append(encoded.float().cpu().numpy())
        batch.clear()
        batch_ids.clear()

    for object_id, crop in iter_crops(h5_path, isolate, keep):
        tensor = preprocess(crop)
        if tensor is None:
            n_skipped += 1
            continue
        batch.append(tensor)
        batch_ids.append(object_id)
        if len(batch) >= batch_size:
            flush()
    flush()

    columns = ["object_id"] + [f"top{i + 1}" for i in range(k)] + [f"prob{i + 1}" for i in range(k)]
    predictions_path = out_dir / "predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
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
    ap.add_argument("--isolate-crops", action="store_true",
                    help="mask neighbours out to white using the segmenter's "
                         "mask. Must match how the model was trained.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--emit-embeddings", action="store_true",
                    help="write the encoder CLS vectors, for EcoTaxa "
                         "similarity search. Use on the stage that sees every "
                         "object, i.e. the binary one.")
    ap.add_argument("--restrict-to", default=None,
                    help="predictions.csv from an earlier stage, or that stage's output directory")
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

    keep = None
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
        keep = load_restriction(restrict_path, labels)
        print(f"Restricted to {len(keep)} objects labelled {sorted(labels)}")
        if not keep:
            print("ERROR: restriction matched no objects", file=sys.stderr)
            return 1

    print(f"Classifying {src}")
    n = classify(src, Path(args.output), args.model,
                 isolate=args.isolate_crops, batch_size=args.batch_size,
                 top_k=args.top_k, emit_embeddings=args.emit_embeddings, keep=keep)
    if n == 0:
        print("ERROR: nothing classified", file=sys.stderr)
        return 1
    print(f"Done: {n} objects -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
