#!/usr/bin/env python3
"""Fine-tune one stage of the dual ViT on a validated crop set.

Takes an imagefolder (``<class>/<crop>.png``) as built by
``build_trainset.py`` and writes ``best_model/`` plus an evaluation report.
Run it twice -- once on ``binary/``, once on ``multiclass/`` -- because the
two stages start from different checkpoints and answer different questions.

Preprocessing matches the deployed pipeline exactly: resize the longest edge
to 224, centre-pad to 224x224 with white, normalise to [-1, 1]. That is
asserted at startup against ``vit_classify_hdf5.preprocess``, the function
inference actually uses, so the two cannot drift apart unnoticed. Rotation
augmentation is applied to TRAIN only; eval and test stay deterministic,
exactly as inference is.

The training loop is plain PyTorch rather than ``transformers.Trainer``: the
processing workers carry torch, transformers, torchvision and scikit-learn,
and this avoids pulling in the ``datasets``/``evaluate``/``accelerate`` stack
for what is a single-GPU fine-tune. The recipe is unchanged -- AdamW at
``--lr`` with linear decay to zero and no warmup, a stratified 70/15/15
split, and the checkpoint with the best validation accuracy is what gets
saved.

Usage:
    train_vit.py --data trainset/binary --out ViT_run_binary --init <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vit_classify_hdf5 import PADDING_COLOUR, TARGET_SIZE, preprocess  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def fit_and_pad(image: Image.Image) -> Image.Image:
    """Longest edge to 224, then centre-pad to 224x224 with white."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = TARGET_SIZE / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    if new_w < 1 or new_h < 1:
        return None
    resized = TF.resize(image, (new_h, new_w))
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_top = (TARGET_SIZE - new_h) // 2
    return TF.pad(resized, [pad_left, pad_top,
                            TARGET_SIZE - new_w - pad_left, TARGET_SIZE - new_h - pad_top],
                  fill=PADDING_COLOUR)


def to_pixel_values(image: Image.Image, rotate) -> torch.Tensor | None:
    """One image -> the normalised 3x224x224 tensor the model consumes.

    ``rotate`` (a RandomRotation, or None) is applied after padding, so the
    object turns inside the 224 canvas against the same white ground -- which
    is what the augmentation did when these models were first trained.
    """
    padded = fit_and_pad(image)
    if padded is None:
        return None
    if rotate is not None:
        padded = rotate(padded)
    tensor = TF.to_tensor(padded)
    return TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def assert_matches_inference() -> None:
    """Deterministic preprocessing here must equal the inference path.

    A classifier is only valid for crops that look like its training data;
    this is the cheapest possible guard on the one axis we control in code.
    """
    rng = np.random.default_rng(0)
    for shape in [(37, 91), (224, 224), (300, 12)]:
        crop = rng.integers(0, 255, shape, dtype=np.uint8)
        mine = to_pixel_values(Image.fromarray(crop), None)
        theirs = preprocess(crop)
        if mine is None or theirs is None or not torch.allclose(mine, theirs):
            raise SystemExit("ERROR: training preprocessing no longer matches "
                             "vit_classify_hdf5.preprocess. Fix that before training.")


class Crops(Dataset):
    def __init__(self, samples, augment: bool):
        self.samples = samples
        self.rotate = transforms.RandomRotation(degrees=180, fill=255) if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as image:
            tensor = to_pixel_values(image, self.rotate)
        if tensor is None:                     # unresizable crop: a white square
            tensor = torch.zeros(3, TARGET_SIZE, TARGET_SIZE) + 1.0
        return tensor, label


def load_imagefolder(root: Path):
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise SystemExit(f"ERROR: {root} holds {len(classes)} class directories; "
                         f"at least 2 are needed")
    samples = []
    for index, name in enumerate(classes):
        for path in sorted((root / name).rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, index))
    return classes, samples


def stratified_split(samples, classes, seed: int):
    """70/15/15, stratified, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    by_class = {i: [] for i in range(len(classes))}
    for index, (_path, label) in enumerate(samples):
        by_class[label].append(index)
    thin = {classes[c]: len(v) for c, v in by_class.items() if len(v) < 3}
    if thin:
        raise SystemExit(f"ERROR: these classes have fewer than 3 crops and cannot "
                         f"be split into train/validation/test: {thin}. Raise "
                         f"--min-per-class when building the set, or drop them.")
    train, validation, test = [], [], []
    for indices in by_class.values():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, round(n * 0.70))
        n_validation = max(1, round((n - n_train) / 2))
        n_train = min(n_train, n - 2)          # leave one each for val and test
        train += indices[:n_train]
        validation += indices[n_train:n_train + n_validation]
        test += indices[n_train + n_validation:]
    for part in (train, validation, test):
        rng.shuffle(part)
    return ([samples[i] for i in train], [samples[i] for i in validation],
            [samples[i] for i in test])


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    y_true, y_pred = [], []
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits = model(pixel_values=pixel_values).logits
        y_pred += logits.float().argmax(-1).cpu().tolist()
        y_true += labels.tolist()
    return np.array(y_true), np.array(y_pred)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="imagefolder directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--init", required=True,
                    help="checkpoint to fine-tune: a directory, or a Hugging Face "
                         "repo id. The head is reinitialised when the class set differs.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="5e-5 suits fine-tuning; training from "
                         "google/vit-base-patch16-224-in21k wants roughly 2e-4")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto uses fp16 only where the GPU has tensor cores "
                         "(compute capability 7.0+); on Pascal cards fp16 is slower "
                         "than fp32, not faster")
    a = ap.parse_args()

    assert_matches_inference()
    print("preprocessing matches vit_classify_hdf5.preprocess", flush=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    data = Path(a.data)
    if not data.is_dir():
        print(f"ERROR: {data} is not a directory", file=sys.stderr)
        return 1
    classes, samples = load_imagefolder(data)
    if not samples:
        print(f"ERROR: no images under {data}", file=sys.stderr)
        return 1
    counts = {c: sum(1 for _p, l in samples if l == i) for i, c in enumerate(classes)}
    run_name = data.name          # "binary" / "multiclass": the stage, not the task dir
    print(f"[{run_name}] {len(samples)} images · "
          f"{len(classes)} classes: {classes}", flush=True)
    print(f"  per class: {counts}", flush=True)

    train_s, validation_s, test_s = stratified_split(samples, classes, a.seed)
    print({"train": len(train_s), "validation": len(validation_s), "test": len(test_s)},
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capability = torch.cuda.get_device_capability() if device.type == "cuda" else (0, 0)
    if a.precision == "fp16" or (a.precision == "auto" and capability >= (7, 0)):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None
    if device.type == "cuda":
        print(f"  device: {torch.cuda.get_device_name()} (sm_{capability[0]}{capability[1]}), "
              f"{'fp16 autocast' if autocast_dtype else 'fp32'}", flush=True)
    else:
        print("  device: cpu -- this will be slow", flush=True)

    loaders = {
        "train": DataLoader(Crops(train_s, True), batch_size=a.bs, shuffle=True,
                            num_workers=a.workers, pin_memory=device.type == "cuda",
                            drop_last=False),
        "validation": DataLoader(Crops(validation_s, False), batch_size=a.bs,
                                 num_workers=a.workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(Crops(test_s, False), batch_size=a.bs,
                           num_workers=a.workers, pin_memory=device.type == "cuda"),
    }

    model = ViTForImageClassification.from_pretrained(
        a.init, num_labels=len(classes),
        id2label={str(i): c for i, c in enumerate(classes)},
        label2id={c: str(i) for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps = max(1, len(loaders["train"]) * a.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (steps - step) / steps))
    scaler = torch.amp.GradScaler(device.type, enabled=autocast_dtype is torch.float16)

    out = Path(a.out)
    best_dir = out / "best_model"
    out.mkdir(parents=True, exist_ok=True)
    best_accuracy, best_epoch = -1.0, -1
    history = []
    started = time.time()

    for epoch in range(1, a.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        epoch_started = time.time()
        for step, (pixel_values, labels) in enumerate(loaders["train"], start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / seen:.4f}", flush=True)

        y_true, y_pred = evaluate(model, loaders["validation"], device, autocast_dtype)
        accuracy = float((y_true == y_pred).mean())
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "validation_accuracy": accuracy,
                        "seconds": round(time.time() - epoch_started, 1)})
        marker = ""
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            model.save_pretrained(best_dir)
            marker = "  <- best, saved"
        print(f"epoch {epoch}/{a.epochs}  loss {running / max(seen, 1):.4f}  "
              f"val_acc {accuracy:.4f}  ({time.time() - epoch_started:.0f}s){marker}",
              flush=True)

    if best_epoch < 0:
        print("ERROR: no epoch completed", file=sys.stderr)
        return 1
    print(f"\nBest epoch {best_epoch}, validation accuracy {best_accuracy:.4f}", flush=True)

    # Test with the saved best checkpoint, not with whatever the last epoch left
    # in memory -- the same thing load_best_model_at_end did.
    model = ViTForImageClassification.from_pretrained(best_dir).to(device)
    y_true, y_pred = evaluate(model, loaders["test"], device, autocast_dtype)
    test_accuracy = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    (out / "eval_report.json").write_text(json.dumps({
        "labels": classes, "report": report,
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(len(classes)))).tolist(),
        "test_metrics": {"accuracy": test_accuracy},
        "validation_accuracy": best_accuracy, "best_epoch": best_epoch,
        "history": history, "init": a.init, "n": len(samples),
        "class_counts": counts, "data": str(data),
        "epochs": a.epochs, "lr": a.lr, "batch_size": a.bs, "seed": a.seed,
        "minutes": round((time.time() - started) / 60, 1),
    }, indent=1))
    (out / "labels.json").write_text(json.dumps(classes, indent=1))

    print(classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0), flush=True)
    print(f"test accuracy {test_accuracy:.4f}", flush=True)
    print(f"SAVED {best_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
