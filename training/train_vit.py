"""Fine-tune a ViT on the ATAIIR2604_NorthSea validated crop set.
Preprocessing matches the deployed pipeline exactly (utils.custom_image_processor):
resize longest edge to 224, centre-pad to 224x224 with white, normalise to [-1,1].
Rotation augmentation is applied to TRAIN only; eval/test stay deterministic,
exactly as inference does (RandomRotation is commented out there)."""
import os, sys, json, argparse, numpy as np, torch, logging
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

from datasets import load_dataset, DatasetDict
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

def resize_to_larger_edge(image, target_size=224):
    w, h = image.size
    s = target_size / max(w, h)
    try: return TF.resize(image, (int(h * s), int(w * s)))
    except ValueError: return None

def make_proc(train):
    chain = ([transforms.RandomRotation(degrees=180, fill=255)] if train else []) + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    tc = transforms.Compose(chain)
    def proc(image):
        if image.mode != "RGB": image = image.convert("RGB")
        r = resize_to_larger_edge(image, 224)
        if r is None: return None
        nw, nh = r.size
        pl = (224 - nw)//2; pt = (224 - nh)//2
        r = transforms.Pad((pl, pt, 224-nw-pl, 224-nh-pt), fill=255)(r)
        return tc(r)
    return proc

def batcher(train):
    proc = make_proc(train)
    def f(b):
        px, lb = [], []
        for img, l in zip(b["image"], b["label"]):
            t = proc(img)
            if t is not None: px.append(t); lb.append(l)
        return {"pixel_values": torch.stack(px), "label": lb}
    return f

full = load_dataset("imagefolder", data_dir=a.data)["train"]
labels = full.features["label"].names
print(f"[{os.path.basename(a.out)}] {len(full)} images · {len(labels)} classes: {labels}", flush=True)
s1 = full.train_test_split(test_size=0.30, stratify_by_column="label", seed=42)
s2 = s1["test"].train_test_split(test_size=0.50, stratify_by_column="label", seed=42)
ds = DatasetDict(train=s1["train"], validation=s2["train"], test=s2["test"])
print({k: len(v) for k, v in ds.items()}, flush=True)
prepared = DatasetDict(train=ds["train"].with_transform(batcher(True)),
                       validation=ds["validation"].with_transform(batcher(False)),
                       test=ds["test"].with_transform(batcher(False)))

model = ViTForImageClassification.from_pretrained(
    a.init, num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True)

def collate(b):
    return {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
            "labels": torch.tensor([x["label"] for x in b])}
acc = evaluate.load("accuracy")
def metrics(p): return acc.compute(predictions=np.argmax(p.predictions, 1), references=p.label_ids)

args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs,
    per_device_eval_batch_size=a.bs, eval_strategy="epoch", save_strategy="epoch",
    fp16=True, num_train_epochs=a.epochs, logging_steps=200, learning_rate=a.lr,
    save_total_limit=1, remove_unused_columns=False, push_to_hub=False,
    report_to="tensorboard", load_best_model_at_end=True,
    metric_for_best_model="accuracy", greater_is_better=True)
tr = Trainer(model=model, args=args, data_collator=collate, compute_metrics=metrics,
             train_dataset=prepared["train"], eval_dataset=prepared["validation"])
res = tr.train()
best = os.path.join(a.out, "best_model"); tr.save_model(best)
tr.log_metrics("train", res.metrics); tr.save_metrics("train", res.metrics); tr.save_state()
m = tr.evaluate(prepared["test"]); tr.log_metrics("test", m); tr.save_metrics("test", m)
out = tr.predict(prepared["test"])
y, yp = out.label_ids, out.predictions.argmax(1)
rep = classification_report(y, yp, target_names=labels, output_dict=True, zero_division=0)
json.dump({"labels": labels, "report": rep,
           "confusion": confusion_matrix(y, yp).tolist(),
           "test_metrics": m, "init": a.init, "n": len(full)},
          open(os.path.join(a.out, "eval_report.json"), "w"), indent=1)
print(classification_report(y, yp, target_names=labels, zero_division=0), flush=True)
print("SAVED", best, flush=True)
