# Dual-ViT training

Rebuilds and retrains the two classifiers the pipeline uses: a binary
living / not-living stage, then a taxon classifier over the living crops.

The training set is *derived*, not stored here: labels come from validated
EcoTaxa objects and the images from the run those objects were exported from.
`build_trainset.py` reassembles it, so only the recipe is versioned.

Crops are read either from the segmenter's **`result.h5`**, through the same
function inference uses, or from the old python pipeline's **`deconv_crops`
zips** -- which is how projects validated before the Hub existed can still be
trained from. The EcoTaxa *export* zips are not a source: their PNGs carry a
scale bar no model was trained with.

The whole recipe also runs as three Hub tools; see [Running it on the
Hub](#running-it-on-the-hub) below.

## Why the crops are not committed

118 MB of PNGs that are fully reproducible from (a) the EcoTaxa validations and
(b) the run's exported zips. Committing them would bloat the repo and, worse,
let the images drift from the labels they were built against.

## The coupling that matters

A classifier is only valid for crops that look like the ones it was trained on.
The models in this pipeline are tied to **both**:

* the **deconvolution model** (`PISCO_SEGMENTER_MODEL_PATH`), and
* whether crops were **neighbour-isolated** (`PISCO_ISOLATE_CROPS`).

Swapping either without retraining is an out-of-domain shift: doing so once
dropped the living rate by 83-93% at unchanged confidence. Treat deconvolution,
isolation and classifier as one versioned unit.

`train_vit.py` mirrors `utils.custom_image_processor` exactly - resize the
longest edge to 224, centre-pad to 224x224 with white, normalise to [-1, 1] -
and applies rotation augmentation to the training split only, because inference
is deterministic (its `RandomRotation` is commented out).

That is no longer a claim in a comment: at startup `train_vit.py` runs its own
preprocessing and `vit_classify_hdf5.preprocess` over the same crops and
**aborts if they disagree**. The two cannot drift apart unnoticed.

## What the trainer is

A plain PyTorch loop rather than `transformers.Trainer`. The processing workers
carry torch, transformers, torchvision and scikit-learn; using the Trainer would
mean adding `datasets`, `evaluate` and `accelerate` to every worker for what is
a single-GPU fine-tune, and pinning against a `TrainingArguments` API that has
already changed under us. The recipe is unchanged: AdamW at `--lr` decayed
linearly to zero with no warmup, a stratified 70/15/15 split, and the checkpoint
with the best validation accuracy is the one saved.

`--precision auto` uses fp16 only where the GPU has tensor cores (compute
capability 7.0+). On Pascal cards - the Tesla P4s in the test cluster - fp16 is
*slower* than fp32, so auto correctly stays in fp32 there.

## Reproducing ATAIIR2604_NorthSea

Run: ATAIR-BSH, April 2026 (North Sea), 3 profiles, LUCYD **v5a** deconvolution
with **isolated** crops.

```bash
python training/build_trainset.py \
    --projects 23022 23052 23196 \
    --crops-root /media/veit/T710_data/pisco_processed/ATAIR-BSH_v5alucyd_iso \
    --out ~/ATAIIR2604_NorthSea_trainset \
    --isolate-crops \
    --nonliving Unknowns not-living bubble t001 \
    --exclude-multiclass "multiple species" \
    --min-per-class 20

# binary: fine-tune the previous binary model, whose label set it matches
python training/train_vit.py \
    --data ~/ATAIIR2604_NorthSea_trainset/binary \
    --out  ~/ViT_ATAIIR2604_NorthSea_binary \
    --init /home/veit/PIScO_dev/ViT_custom_size_sensitive_binary/best_model

# multiclass: head is reinitialised (7 new classes vs 13 old)
python training/train_vit.py \
    --data ~/ATAIIR2604_NorthSea_trainset/multiclass \
    --out  ~/ViT_ATAIIR2604_NorthSea_multiclass \
    --init /home/veit/PIScO_dev/ViT_custom_size_sensitive_v5/best_model
```

`--projects` fetches the labels inline and needs EcoTaxa credentials on this
machine; `--labels labels.csv` uses what `ecotaxa_fetch_labels.py` already
fetched, which is how the Hub runs it on a worker that has none. Those crops
came from the old pipeline's zips, so `--isolate-crops` (which applies to the
HDF5 path) changes nothing there - it is shown because a rebuild from
`result.h5` needs it to match `PISCO_ISOLATE_CROPS`.

Projects are listed in priority order. 23052 and 23196 are *subsets* (copies)
of 23022, so the same crop recurs; the last project wins, letting a
re-validation supersede the original call. This produced 12,470 distinct
labelled crops from 12,612 validated rows.

`--lr` defaults to 5e-5, appropriate for fine-tuning. Training from
`google/vit-base-patch16-224-in21k` instead wants roughly 2e-4.

### Class decisions

`bubble` and `t001` sit on the **not-living** side: they are imaging artifacts,
and calling them living reintroduces the false positives the not-living
re-screening removed. `multiple species` is excluded from the multiclass set -
fused objects are a segmentation problem, to be handled by an object detector
rather than by asking the classifier to name a clump. Classes below
`--min-per-class` are dropped because they cannot be split into train/val/test.

### Result

| set | crops | classes | test accuracy |
|---|--:|--:|--:|
| binary | 12,470 | 2 | 98.18% |
| multiclass | 6,599 | 7 | 98.28% |

Per class (multiclass): pluteus 99.0 F1 (n=387), Copepoda 98.8 (377),
Appendicularia 98.5 (163), gelatinous 94.1 (34), Cnidaria 84.8 (19),
Chaetognatha 90.9 (6), Ctenophora 80.0 (4).

Read the tail classes with care: single-digit support means the macro average
(92.3 F1) is not a statement about model quality. Cnidaria's 73.7% recall is
the one real signal - it needs more validated examples.

`pluteus` did not exist in the previous model, which called those objects
*Rhizaria*; that systematic confusion is resolved by construction here.

## Running it on the Hub

Three tools, in the toolbox manifest:

| tool | runs on | does |
|---|---|---|
| `pisco_ecotaxa_labels` | the EcoTaxa machine | fetches validated labels -> `labels.csv` |
| `pisco_vit_trainset` | a processing worker | labels + crops -> `binary/` and `multiclass/` |
| `pisco_vit_train` | a GPU worker | one stage -> `best_model/` + `eval_report.json` |

The split is about **credentials, not connectivity**: EcoTaxa credentials live
on the EcoTaxa machine rather than in task settings, which the Hub renders onto
shared storage in plain text. So the fetch happens there, and everything after
it works from `labels.csv`.

```json
{"name": "retrain the dual ViT", "tasks": [
  {"tool": "pisco_ecotaxa_labels", "settings": {
    "projectIds": "23022,23052,23196", "targetPath": ""}},

  {"tool": "pisco_vit_trainset", "settings": {
    "labelsPath": {"$from": {"task": 0, "output": "labels"}},
    "cropsPath": "/mnt/pisco/.../0_pisco_segment/results/",
    "targetPath": "", "isolateCrops": true,
    "nonLiving": "Unknowns,not-living,bubble,t001",
    "excludeMulticlass": "multiple species", "minPerClass": 20}},

  {"tool": "pisco_vit_train", "settings": {
    "dataPath": {"$from": {"task": 1, "output": "trainset"}},
    "stage": "binary", "targetPath": "", "epochs": 30,
    "initModel": "/opt/pisco/models/ViT_custom_size_sensitive_binary/best_model"}},

  {"tool": "pisco_vit_train", "settings": {
    "dataPath": {"$from": {"task": 1, "output": "trainset"}},
    "stage": "multiclass", "targetPath": "", "epochs": 30,
    "initModel": "/opt/pisco/models/ViT_custom_size_sensitive_v5/best_model"}}
]}
```

Both training tasks read the same trainset, so they can run concurrently on two
GPU workers. Each writes `best_model/`, and `pisco_vit_classify` takes the task
output directory as its `modelPath` - it finds `best_model/` inside - so a
retrained pair chains straight into a classification run:

```json
  {"tool": "pisco_vit_classify", "settings": {
    "sourcePath": "/mnt/pisco/.../results/", "targetPath": "",
    "modelPath": {"$from": {"task": 2, "output": "model"}},
    "isolateCrops": true, "emitEmbeddings": true}}
```

`isolateCrops` must be the same on the trainset task and on every classify task
that uses the result. It is the axis this README is about, and nothing checks it
for you.

## Publishing

Weights live on the Hub, so `--dualvit-model <run>` resolves them anywhere:

```bash
process_pisco_profiles.py --dualvit-model ATAIIR2604_NorthSea
```

Local training output wins when present; otherwise the run name maps to
`<owner>/<run>_{binary,multiclass}` (owner via `PISCO_DUALVIT_HF_NAMESPACE`).
The run prints whether it used local or Hub weights.
