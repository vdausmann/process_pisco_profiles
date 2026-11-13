# 04 — Dual-ViT retraining and model publishing

**State: trained, published, wired into the pipeline. Evaluation unfinished.**
Updated 2026-11-13: the `training/` recipe was **rewritten** upstream and now
runs as Hub tasks — see *Where things live*.

## Run `ATAIIR2604_NorthSea`
Retrained both stages on the ATAIR validations, because the old ViT was out of
domain on v5a + isolated crops.

| Set | crops | classes | test accuracy |
|---|--:|--:|--:|
| binary | 12,470 | 2 | **98.18%** |
| multiclass | 6,599 | 7 | **98.28%** |

Per class (multiclass F1): pluteus 99.0 (n=387), Copepoda 98.8 (377),
Appendicularia 98.5 (163), gelatinous 94.1 (34), Cnidaria 84.8 (19),
Chaetognatha 90.9 (6), Ctenophora 80.0 (4).

**Read the tail with care** — single-digit support means the macro average
(92.3 F1) says little. Cnidaria's 73.7% recall is the one real signal; it needs
more validated examples.

## Decisions baked in
* Binary not-living side = `Unknowns + not-living + bubble + t001` (artifacts
  belong with non-organisms).
* `multiple species` **excluded** from the multiclass set — fused objects are a
  segmentation problem for an object detector, not a naming problem.
* Classes under 20 crops dropped (cannot split train/val/test).
* Fine-tuned from the previous ViTs (lr 5e-5), not from ImageNet.
* Dedup matters: EcoTaxa subsets are **copies**, so 12,612 validated rows
  collapse to 12,470 distinct crops; last project wins.

## Deployment result on ATAIR (v5a + isolation + new ViT)
645,090 objects, **36,178 living (5.61%)** vs 7,255 (1.12%) with the old ViT —
about 5x. Totals identical, so this is purely the classifier.

**This number is not yet trusted.** The 98.2% test accuracy comes from a split
of the *validated pool*, which is not a random sample of all objects. Project
**23210** (500 random living calls + 500 random not-living) was built to measure
real precision and recall — **still unvalidated**. `gelatinous` is the most
likely over-call (3,792 predicted from 229 training examples).

## Where things live
* Recipe: `training/build_trainset.py`, `training/train_vit.py`,
  `training/README.md`. Landed as `cc30f6e`, then **substantially rewritten**
  (`a0c5b8d`, `f9e8482`) — if you remember the original scripts, re-read them:
  * Crops now come from the segmenter's **`result.h5` via
    `vit_classify_hdf5.iter_crops`** — literally the inference function, so
    training crops cannot drift from what inference produces. Old projects
    still train from `deconv_crops*.zip`. EcoTaxa **export** zips are
    deliberately *not* a source: their PNGs carry a scale bar (topic 06).
  * `train_vit.py` **asserts at startup** that its preprocessing equals
    `vit_classify_hdf5.preprocess`, and exits if it has drifted.
  * `--isolate-crops` reproduces neighbour isolation from the HDF5 mask.
  * Runs as three Hub tasks: `pisco_ecotaxa_labels` -> `pisco_vit_trainset`
    -> `pisco_vit_train` (topic 09).
  * Crops still deliberately not committed — they are derived (118 MB).
* Weights on Hugging Face, **all public**:
  `Veit/ATAIIR2604_NorthSea_{binary,multiclass}` (v5a + isolated) and
  `Veit/M181_{binary,multiclass}` (231204 + non-isolated, the pipeline default).
* Select with `--dualvit-model ATAIIR2604_NorthSea`; local dirs win, otherwise
  it resolves `<owner>/<run>_{binary,multiclass}` from the Hub
  (`PISCO_DUALVIT_HF_NAMESPACE`). Prints `(local)` or `(Hugging Face)`.
  Still present in `process_pisco_profiles.py` after the rebase.
  The HDF5 route selects a model per stage instead, via `pisco_vit_classify`.

## Open
* **Validate 23210** and compute precision/recall/F1 with CIs. If precision is
  high the 5x is real recovery; if not, the model over-calls.
* Recall needs the not-living half; at 500 draws it yields a bound, not a tight
  estimate.

## New-chat prompt
> Read docs/handoff/04-dualvit-training.md. I validated project 23210. Pull the
> results and compute precision, recall and F1 per class with CIs.
