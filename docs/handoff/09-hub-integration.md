# 09 — PIScO Hub integration, and what it changes for us

**State: landed on both repos' GitLab `main`.** `process_pisco_profiles` is
now **reconciled** (this session's work rebased on top, 2026-11-13).
`pisco_segmenter` local is still **9 commits behind** and was reviewed
read-only — nothing pulled or merged.

> **The old behaviour is pinned.** `pisco_segmenter` GitLab branch
> **`pre-hub-background`** -> `b805b9a` is the exact code every result in
> topics 01-07 was produced with: per-frame rolling-window background, before
> the per-block change below. Created 2026-11-13. It is a pointer at a commit
> already on `main`, so it adds nothing and can be deleted freely.
> Check it out to reproduce or continue any of this session's runs:
> `git checkout -b pre-hub-background --track gitlab/pre-hub-background`

Read this first: it reframes where the other topics' code lives. Everything in
topics 01-07 was done on the **old python route**; the Hub work adds a
**second, now-primary route** through the C++ segmenter's `result.h5`.

## The two routes

| | old python route | HDF5 / Hub route |
|---|---|---|
| driver | `process_pisco_profiles.py`, `orchestrate_gpus.py` | Hub tasks in `toolbox.json`, `run*.sh` |
| crops | `Crops/`, `Deconv_crops/` + per-profile TSV | inside **`result.h5`** |
| export | `utils.create_ecotaxa_zips`, `export_ecotaxa_zips.py` | **`export_hdf5_to_ecotaxa.py`** |
| classify | in-pipeline dual ViT | **`vit_classify_hdf5.py`** |
| scale bar | not wired, old non-reversible fn | **on by default**, reversible (topic 06) |

Both still exist; the old route is what produced every number in this session.

## What landed on `process_pisco_profiles` (31 commits, 2026-09-09 .. 10-21)

A **closed retraining loop**, each step a Hub task with a `run_*.sh` wrapper:

```
segmenter result.h5
  → pisco_vit_classify     (vit_classify_hdf5.py; fp16 on tensor-core GPUs,
                            loader workers; chainable so stage 2 only sees
                            stage 1's living calls)
  → pisco_ecotaxa_export   (export_hdf5_to_ecotaxa.py; PNGs straight into the
                            zip from a process pool, predictions carried in
                            the TSV, scale bar, object_esd, sample_ctd_cast,
                            object_frame_index, link back to raw data)
  → pisco_ecotaxa_import   (ecotaxa_import.py; creates or adds to a project,
                            single pass since predictions ride along)
  → pisco_ecotaxa_features (ecotaxa_write_features.py; ViT CLS embeddings into
                            EcoTaxa so "similar to" works — the tool used
                            heavily during the not-living re-screening, topic 03)
  →  … humans annotate …
  → pisco_ecotaxa_labels   (ecotaxa_fetch_labels.py)
  → pisco_vit_trainset     (training/build_trainset.py)
  → pisco_vit_train        (training/train_vit.py, run once per stage)
  → back to pisco_vit_classify
```

Supporting pieces: `scale_bar.py` (topic 06), `crops.py` (`tight_masked`),
`ecotaxa_taxonomy_map.{py,json}` translating ViT labels to EcoTaxa taxon ids
— including the ATAIIR2604 multiclass labels — `ecotaxa_mirror.py` to copy
projects between EcoTaxa instances, and `ecotaxa_grant_annotators.py`.

Two details worth carrying:
* **Objects land as Biota**, with numeric EcoTaxa taxon ids written from the
  taxonomy map. The deprecated-taxon remap of topic 03 is the same problem this
  map now solves properly.
* `.gitignore` now excludes the **local** credentials file; only
  `process_pisco_profiles.config.example.json` is versioned (see topic 08).

## What landed on `pisco_segmenter` (9 commits, not pulled)
`88c8015` headless GPU worker · `0b8b3be` describe pipeline as a Hub tool ·
`87cd167`+`87df281` write `result.h5` alongside PNG/CSV · `acab6ed` make the
circular FOV mask switchable · `9f5fe67` **estimate the background as the C++
segmenter does** · `d15dcea` merge · `4af9b4d` **correct the background per
block** · `8eff671` read frames from a profile dataset as well as a directory.

**All four of this session's segmenter commits survive intact** on that branch
(`99a0d88`, `c1e03cd`, `8480520` crop isolation, `b805b9a` hang fix) — verified
by ancestry and by grepping the merged files.

## Does it change our workflow? **Yes — materially.**

### Background estimation changed, and there is no switch
| | before (what every result in this session used) | after |
|---|---|---|
| scope | per-frame **rolling window**, frame excluded | **per block of 12**, one background shared by the block |
| estimator | `np.max(bg_imgs)` | **max over pairwise minima** |

`PISCO_BG_SIZE` overrides the block size (default **12**); the *estimator*
itself has no legacy path.

**Why it was done** is sound and worth knowing: 12 is the stack the LUCYD
background textures were built with, so this keeps the **deconvolution input in
distribution**. That is a real improvement to the deconvolution step.

**But it changes the cleaned frame**, therefore the deconvolved crop,
therefore the thresholded contours. So:
1. **Object counts and crop appearance will differ** from everything recorded
   in this session. Abundances are not comparable across the change.
2. It is a **fourth domain shift** stacked on deconvolution model, crop
   isolation and classifier. Both published ViTs (`Veit/M181_*`,
   `Veit/ATAIIR2604_NorthSea_*`) were trained on **old-background** crops.
3. The gain is real but **unmeasured for classification** — better LUCYD input
   does not automatically mean better crops for a classifier trained on the old
   look. Treat it the way the deconvolution comparison was treated: measure,
   do not assume.

### Smaller changes
* **Input**: `frames.py` reads a directory **or** a `profile_dataset.h5`.
  Directory input still works, so existing invocations are unaffected.
* **Output**: `result.h5` is written **alongside** the PNGs and CSVs, not
  instead of them. Note its mask is **the object's own contour**, unlike the
  PNG crops, which are cut with the **accumulated frame mask** — independent
  confirmation of the accumulation behaviour found during the isolation work
  (topic 02). The HDF5 route's `--isolate` reproduces isolation from that mask.
* **FOV mask** is now overridable (`mask_img_override`); the default is
  unchanged (on).
* New knobs: `PISCO_BG_SIZE` (12), `PISCO_DECONV_BATCH`, `PISCO_LOG_FILE`.
* `run_profile.py` + `toolbox.json` package the segmenter as a Hub tool — a
  second entry point beside `orchestrate_gpus.py`.

## Decisions this forces
1. **Pull the segmenter or stay pinned?** Pulling silently changes the output
   of every future run. If SO308 is to stay comparable with the Madagascar
   subset already processed, either finish SO308 on the current code or re-run
   the two completed profiles after pulling.
   The `pre-hub-background` branch above makes "stay pinned" an explicit,
   reproducible choice rather than an accident of not having pulled — but note
   the local checkout is still on **`main`**, so a casual `git pull` there
   would still jump it forward mid-campaign. Switch the checkout to the
   pinned branch (a no-op for the files, same commit) if that risk matters.
2. **If standardising on the block background**, both ViTs want retraining on
   matching crops — and that is also the moment to settle the deconvolution
   choice (topic 02) and switch crop isolation on, so one retrain absorbs all
   three shifts instead of three retrains. The Hub loop above now makes that a
   chain of tasks rather than hand-run scripts.
3. **GitHub mirror of the segmenter is behind** and has not received these
   commits (topic 08).

## New-chat prompt
> Read docs/handoff/09-hub-integration.md. The segmenter's background
> estimation changed to per-block max-of-pairwise-minima and the HDF5/Hub route
> is now primary. I want to <measure the background change's effect on
> crops/classification before pulling / pull and retrain the ViTs through the
> Hub task chain / keep SO308 pinned to the current code>.
