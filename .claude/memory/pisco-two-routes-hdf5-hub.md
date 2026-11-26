---
name: pisco-two-routes-hdf5-hub
description: "process_pisco_profiles has TWO routes since 2026-09 — the old python pipeline and the now-primary HDF5/Hub route; check which one a task belongs to before editing"
metadata:
  type: project
---

Since 2026-09 (31 commits by Veit, GitLab `main`) `process_pisco_profiles` has **two parallel routes**. Check which one you are working on before editing — the old file paths still exist and still work.

| | old python route | HDF5 / Hub route (primary) |
|---|---|---|
| driver | `process_pisco_profiles.py`, `orchestrate_gpus.py` | Hub tasks in `toolbox.json`, `run*.sh` |
| crops | `Crops/`, `Deconv_crops/` + per-profile TSV | inside the C++ segmenter's **`result.h5`** |
| export | `utils.create_ecotaxa_zips`, `export_ecotaxa_zips.py` | **`export_hdf5_to_ecotaxa.py`** |
| classify | in-pipeline dual ViT | **`vit_classify_hdf5.py`** |

The Hub route closes a full retrain loop: `pisco_vit_classify` → `pisco_ecotaxa_export` → `pisco_ecotaxa_import` → `pisco_ecotaxa_features` → (humans annotate) → `pisco_ecotaxa_labels` → `pisco_vit_trainset` → `pisco_vit_train`.

**Facts that catch people out:**
- Scale bars live in **`scale_bar.py`** (lifted out of `utils.py` because `utils` imports torch/transformers/pandas/matplotlib at module scope, which the processing workers don't have). Wired into `export_hdf5_to_ecotaxa.py`, `--scale-bar` **default on**. See [[scale-bar-pixel-size-22-5]].
- **No model ever sees a barred crop**: classification and training both read `result.h5` via `vit_classify_hdf5.iter_crops`; `train_vit.py` asserts its preprocessing equals `vit_classify_hdf5.preprocess` at startup. EcoTaxa *export* zips are deliberately not a training source.
- `create_ecotaxa_zips` (old route) still defaults to `pixel_resolution=23` and the old non-reversible `add_scale_bar` — 2.2% wrong if ever switched on.
- Everything in `docs/handoff/01..07` describes the **old** route; data findings carry over, file paths may not.

**Pinned old behaviour:** `pisco_segmenter` GitLab branch **`pre-hub-background`** -> `b805b9a` is the exact code every pre-2026-09 result came from (per-frame rolling-window background, before the per-block max-of-pairwise-minima change). Created 2026-11-13. Check it out with `git checkout -b pre-hub-background --track gitlab/pre-hub-background`. The local checkout is still on `main`, so a plain `git pull` there silently jumps the background estimator forward — which changes crops, object counts and abundances. See [[deconv-model-swap-breaks-vit]].

**Where the full writeup is:** `docs/handoff/` in the repo (10 files, index in `README.md`); topic 09 maps both routes. Related: [[deconv-model-swap-breaks-vit]], [[crop-isolation-erase-other-contours]].

ATAIR is (re)processed on the Hub route too (user, 2026-11-18): runs live in /media/veit/T710_data/hub-data/workflows/<uuid>/
with numbered steps 0_prepare_profile -> 1_segment (C++ bg, blocks of 12, max of pairwise minima, 4 MPI ranks split the
profile first) -> 2_deconvolve (pisco_deconvolution_py, v5a, crops packed into texture-filled atlases) -> 3_segment
(object extraction) -> 4/5 ViT -> 6 export -> 7 import. Each step's settings.ini shows the exact parameters.
Example: c5373721-... (NSGR2, 20260422-2051).
