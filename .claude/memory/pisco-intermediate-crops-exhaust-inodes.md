---
name: pisco-intermediate-crops-exhaust-inodes
description: PISCO processing writes millions of per-particle files; use --cleanup-intermediate or the disk runs out of inodes
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-07-31T07:30:10.122Z
---

Each processed PISCO profile writes one file per detected particle into three dirs — `Crops/`, `Deconv_crops/`, `Masks/` — under `<profile>_Results/`. Counts are huge: e.g. M181 CTD-057 had ~12.8M particles → ~38M files for that one profile; M181's ~15 profiles totalled ~119M files.

On 2026-07-30 a 15-profile M181 run to `/media/veit/T710_data` (ext4, 244M inodes) **exhausted inodes at 100%** (only ~1.2TB of 3.6TB bytes used) and stalled mid-run with `OSError: [Errno 28] No space left on device`. Byte-space was fine — it was inodes.

**How to apply:** pass `--cleanup-intermediate` to `process_pisco_profiles.py` (added 2026-07-31, commit 02657c0). After each profile's EcoTaxa TSV is written it deletes `Crops/Deconv_crops/Masks/Images`, keeping only `Data/` + `EcoTaxa/`. The orchestrator ([[pisco-run-with-pisco-conda-env]] context) forwards it via passthrough: `orchestrate_gpus.py ... -- --cleanup-intermediate`. Only `Data` + the EcoTaxa TSV are needed for abundance/biovolume plots and downstream; crops are only needed if uploading vignettes to EcoTaxa.

Recovery when it happens: kill the run; `rm -rf` the whole dir for any partially-segmented profile (partial `Data/*.csv` makes the pipeline wrongly skip segmentation on resume, process_pisco_profiles.py:~1260); strip Crops/Deconv_crops/Masks from completed profiles; then resume with `--cleanup-intermediate`.
