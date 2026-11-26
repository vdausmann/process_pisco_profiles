---
name: gpu-deconv-nondeterministic
description: "GPU deconvolution is non-deterministic, so re-cropping can't reproduce an existing TSV's crops — re-run the full pipeline instead"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-05T14:58:36.675Z
---

The PISCO deconvolution step on the AMD GPUs (ROCm) is **not bit-reproducible** run-to-run. Re-segmenting a profile produces slightly different deconvolved images → slightly different contours → a different set/count of crops than the original run.

**Consequence:** `regenerate_deconv_crops.py` (the recrop tool that re-makes `Deconv_crops/` from a preserved `settings.csv` + TSV, for profiles cleaned by [[pisco-intermediate-crops-exhaust-inodes]]) only works when the regenerated crops happen to match the TSV's `object_full_path`/`img_file_name` references. On 2026-08-05 it hit **89.7% and 99.3% coverage** for freshly-reprocessed M181 CTD-036/CTD-057 (thousands of TSV-referenced crops missing) and correctly refused to export (coverage guard). The 15 other profiles recropped at 100% by luck.

**How to apply:** to regenerate uploadable zips for a profile whose crops were deleted, prefer re-running the **full pipeline** with `--export-zip` (segmentation + detection + ViT + TSV + zip all from one segmentation pass, so crops and TSV match by construction) rather than the recrop tool. There's no `--force` flag: delete the profile's `<profile>_Results/` first (also frees inodes) so the skip logic doesn't short-circuit, then run. On the dual 7900XTX use `orchestrate_gpus.py --gpus 0,1 ... -- --export-zip --cleanup-intermediate` ([[pisco-run-with-pisco-conda-env]]).

Only relevant when you must reproduce an *existing* TSV's exact crops. A fresh full run is always internally consistent.
