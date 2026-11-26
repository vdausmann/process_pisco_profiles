---
name: pisco-run-with-pisco-conda-env
description: Always run process_pisco_profiles.py with the pisco conda env (ROCm torch) — the base env falls back to CPU
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-07-29T14:41:43.638Z
---

Run `process_pisco_profiles.py` with `/home/veit/miniconda3/envs/pisco/bin/python`, NOT the base `/home/veit/miniconda3/bin/python`.

**Why:** This machine has AMD Radeon RX 7900 XTX GPUs (×2). The base env has `torch 2.7.1+cu126` (a CUDA/NVIDIA build) → `torch.cuda.is_available()` is False → the ViT step falls back to CPU (utils.py:600) and is very slow. The `pisco` env has `torch 2.6.0+rocm6.4.1` with `cuda.is_available: True` and sees both GPUs; the code's `"cuda"` device string maps to ROCm/HIP, so no code change is needed.

**How to apply:** Launch with the pisco-env python. Confirm the run's log shows `Using device: cuda` (not `cpu`); `rocm-smi --showuse` should show GPU[0] active during ViT. On 2026-07-29 a run was launched with the base env by mistake, wasting ~17h of CPU ViT before being caught.

Resume is safe: segmentation skips when `Data/*.csv` exist (process_pisco_profiles.py:1260) and ViT reloads existing predictions (:666), so killing and relaunching does not redo completed segmentation.
