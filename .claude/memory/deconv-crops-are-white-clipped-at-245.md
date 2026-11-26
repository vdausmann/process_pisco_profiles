---
name: deconv-crops-are-white-clipped-at-245
description: "Saved Deconv_crops are hard white-clipped at 245 (>=245 -> 255) — not masked cut-outs; this is what the ViT sees, so benchmarks on raw deconvolution output measure the wrong image"
metadata:
  node_type: memory
  type: project
---

The `Deconv_crops/` the pipeline saves (and packs into `deconv_crops.zip`, and feeds to the
ViT) are **hard white-clipped**: every pixel >= 245 is snapped to exactly 255. Proven
2026-08-07 — taking a raw v5d full-frame deconvolution, cutting the same box and applying
`img[img >= 245] = 255` reproduces the uploaded crop at **MAE 0.002 / 100.0% exact**, versus
MAE 1.50 / 37.5% exact with no clip. 245 is the hardcoded detection threshold (255 - 10).

**It is NOT a mask cut-out.** Object structure, internal texture and even faint neighbouring
particles are fully preserved; there is no contour-based erasure. It only *looks* like objects
pasted on a white canvas because the background is a single uniform 255 with a hard edge where
the object crosses 245. Typical crops are 40-50% pure-255 pixels and have essentially nothing
in the 245-254 band (v5d 0.07%, old 0.68%).

**Applied in BOTH the old and the v5d runs**, so the clip does not by itself explain the
living-rate collapse in [[deconv-model-swap-breaks-vit]]. But it interacts with it: in the
crops the ViT actually consumes, v5d objects are lighter than old ones (min value median
**152 vs 124**), so proportionally more of each object sits in the clipped >=245 zone and is
destroyed before the classifier sees it.

**Why this matters (methodological trap):** any deconvolution benchmark run on RAW model
output — e.g. `underwater-lucyd/eval/copepod_bench.py` and `sim/copepod_compare.png` — is
measuring a different image from the one the classifier and the crops actually carry. Faint
peripheral structure below contrast 10, which the v5 training work spent effort on (Task-2
recall, boundary rendering), is thrown away by the clip and cannot influence classification at
all. The user spotted this by noticing EcoTaxa crops looked unlike `compare.png`.

**Re-run under the clip (2026-08-10).** `eval/copepod_bench.py` now clips by default
(`EVAL_PIPELINE_CLIP=0` disables) and `ctrl_deconv_vs_vit.py` clips too. Results:

- Controlled same-object ViT test got *stronger*: old 23/4947 living (0.46%, was 0.38%
  unclipped), v5d 5/4947 (0.10%, unchanged), **18 flips living->not-living, 0 reverse**.
- Copepoda core contrast / edge_drop on clipped crops, <500um: **old 127.4 / 29.1**,
  v5d 101.1 / 13.2, v4 93.0 / 11.8. old leads on every size bin and its edge_drop is
  **>2x** every new model's.
- Task-2 IoU is UNAFFECTED by the clip (thresholding at 245 yields the same mask whether or
  not >=245 was snapped to 255), so those numbers stand as measured.

**CRITICAL interpretation caveat:** with a ViT trained on OLD-model crops, this benchmark
partly measures *similarity to old*, not image quality. Any deviation from old's look costs
classification by construction, so "old wins" here does NOT establish old is the better
deconvolution — only that it is the one the current classifier expects. Choosing which model
to retrain the classifier *on* cannot be decided by this metric; use human judgement on crop
quality plus clip-independent segmentation metrics (Task-2 IoU, background flatness).

**How to apply:** clip before scoring anything meant to predict downstream/ViT behaviour.
Do not deploy a new deconvolution model with the existing classifier. Consider whether the
245 clip should exist at all — it discards exactly the low-contrast information a better
deconvolution produces, capping the achievable benefit of any improved model.
Related: [[deconv-model-swap-breaks-vit]], [[gpu-deconv-nondeterministic]].
