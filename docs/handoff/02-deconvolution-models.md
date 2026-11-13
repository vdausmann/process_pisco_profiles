# 02 — LUCYD deconvolution models + crop isolation

**State: evaluated, decision pending.** Four models compared on identical
frames; crop isolation implemented. No model is "approved" yet because the
comparison is confounded by the classifier.

## Model comparison (3 ATAIR profiles, identical pipeline, only deconv changed)

| Model | Total objects | Living | Rate |
|---|--:|--:|--:|
| v4 (`..._v4_20260722`) | 1,077,477 | 3,126 | 0.29% |
| v5d (`..._v5d_20260807`) | 1,076,221 | 4,526 | 0.42% |
| v5 (`..._v5_20260806`) | 799,922 | 6,812 | 0.85% |
| **v5a (`..._v5a_20260807`)** | **645,090** | **7,364** | **1.14%** |

Two trends as the model improves: **total detections fall** (fewer spurious
particles) while **living rises**, monotonically on every profile.

## The confound — read before choosing a model
The ViT was trained on crops from the **old 231204** deconvolution. A
controlled same-object test (`ctrl_deconv_vs_vit.py`) showed the newer models
push real plankton into not-living: 18 objects flipped living -> not-living,
0 the other way. So **a higher living rate here partly measures similarity to
the old model, not image quality.** Choosing a deconvolution on this metric is
circular; use human judgement on crop quality plus clip-independent
segmentation metrics instead.

Related: deconv crops are **hard white-clipped at 245**, so any benchmark run
on raw model output measures a different image from what the classifier sees.

## Crop isolation (`PISCO_ISOLATE_CROPS=1`, commit `8480520`)
Whites out every contour except the target, so a crop carries only its own
object. Measured: multi-object crops **15.0% -> 0.5%**, **crop count
unchanged** (detection untouched). Off by default.

Context: of crops with neighbours, roughly half are separately-cropped objects
(classified in their own vignette — do not re-annotate, it double-counts) and
half are sub-50px specks with no crop of their own.

## A fourth variable arrived after this comparison
The segmenter's **background estimation** changed in Sep 2026 (per-block
max-of-pairwise-minima, block 12) explicitly to keep the deconvolution input in
distribution with LUCYD's training stack. That invalidates nothing above, but
any *re-run* of this comparison must be on one side of that change or the
other, not a mix. See [09](09-hub-integration.md).

## Where things live
* Models: `underwater-lucyd/models/` — 18 checkpoints now **tracked in git**
  (commit `7091a78`), on GitLab and GitHub.
* Select with `PISCO_SEGMENTER_MODEL_PATH=<path to .pth>`.

## Open
* **Which deconvolution to standardise on** — needs the retrain-and-compare
  described in topic 04, or a clip-independent metric.
* Isolation is opt-in and has never been paired with a production run other
  than the ATAIR v5a one.

## New-chat prompt
> Read docs/handoff/02-deconvolution-models.md. I want to decide which LUCYD
> deconvolution model to standardise on, given the classifier confound.
