---
name: deconv-model-swap-breaks-vit
description: "Swapping the LUCYD deconvolution model collapses the ViT's living-object rate (-83% to -93%) — the classifier was trained on old-model crops and goes out of domain"
metadata:
  node_type: project
  type: project
---

Measured 2026-08-07. The three ATAIR profiles were re-run through the full pipeline with the
new LUCYD **v5d** deconvolution model (`PISCO_SEGMENTER_MODEL_PATH=.../lucyd-edof-plankton_v5d_20260807.pth`),
output `/media/veit/T710_data/pisco_processed/ATAIR-BSH_v5dlucyd`. Same ViT
(`ViT_custom_size_sensitive_v5`), same pipeline, same frames — only the deconvolution changed:

| profile | old objects / living | v5d objects / living | living% old -> v5d |
|---|---|---|---|
| 20260422-2051 | 5,877 / 426 | 365,074 / 572 | 7.25% -> 0.16% |
| 20260423-1100 | 470,023 / 4,750 | 254,690 / 340 | 1.01% -> 0.13% |
| 20260423-1705 | 804,213 / 21,445 | 456,457 / 3,614 | 2.67% -> 0.79% |

**Why:** the ViT was trained on crops produced by the OLD (231204) deconvolution, whose
signature look is a cracked/granular mesh. v5d renders smooth, darker, speckle-free bodies
(that was the point — see the underwater-lucyd project's `lucyd-v5-sweep-results`), which is a
large appearance shift, so the classifier dumps most real plankton into `not-living`.
Confidence stays high (`prob_1` median 0.99), so this is confident-and-wrong, not low-confidence.

**DO NOT be misled** (cost an inference here): the v5d run's class names read
`Copepoda<Multicrustacea` / `Appendicularia<Tunicata` vs the old run's `Copepoda` /
`Appendicularia`. That is NOT a different classifier — it is a taxon remap added later in
`utils.py:37-39` (and `export_ecotaxa_zips.py:100`). Both runs used ViT v5.

**CONFIRMED by a controlled test** (`process_pisco_profiles/ctrl_deconv_vs_vit.py`, 2026-08-07),
which removes the detection confound: rebuild the cleaned frame, deconvolve the FULL frame with
each model (as the pipeline does), cut crops at the IDENTICAL boxes from the old run's metadata,
run the same binary ViT on both sets. 4,947 objects from 12 frames of 20260423-1100:

    old deconv  19/4947 living (0.38%)
    v5d deconv   5/4947 living (0.10%)
    14 objects flipped living -> not-living, 0 flipped the other way (sign test p~6e-5)
    flipped taxa: Rhizaria 5, Copepoda 3, Appendicularia 3, Noctiluca 2, Thaliacea 1;
    median area 1465 px -> sizeable real organisms, not marginal specks

So the loss is real and ONE-DIRECTIONAL. The full-run drop for this profile was -93% vs -74%
in the control; the remainder is the detection change (object sets also differ, and
20260422-2051 jumped 62x from 5,877 to 365,074 objects — that profile was ALSO the ATAIR
background source for the v5 training set, so check for contamination).

**Verification chain for "did the run really use model X?" (all four confirmed 2026-08-07):**
1. Classification input: `process_pisco_profiles.py:702` sets
   `img_dir = os.path.join(results_folder, "Deconv_crops")` — the ViT classifies DECONVOLVED
   crops, so the deconvolution model directly drives classification.
2. `PISCO_SEGMENTER_MODEL_PATH` (read in `pisco_segmenter/deconvolution.py:_get_model_path`)
   is honoured and propagates through `orchestrate_gpus.py` (`env = os.environ.copy()`).
3. Proof the crops came from v5d: extract a crop from `EcoTaxa/*/deconv_crops.zip`, rebuild the
   cleaned frame with `lib_bgcorr`, deconvolve the FULL frame with each candidate model and cut
   the same box. Result: v5d MAE **1.4-1.8**, v4 3.3-6.5, old 5.7-19.9. The ~1.5 residual is the
   documented GPU non-determinism ([[gpu-deconv-nondeterministic]]), not a mismatch.
4. `--cleanup-intermediate` deletes `Deconv_crops/`, but the zips preserve them, so this check
   still works after cleanup.

**TRAP — two zips per group:** `EcoTaxa/{other,non-living}/` each contain `crops.zip` AND
`deconv_crops.zip`. `crops.zip` holds the CLEANED (bg-corrected) crops by design
(`crop_type == 'crops'` branch, `utils.py:2099`); only `deconv_crops.zip` holds deconvolved
ones. Comparing against `crops.zip` gives MAE 0.00 vs the cleaned frame and looks alarmingly
like "deconvolution never ran" — it isn't. Upload `deconv_crops.zip`.

**Verifying which ViT checkpoints ran (easy to get wrong):** the crops metadata
column `prediction_model` records ONLY the binary model — confirmed identical across both runs
(`ViT_custom_size_sensitive_binary/best_model`, `model.safetensors` mtime 2025-07-01, unchanged
since before either run). The LIVING model path is recorded nowhere (not in metadata, not in the
old logs), so which one the May run used is unknown; it does not affect the living COUNT (that
is the binary model's call) but does make the taxa breakdown non-comparable. v5 and v6 have
IDENTICAL `id2label`, so class names cannot identify the checkpoint.

**How to apply:** a new deconvolution model cannot be deployed on its own — the ViT must be
re-trained or fine-tuned on crops from that model first, or taxon abundance/biovolume becomes
meaningless. Treat deconvolution + classifier as one coupled unit and version them together.
The decisive confirmation test (not yet run): take the same physical crops, deconvolve with old
vs v5d, and run the ViT on both — if the same object flips Copepoda -> not-living, it is purely
domain shift. Related: [[ecotaxa-import-throughput-living-only]], [[custom-folders-mode-regression]].
