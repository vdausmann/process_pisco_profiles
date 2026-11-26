---
name: crop-isolation-erase-other-contours
description: "PISCO_ISOLATE_CROPS=1 whites out neighbouring contours in each crop (15%->0.5% multi-object); crop count unchanged, but it's an appearance change so pair with a ViT retrain"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-10T15:12:34.454Z
---

Added 2026-08-10 (pisco_segmenter commit **8480520**). Env var **`PISCO_ISOLATE_CROPS=1`** turns on opt-in crop isolation in `detection.py`: for each saved crop, every OTHER detected contour's pixels are set to 255, keeping the target object plus its own halo/background. Neighbours are erased with a 1px dilation (to catch the neighbour's faint edge); the target contour is protected from that dilation so it's never eaten into. Off by default — existing behaviour preserved. Wired via `DetectionSettings.isolate_crops`, read from the env var in `segmenter.py`; `orchestrate_gpus.py` propagates the env (`os.environ.copy()`).

**It fixes a real but pre-existing bug intent:** [detection.py] built a `crop_masked` meant to keep only the object, but the frame `mask` accumulated contours across the loop so neighbours leaked in — which is why crops carried secondary particles ([[deconv-crops-are-white-clipped-at-245]] confirmed crops are white-clipped, NOT masked cut-outs). Isolation uses an all-contours mask minus a fresh per-object mask, sliced with the SAME `[y:y+h, x:x+w]` as the crop (resize=False for these runs, so native coords line up).

**Key facts (verified on the 3 ATAIR profiles, v5a):**
- **Crop count is identical** with/without isolation (detection is unchanged — only pixel content differs): 200534 / 193039 / 251517 both ways.
- Multi-object crops (a 2nd blob >=5% of the primary) dropped **15.0% -> 0.5%**.
- Living count moved only slightly (7364 -> 7255), because the current ViT still expects the old look.

**How to apply / caveat:** isolation changes crop APPEARANCE, so it's a domain shift for a classifier trained on non-isolated crops — same coupling as [[deconv-model-swap-breaks-vit]]. Deploy it **together with a ViT retrain on isolated crops**, not under the current classifier expecting a free accuracy win. Only ~0.7-15% of crops are affected (depending on threshold), so it's a retrain-cycle cleanup, not an urgent fix. Bbox-overlap quantification method (which crops have neighbours, and whether the neighbour has its own crop) is in `<scratchpad>/bbox_overlap.py`; ~half of real neighbours are separately-cropped objects (Case 1, don't re-annotate — double-counts), ~half are sub-50px specks with no own crop. Related: [[ecotaxa-import-throughput-living-only]].
