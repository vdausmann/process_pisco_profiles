# 06 — Scale bars on annotator crops

**State: OPEN — library landed, not wired into the pipeline.**

## Why it is built the way it is
Crops shown to annotators need a size reference, but the same crops are
**classifier input**. A bar burnt into the image is an appearance change the
ViT never saw in training — the same class of mismatch that cost 83-93% of the
living rate when the deconvolution changed. So the bar must be **exactly
removable**.

The pre-existing `utils.add_scale_bar` could not be undone: it padded **left
and right** for narrow crops, and centred padding is unrecoverable from pixels.

## What was added (commit `7492061`, `utils.py`)
* `add_scale_bar_strip(img, pixel_resolution=23)` — appends a fixed
  **34-row strip below**; never alters the crop's pixels.
* Bar is **1 mm where it fits, else 500 um**, with no shorter step. A crop too
  narrow for a 500 um bar is **padded on the right**, so the object stays
  left-aligned in the same place every time.
* `save_crop_with_scale_bar()` records the original size in a **PNG text
  chunk**; `strip_scale_bar()` restores it exactly (strip **and** padding).
* Verified byte-identical round trips from 10 px wide (padded to 28) to 1951 px.

The 500 um floor came from real data: on an SO308 sample **19% of living and
87% of not-living** crops were too narrow — the earlier adaptive ladder gave
those useless 20-50 um bars.

## Caveat
Undoing the **padding** depends on the PNG text chunk surviving. Read crops
with PIL, which preserves it. If they come back through something that
re-encodes and drops metadata, only the bottom strip can be removed — right
padding is indistinguishable from genuine white background. The alternative,
if that is a problem, is padding *every* crop to a fixed minimum width so the
geometry is uniform and inferable.

## Open — state as observed 2026-11-13

Veit reports having wired these in "at another point", but a search of both
repos does not find it, so **this needs confirming before relying on it**:

* `add_scale_bar_strip` / `save_crop_with_scale_bar` / `strip_scale_bar` are
  defined in `utils.py` and **called from nowhere** in the repo.
* All four export call sites in `process_pisco_profiles.py` still hardcode
  `add_scale_bar_to_deconv=False`, and `create_ecotaxa_zips` still calls the
  old non-reversible `add_scale_bar`.
* `upload_to_ecotaxa.py`, `export_ecotaxa_zips.py` and the segmenter /
  Hub branch contain no scale-bar code at all.
* Project 23286's bars were applied by **post-processing the zips by hand**.

So either the wiring lives somewhere outside these two repos, or it is still
to do. Wanted either way: a `--scale-bar` flag on export, and a
`strip_scale_bar()` call in the ViT preprocessing path — without the latter,
anything trained or classified on scale-barred crops sees the bar.

## New-chat prompt
> Read docs/handoff/06-scale-bars.md. Wire the reversible scale bar into
> create_ecotaxa_zips behind a --scale-bar flag, and call strip_scale_bar in
> the ViT preprocessing path.
