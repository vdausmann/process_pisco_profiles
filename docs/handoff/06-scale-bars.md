# 06 — Scale bars on annotator crops

**State: CLOSED — landed, wired, and on by default in the HDF5/Hub export.**
Updated 2026-11-13 after rebasing onto GitLab `main`: the wiring Veit referred
to does exist, and an earlier revision of this file wrongly called it missing.

## Why it is built the way it is
Crops shown to annotators need a size reference, but crops are also
**classifier input**. A bar burnt into the image is an appearance change the
ViT never saw in training — the same class of mismatch that cost 83-93% of the
living rate when the deconvolution changed. So the bar must be **exactly
removable**, and it must never reach a model.

The pre-existing `utils.add_scale_bar` could not be undone: it padded **left
and right** for narrow crops, and centred padding is unrecoverable from pixels.

## Where it lives now: `scale_bar.py`
Added in `7492061` inside `utils.py`, then **lifted into its own module**
(`6cb4b48`) so callers needing only a bar do not pay for `utils`' imports —
that module pulls in torch, transformers, pandas and matplotlib at module
scope, none of which are installed on the processing workers. `scale_bar.py`
needs nothing but Pillow. `utils` re-exports the names, so old callers still
work.

* `add_scale_bar_strip(img, pixel_resolution=PIXEL_SIZE_UM)` — appends a fixed
  **34-row strip below**; never alters the crop's own pixels.
* Bar is **1 mm where it fits, else 500 um**, with no shorter step. A crop too
  narrow for a 500 um bar is **padded on the right**, so the object stays
  left-aligned in the same place every time.
* `save_crop_with_scale_bar()` records the original size in a **PNG text
  chunk**; `strip_scale_bar()` restores it exactly (strip **and** padding).
* Verified byte-identical round trips from 10 px wide (padded to 28) to 1951 px.

The 500 um floor came from real data: on an SO308 sample **19% of living and
87% of not-living** crops were too narrow — the earlier adaptive ladder gave
those useless 20-50 um bars.

### Pixel size was corrected to 22.5 um/px (`ad1b926`)
`PIXEL_SIZE_UM = 4.5 * (400/160) * 2` — PISCO2 optics at 2560x2560: a 4.5 um
sensor pixel, 400/160 inverse magnification, binning 2. It was **23** until
2026-09-09, which disagreed with `area_to_esd()` by **2.2%**. Defined once and
imported by the exporter, so a **drawn bar and a reported `object_esd` now
agree**. Anything measured off a bar drawn before that date is 2.2% out.

## How it is wired
`export_hdf5_to_ecotaxa.py` imports `PIXEL_SIZE_UM`, `SCALE_BAR_STRIP_H` and
`save_crop_with_scale_bar`, and exposes:

* `--scale-bar / --no-scale-bar` (`BooleanOptionalAction`, **default on**)
* `--pixel-size` (default `PIXEL_SIZE_UM`), used for the bar **and** `object_esd`

Both are surfaced as Hub parameters in `toolbox.json` under
`pisco_ecotaxa_export`.

## Why no `strip_scale_bar()` is needed in the ViT path
This was listed as an open item; it is not one. In the current architecture
**no model ever sees a barred crop**:

* `vit_classify_hdf5.iter_crops()` reads crops straight from the segmenter's
  `result.h5`. The bar is drawn only at **export** time, after classification.
* `training/build_trainset.py` imports that **same `iter_crops`** — literally
  the inference function — or falls back to the old pipeline's
  `deconv_crops*.zip`. `training/README.md` states the rule explicitly: *"The
  EcoTaxa export zips are not a source: their PNGs carry a scale bar no model
  was trained with."*
* `train_vit.py` asserts at startup that its preprocessing equals
  `vit_classify_hdf5.preprocess` and exits if it has drifted.

So the bar is confined to the annotator-facing export. Keep it that way: the
moment anyone trains from an export zip, this guarantee is gone.

## Caveat that remains
Undoing the **padding** depends on the PNG text chunk surviving. Read crops
with PIL, which preserves it. If they come back through something that
re-encodes and drops metadata, only the bottom strip can be removed — right
padding is indistinguishable from genuine white background. The alternative,
if that ever matters, is padding *every* crop to a fixed minimum width so the
geometry is uniform and inferable without metadata.

## Loose ends in the OLD python route
The old `process_pisco_profiles.py` route never got the reversible bar:

* all four `create_ecotaxa_zips` call sites still pass
  `add_scale_bar_to_deconv=False`;
* `create_ecotaxa_zips` still calls the **old, non-reversible** `add_scale_bar`
  and still defaults to **`pixel_resolution=23`**, the superseded figure —
  while `add_scale_bar` itself now defaults to `PIXEL_SIZE_UM`. If that path is
  ever switched on, it will draw 2.2%-wrong bars.
* Project **23286**'s bars were applied by post-processing the zips by hand,
  before the wiring existed.

Only worth fixing if the old route is still to be used; otherwise the HDF5
route supersedes it.

## New-chat prompt
> Read docs/handoff/06-scale-bars.md. The reversible scale bar is wired into
> export_hdf5_to_ecotaxa.py. I want to <fix create_ecotaxa_zips in the old
> route to use the reversible bar and 22.5 um/px / verify a round trip on a
> real export / leave the old route alone>.
