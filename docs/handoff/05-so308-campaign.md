# 05 — SO308 processing campaign

**State: OPEN. This is the live thread.** 2 of 42 profiles processed.

## The profile list
From `RELEVANT_202506003-SO308_Stations_overview.xlsx`, sheet
**`Interesting_Stations`** (49 rows). Match on **`D_ship_ID`**, not the
`# PISCO profile` number — they diverge (profile 9 is SO308_012). Directories
are `<D_ship_ID>_PISCO2_<date>-<time>` and the time differs slightly from the
sheet, so match the id prefix and disambiguate by date.

* **40** matched one directory
* **2** ambiguous (same ship-id, two casts) — resolved by date+time:
  `SO308_1-15-1` -> `20241112-2304`, `SO308_1-19-1` -> `20241115-1249`
* **7 have no source data on the filer** despite the sheet marking
  `PNG on Filer = x`:
  `1-12-1, 1-13-2, 1-16-1, 1-17-4, 1-18-3, 1-33-1, 1-45-1`.
  Two survive only as segmentation-result ZIPs in `SO308-Results/`; five are
  nowhere under `/mnt/filer/SO308`. **Check the external HDD.**

Layout needs no workaround — `/mnt/filer/SO308/SO308-PISCO-Profiles/<profile>/<profile>_Images-PNG`
matches cruise mode, and SO308's CTD/logfile configs are already in the
pipeline. Images are 2560x2560 PNG.

## Scale of the full job
1,490,620 frames across 42 profiles (mean 35,490, max 65,290). At the measured
rate (~9,700 frames/h on both GPUs) that is **~150 h / 6+ days**, and hundreds
of millions of intermediate crop files. `--cleanup-intermediate` is mandatory.
For comparison, all of ATAIR was 2,464 frames.

## What has been run (Madagascar subset)
Two profiles closest to Madagascar, chosen so both GPUs finish together:

| Profile | Objects | Living | Rate |
|---|--:|--:|--:|
| SO308_1-20-1 (~120 km SE) | 750,897 | 5,591 | 0.74% |
| SO308_1-21-3 (~50 km E) | 403,788 | 2,121 | 0.53% |
| **Total** | **1,154,685** | **7,712** | **0.67%** |

Ran in 7 h. Output `/media/veit/T710_data/pisco_processed/SO308_v5a_madagascar`.
Config: **v5a deconvolution + OLD M181-trained ViT + isolation OFF** (isolation
off deliberately — the M181 ViT was trained on non-isolated crops).

Living taxa: Copepoda 3,001 · Appendicularia 2,281 · Trichodesmium 809 ·
Cnidaria 556 · Rhizaria 361 · Noctiluca 151 · Bacillariophyceae 140 ·
Chaetognatha 127 · Pyrocystis 108 · Thaliacea 98 · Asteroidea 47 ·
Eumalacostraca 28 · Ctenophora 5.

## Validation set uploaded
Project **23286** `SO308_Madagascar_v5a_validation` (private; managers Veit,
Tobias Strickmann, Petra Bilic): **7,712 living** (all available — fewer than
the 10k requested) + **5,000 random not-living** = 12,712 objects, **with scale
bars**. Unvalidated.

## Things to watch
* 0.67% living is *lower* than ATAIR's 1.12% under the same v5a + old-ViT
  pairing — the domain mismatch bites here too, and this is oligotrophic open
  ocean (~10 objects/frame vs ATAIR's ~204).
* Taxa look M181-shaped: **`Rhizaria` 361 and no pluteus**. Given the ATAIR
  finding that this classifier calls pluteus larvae Rhizaria, watch for that
  during validation.

## Open
1. Resolve the 7 missing profiles (external HDD?).
2. Decide whether to run the remaining 40 (~6 days) and with which classifier.
3. Validate 23286.

## New-chat prompt
> Read docs/handoff/05-so308-campaign.md. 2 of 42 SO308 profiles are done
> (Madagascar). I want to <run the rest / resolve the 7 missing / validate 23286>.
