---
name: atair-bsh-writeup
description: "Publication-structured writeup of the whole ATAIR-BSH PISCO effort — deconv-model comparison, crop isolation, EcoTaxa upload, dual-ViT predictions, and the human validation + not-living re-screening, with the number flow and the validators' caveats"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-09-01T12:59:14.335Z
---

# PISCO imaging of the ATAIR‑BSH 2026 profiles: processing, dual‑ViT classification, and expert validation

**Cruise / platform:** ATAIR‑BSH, April 2026. Underwater dark‑field plankton imager (PISCO), ~23 µm/pixel.
**Profiles (3):** `20260422-2051`, `20260423-1100`, `20260423-1705` — 701 + 532 + 1231 = **2,464 frames** (2560×2560, `.tif`).
**Processing:** 2026‑08‑10. **Validation:** through 2026‑08‑14. **Validators:** Veit Dausmann, Paula Ehlebracht, Fanny Brodbek (EcoTaxa on ecotaxa.obs‑vlfr.fr).

---

## 1. Image‑processing pipeline
Each frame passed through: rolling‑window background correction → **LUCYD deconvolution** (learned Richardson–Lucy variant) → threshold‑based segmentation (fixed threshold 10 on the inverted, background‑corrected frame; contours ≥ 50 px saved) → per‑object crops. Objects were classified by a **dual‑ViT**: a binary living / not‑living classifier followed, for the living, by a fine‑grained taxonomic classifier (`ViT_custom_size_sensitive_v5`). Deconvolved crops were exported to EcoTaxa in two fractions per profile (`other/` = living, `non-living/` = not‑living).

## 2. Deconvolution‑model comparison (dual‑ViT, all 3 profiles)
Four LUCYD models were run with the **identical** downstream pipeline and ViT; only the deconvolution changed:

| Deconv model | Total objects | Living | Living rate |
|---|--:|--:|--:|
| v4 (`…_v4_20260722`) | 1,077,477 | 3,126 | 0.29% |
| v5d (`…_v5d_20260807`) | 1,076,221 | 4,526 | 0.42% |
| v5 (`…_v5_20260806`) | 799,922 | 6,812 | 0.85% |
| **v5a (`…_v5a_20260807`)** | **645,090** | **7,364** | **1.14%** |

Two consistent trends as the model improves (v4→v5a): **total detections fall** (fewer spurious particles on noisy background) while the **living count rises** — monotonically on every profile. **v5a** was selected for validation.

**Critical caveat (retained as a limitation).** The ViT was trained on crops from the *old* deconvolution model, so a new deconvolution is an out‑of‑domain appearance shift: in a controlled same‑object test the newer models pushed real plankton into *not‑living* (living rate collapse of ~74–93% vs the old model), at high confidence. **Absolute abundances from any of these runs are therefore indicative only** — the ViT must be re‑trained on crops from the deployed deconvolution model before abundances/biovolumes are quantitative. Deconvolution and classifier must be versioned as one coupled unit.

## 3. Crop isolation (for the eventual re‑training set)
Saved crops are rectangular bounding boxes and, because segmentation is threshold‑based, a crop can contain neighbouring objects. Quantified on profile `20260423-1705` (251,020 crops): **4.8%** of crops contained a neighbour object's centre, **0.7%** a *substantial* neighbour (≥10% of the primary's area); of the neighbours, ~half were separately‑cropped objects and ~half sub‑50 px specks. An opt‑in isolation step (`PISCO_ISOLATE_CROPS`, "erase other contours" — whites out every contour except the target, target protected from the 1 px dilation) was implemented; it leaves detection (and thus the crop count) unchanged but drops multi‑object crops from **15.0% → 0.5%**. v5a was re‑run with isolation for the uploaded project. Isolation is an appearance change and is intended to pair with the ViT re‑training, not to be trusted under the current classifier (living moved only 7,364 → 7,255).

## 4. Taxonomic harmonization
Two classifier category names are deprecated in EcoTaxa and were remapped at export so imports are validatable: `Asteroidea larvae → Asteroidea` (12877) and `Noctiluca sp. → Noctiluca` (58408). These remapped names are *ambiguous* on import (several lineages share the name), and were resolved to the **approved** ids at import time to avoid landing objects on deprecated taxa (which then cannot be validated).

## 5. EcoTaxa projects
| Project | id | Content |
|---|--:|---|
| ATAIR‑BSH_PISCO_v5a_lucyd (isolated) | **23022** | 7,255 living + 20,000 random not‑living subset = 27,255 |
| ATAIR‑BSH_PISCO_v4_lucyd | 23021 | 3,126 living + 20,000 not‑living = 23,126 |
| ATAIR_PIScO_v5d_lucyd | 23014 | v5d living + 20,000 not‑living = 24,526 |

Only the **living** fraction plus a **random 20,000‑object subset of the not‑living** were uploaded per model (the full not‑living fraction is ~638k objects; EcoTaxa ingest runs at a fixed ~52 objects/s, so uploading all of it was impractical and unnecessary for validation). The random not‑living subset is the vehicle for estimating the miss rate and hunting missed living.

## 6. Human validation of the living fraction (project 23022)
The dual‑ViT's living predictions were reviewed and corrected in EcoTaxa. The dominant shift:

| Taxon | dual‑ViT predicted | after validation |
|---|--:|--:|
| Copepoda | 2,931 | 2,320 |
| **Rhizaria → pluteus (Echinoidea)** | 2,354 (as *Rhizaria*) | **2,212 (as *pluteus*)** |
| Appendicularia | 1,560 | 1,066 |
| Cnidaria | 162 | 127 |
| *(new on validation)* multiple species | — | 667 |
| *(new)* gelatinous | — | 59 |
| *(new)* not‑living / Unknowns / bubble / t001 | — | small buckets |
| Pyrocystis, Thaliacea, Chaetognatha, Trichodesmium, Noctiluca, Eumalacostraca, Asteroidea, Ctenophora, Bacillariophyceae | 53 / 52 / 45 / 29 / 29 / 24 / 10 / 4 / 2 | mostly re‑absorbed / few validated |

**The headline correction:** the classifier's second‑largest living class, "Rhizaria" (2,354), was on inspection almost entirely **echinoderm pluteus larvae** — a systematic confusion the current ViT makes and which validation resolved. New human categories (`multiple species`, `gelatinous`, `Unknowns`, `t001`) were introduced during validation for objects the ViT's label set did not cover.

## 7. Re‑screening the not‑living fraction for missed living
To recover living plankton mis‑binned as not‑living, the not‑living subset was iteratively re‑classified with EcoTaxa's **deep‑feature (SCN, GPU) prediction**, learning from the growing validated set (self‑training / active learning). Key steps:

- A random subset of the not‑living (**23052**, 20,775 objects) was screened. A first prediction with no not‑living class in the learning set forced everything into living classes; a catch‑all **"Unknowns"** category was then created and validated so the model had a fall‑back class.
- Iterative SCN re‑prediction (learning from validated objects) progressively concentrated the mass into *Unknowns* and surfaced a shrinking set of living candidates: **17,832 → 737 candidate Copepoda + a few pluteus/gelatinous → 16,395 Unknowns remaining.** Confidence scores were used to prioritise (low‑score objects reviewed first).
- A final random subset (**23196**, 2,500 objects) drawn from the remaining Unknowns and predicted from the combined validations (23052 + 23022, 10,115 validated objects, 11 classes) sorted as **2,494 Unknowns, 4 gelatinous, 2 pluteus** (median confidence 0.96). On full validation: **2,492 Unknowns, 5 pluteus, 3 dubious.**

The steep diminishing return (17,832 living candidates → ~6 in a fresh 2,500) indicates the living‑in‑not‑living pool was **essentially exhausted** — i.e. the dual‑ViT's living / not‑living split, while it collapses absolute living counts (§2), did not leave a large reservoir of obviously‑living objects in the not‑living fraction of these profiles.

## 8. Limitations and validation caveats (reported verbatim from the validators)
These are honest limitations to state in any publication:

- **"Unknowns" is a hoover.** The validator's own assessment: *"All the images are very hard to tell what's on them, and thus the category 'Unknowns' is very likely correct — but in the end it is a hoover for anything of any shape."* Unknowns is a low‑information catch‑all, not a taxonomic statement.
- **Rapid batch validation.** Late‑stage not‑living validation was done *"flying over batches of 100 objects, mostly selecting everything and validating it as correctly predicted,"* so there is a *"non‑zero chance I might have missed some objects that might be classifiable."* Possible bias toward confirming the predicted (Unknowns) label.
- **"Similar‑to" search as a safety net.** When a classifiable object was spotted (*"very few copepods and pluteus larvae"*), EcoTaxa's *similar‑to* image search was used to pull in visually similar objects, *"so I'm quite confident I got the biggest part of the objects in the right category."*
- **Dubious flag is used inconsistently across projects.** For the **later** (re‑screening) projects, *dubious* meant *"I'm confident it is some kind of zooplankton but I'm not sure what it is."* In **project 23022**, Veit and Paula used it differently — closer to *"not sure whether the object matches the predicted class"* (and in some cases *"but I didn't find a suitable class"*). Dubious‑flagged counts are therefore **not directly comparable** between the living project (23022) and the not‑living re‑screening projects, and should not be pooled without noting this.
- **Absolute abundances are provisional** pending a ViT re‑trained on the deployed deconvolution (§2).

## 9. Number flow (v5a pipeline, summary)

```
ATAIR-BSH · 3 PISCO profiles · 2,464 frames
        │  dual-ViT  (LUCYD v5a deconvolution, isolated crops)
        ▼
   645,090 objects detected
   ├── 7,255 LIVING (1.12%)  ─────────────►  project 23022, expert-validated
   │        predicted (ViT):  Copepoda 2931 · "Rhizaria" 2354 · Appendicularia 1560 · …
   │        validated (human): Copepoda 2320 · pluteus 2212 (← the "Rhizaria")
   │                            · Appendicularia 1066 · multiple 667 · gelatinous 59 · …
   │
   └── 637,835 NOT-LIVING (98.9%)
            │  random subset 20,000  ─────►  re-screen for missed living
            ▼
        23052 (20,775): iterative SCN re-prediction + validation
            · created "Unknowns" catch-all
            · living candidates: 17,832 → 737 → … → 16,395 Unknowns left
            │  random subset 2,500  ─────►  final confirmation
            ▼
        23196 (2,500): 2,492 Unknowns · 5 pluteus · 3 dubious   →  pool exhausted
```

## 10. Provenance (reproducibility)
- Models: `underwater-lucyd/models/lucyd-edof-plankton_{v4_20260722,v5d_20260807,v5_20260806,v5a_20260807}.pth`; classifier `ViT_custom_size_sensitive_v5` (binary + living). Output root `/media/veit/T710_data/pisco_processed/ATAIR-BSH_v5alucyd_iso` (and `_v4lucyd`, `_v5lucyd`, `_v5dlucyd`).
- Code (`process_pisco_profiles` / `pisco_segmenter`): living/non‑living export‑split fix `f0ea603`; deprecated‑taxon remap `405eeb2`; opt‑in crop isolation `8480520`.
- Custom‑folders mode was removed upstream; ATAIR's non‑standard layout was run via a symlink‑tree into cruise mode with `--image-ext .tif` in the pipeline pass‑through.
- EcoTaxa projects: 23022 (v5a‑iso, main), 23021 (v4), 23014 (v5d); not‑living re‑screening subsets 23052 and 23196. Project 23020 (v5a, non‑isolated) was created then erased and replaced by 23022.

Related memories: [[deconv-model-swap-breaks-vit]], [[crop-isolation-erase-other-contours]], [[deconv-crops-are-white-clipped-at-245]], [[ecotaxa-deprecated-taxa-validate-fails]], [[ecotaxa-import-asks-ambiguous-taxa]], [[ecotaxa-import-throughput-living-only]], [[custom-folders-mode-regression]].
