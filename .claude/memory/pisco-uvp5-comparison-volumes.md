---
name: pisco-uvp5-comparison-volumes
description: "PIScO vs UVP5 (M181, projects 20736/17905): how sampled volumes are derived, the mask-overhang volume bug, 27 um ESD in 20736, and the unexplained ~10x gap"
metadata:
  node_type: memory
  type: project
  originSessionId: 758fc6ab-0323-45e0-a86f-3d13f8923156
  modified: 2026-11-17T14:35:00.328Z
---

Built 2026-11-17: `compare_pisco_uvp5.py` (untracked, repo root) -> outputs in `/media/veit/T710_data/pisco_processed/M181_PISCO_vs_UVP5/`. Five shared casts: CTD-037/050/060/075/081.

**PIScO frame volume:** the segmenter's reader *always* resizes frames to 2560 px (raw M181 PNGs are 5120x5120), so 22.5 um/px holds. The mask circle (r=1500) **overhangs** the 1280 px half-width: imaged area is the circle-square intersection = 87% of pi r^2 -> **1.55 L/frame**, not 1.79 (r=1550: 1.59, not 1.91). `calc_abundance_biovolume_pisco.calculate_volume` still uses pi r^2 (overstates volume ~15%). Depth of field 500 mm is that script's unverified constant.

**Project 20736's object_esd was exported at 27 um/px** (esd/sqrt(4A/pi) = 27.000 exactly) -> recompute from area.

**UVP5 volume:** use EcoPart (user downloaded `export_reduced_20261117_14_35.zip`, repo root, all M181 casts; pass `--ecopart`). The fallback proxy (UVP N_img spread along the PIScO timeline) matches EcoPart within 5% per whole cast but is **2-3x too high in 0-100 m**, because EcoPart keeps descending images only (drops soak/heave). UVP vignettes are complete: vignette counts >=1 mm = 1.2x the EcoPart LPM counts on every cast.

**Open question:** all PIScO objects >=1 mm vs UVP LPM (classifier-free): **1.5-3x below 200 m** but **4-15x in 0-200 m**, where PIScO >=1 mm objects are heavily bubble (0-20 m) / streak (50-100 m). **CTD-060 is 8-13x at all depths** — suspect that PIScO cast specifically. Living-only, 0-100 m, >=1 mm: PIScO 7-25x UVP5.

Related: [[scale-bar-pixel-size-22-5]], [[pisco-two-routes-hdf5-hub]].

**Reference figure (report Fig. 14, 0-100 m) decoded 2026-11-18:** PIScO = *validated only, as of Aug 2025* (obj.classif_date < 2025-09-01; 10.4k more validations came Feb-Jun 2026, mostly Trichodesmium, CTD-060/050/081) with the old volume convention (pi r^2 x 500 mm, r=1550, absolute pressure bar*10) -> matches within 2-5% per cast. UVP5 = *all objects (V+P)* with EcoPart volumes -> matches within ~5%. So the reference mixes validated-only PIScO with all-objects UVP5. Validation dates: `obj.classif_date` in the object_set query (classif_when/who are empty here). PIScO project re-predicted 2026-04-15; UVP predictions 2025-08-25, no later validations.

**UVP5 size convention (2026-11-18):** script now defaults to *linear* 0.088 mm/px (acq `pixel` field) for vignette ESD/biovolume (`--uvp-size powerlaw` keeps aa*A^exp). The power law inflates >5 mm vignettes ~3x in biovolume (CTD-060 0-100 m: 0.60 vs reference 0.39, driven by 3 big Cnidaria/Appendicularia). Reference Fig. 15 used a linear pixel; ours with 0.088 is uniformly ~9% below it (a pixel fit gives 0.091 for every group, i.e. a volume/binning offset, same as in abundance).
