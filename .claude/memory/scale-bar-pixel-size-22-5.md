---
name: scale-bar-pixel-size-22-5
description: "PISCO2 pixel size is 22.5 um/px, not 23 — corrected 2026-09-09; bars drawn before that are 2.2% out"
metadata:
  type: project
---

`scale_bar.PIXEL_SIZE_UM = 4.5 * (400/160) * 2` = **22.5 um/px** — PISCO2 optics at 2560x2560: 4.5 um sensor pixel, 400/160 inverse magnification, binning 2.

It was hardcoded as **23** until 2026-09-09 (commit `ad1b926`), which disagreed with what `area_to_esd()` derives by **2.2%**. It is now defined once in `scale_bar.py` and imported by `export_hdf5_to_ecotaxa.py` for both the drawn bar and the reported `object_esd`, so the two agree.

**How to apply:** any scale bar drawn before 2026-09-09 — including the hand-post-processed zips of EcoTaxa project **23286** (SO308 Madagascar) — is 2.2% out. Don't take measurements off those bars without correcting. `utils.create_ecotaxa_zips` in the old python route still defaults to `pixel_resolution=23`, so it would reintroduce the error if switched on. Related: [[pisco-two-routes-hdf5-hub]].
