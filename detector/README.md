# Synthetic raw frames for an instance-segmentation detector

Dense profiles (e.g. ATAIR 20260426-1750) defeat background correction: no
pair of consecutive frames is clear at every pixel, so any background estimated
from the profile itself keeps the haze. The plan is to skip background removal
there and train an instance-segmentation model on **raw** frames, using
synthetic scenes built from validated EcoTaxa objects.

## Steps

```bash
# 0. validated labels (last project wins on duplicates)
python ecotaxa_fetch_labels.py --projects 23022 23196 23210 \
    -o /media/veit/T710_data/detector_trainset/labels \
    --url https://ecotaxa.obs-vlfr.fr --credentials ~/.config/pisco/ecotaxa-online.json

# 1. clean reference background (1705 has a brightness step at ~frame 467)
python deconv_fullframes.py --background-only \
    /media/veit/T710_data/detector_trainset/background_20260423-1705_f470.png \
    --frames 470: /mnt/filer/202604_ATAIR-BSH/202604_ATAIR-BSB_PISCO/20260423/20260423-1705

# 2. transmittance patches of the validated objects, from the raw frames
python detector/extract_transmittance.py \
    --labels /media/veit/T710_data/detector_trainset/labels/labels.csv \
    --out /media/veit/T710_data/detector_trainset/objects

# 3. synthetic scenes + COCO instance annotations
python detector/synth_scenes.py \
    --objects /media/veit/T710_data/detector_trainset/objects \
    --background /media/veit/T710_data/detector_trainset/background_20260423-1705_f470.png \
    --out /media/veit/T710_data/detector_trainset/synth_v1 --n-train 2000 --n-val 200
```

Run in the `pisco` conda env.

## Why it is built this way

* **Transmittance, not cut-outs.** A shadowgraph object attenuates light, so
  `t = frame / background` composes by multiplication; overlapping objects
  come out like real overlapping shadows and the defocus halo is kept. The
  mask is only the label; the patch is faded out 8 px beyond it.
* **Masks** are two-level thresholds on `t` (strong 8 sigma connected to the
  object's box, plus weak 4 sigma within 6 px; 2.5 sigma near the box for very
  faint objects). Thin appendages are partly missed.
* **Dirt is made, not found**: Unknowns / unidentifiable / not-living patches,
  rescaled and blurred, 6-14k per dirty scene. Defaults fitted to 1750's
  per-scale structure (see the `synth_scenes.py` docstring).
* **Labelled classes**: 7 living (Copepoda, pluteus, Appendicularia, Cnidaria,
  gelatinous, Chaetognatha, Ctenophora) + `not-living` (sharp dirt). Overlaps
  keep both full masks, so "multiple" needs no class of its own.
* **Split by source object** (hash of object_id, 15% val).

## Known gaps

* The objects come from three clean profiles only (2051, 1100, 1705).
* Patches touching their patch or frame edge are not pasted (~320 of 9,911).
* No real annotated dense frames yet -- needed to tell whether synthetic
  training carries over.
