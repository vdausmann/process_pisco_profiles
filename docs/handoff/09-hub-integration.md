# 09 — PIScO Hub integration (segmenter), and what it changes for us

**State: landed on `pisco_segmenter_py` GitLab `main`; the local checkout is
behind.** Reviewed 2026-11-13, read-only — nothing was pulled or merged.

## The 9 commits
`88c8015` headless GPU worker · `0b8b3be` describe pipeline as a Hub tool ·
`87cd167`+`87df281` write `result.h5` alongside PNG/CSV · `acab6ed` make the
circular FOV mask switchable · `9f5fe67` **estimate the background as the C++
segmenter does** · `d15dcea` merge · `4af9b4d` **correct the background per
block** · `8eff671` read frames from a profile dataset as well as a directory.

New files: `frames.py`, `hdf5_writer.py`, `run_profile.py`, `toolbox.json`,
`run.sh`, plus the v5 ablation checkpoints vendored into
`pisco_segmenter/models/`.

**All four of this session's segmenter commits survive intact** on that branch
(`99a0d88`, `c1e03cd`, `8480520` crop isolation, `b805b9a` hang fix) — verified
by ancestry and by grepping the merged files.

## Does it change our workflow? **Yes — materially.**

### Background estimation changed, and there is no switch
| | before (what every result in this session used) | after |
|---|---|---|
| scope | per-frame **rolling window**, frame excluded | **per block of 12**, one background shared by the block |
| estimator | `np.max(bg_imgs)` | **max over pairwise minima** |

`PISCO_BG_SIZE` overrides the block size (default **12**); the *estimator*
itself has no legacy path.

**Why it was done** is sound and worth knowing: 12 is the stack the LUCYD
background textures were built with, so this keeps the **deconvolution input in
distribution**. That is a real improvement to the deconvolution step.

**But it changes the cleaned frame**, therefore the deconvolved crop,
therefore the thresholded contours. So:
1. **Object counts and crop appearance will differ** from everything recorded
   in this session. Abundances are not comparable across the change.
2. It is a **third domain shift** stacked on deconvolution model and crop
   isolation. Both published ViTs (`Veit/M181_*`, `Veit/ATAIIR2604_NorthSea_*`)
   were trained on **old-background** crops.
3. The gain is real but **unmeasured for classification** — better LUCYD input
   does not automatically mean better crops for a classifier trained on the old
   look. Treat it the way the deconvolution comparison was treated: measure,
   do not assume.

### Smaller changes
* **Input**: `frames.py` reads a directory **or** a `profile_dataset.h5`.
  Directory input still works, so existing invocations are unaffected.
* **Output**: `result.h5` is written **alongside** the PNGs and CSVs, not
  instead of them. Note its mask is **the object's own contour**, unlike the
  PNG crops, which are cut with the **accumulated frame mask** — independent
  confirmation of the accumulation behaviour found during the isolation work
  (topic 02).
* **FOV mask** is now overridable (`mask_img_override`); the default is
  unchanged (on).
* New knobs: `PISCO_BG_SIZE` (12), `PISCO_DECONV_BATCH`, `PISCO_LOG_FILE`.
* `run_profile.py` + `toolbox.json` package the segmenter as a Hub tool — a
  second entry point beside `orchestrate_gpus.py`.

## Decisions this forces
1. **Pull or stay pinned?** The local checkout is behind. Pulling silently
   changes the output of every future run. If SO308 is to stay comparable with
   the Madagascar subset already processed, either finish SO308 on the current
   code or re-run the two completed profiles after pulling.
2. **If standardising on the block background**, both ViTs want retraining on
   matching crops — and that is also the moment to settle the deconvolution
   choice (topic 02) and switch crop isolation on, so one retrain absorbs all
   three shifts instead of three retrains.
3. **GitHub is also behind** and has not received these commits.

## New-chat prompt
> Read docs/handoff/09-hub-integration.md. The segmenter's background
> estimation changed to per-block max-of-pairwise-minima. I want to
> <measure its effect on crops/classification before pulling / pull and retrain
> the ViTs / keep SO308 pinned to the current code>.
