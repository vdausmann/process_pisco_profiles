---
name: custom-folders-mode-regression
description: commit 02657c0 silently deleted --folder/--folders/--folders-file (custom-folders mode); README still documents it. Symlink-tree workaround for non-standard layouts like ATAIR
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-10T10:33:08.330Z
---

`process_pisco_profiles.py` **no longer has** `--folder`, `--folders` or `--folders-file`.
Commit **02657c0** ("Add --cleanup-intermediate...") was a +72/-121 line change that, besides
adding cleanup, **removed the entire custom-folders code path** (`using_custom_folders`, the
three CLI args, and the folder-as-profile branch). `README.md` sections 5-7 still document it,
so the docs and the code disagree. Found 2026-08-07 when re-running the ATAIR profiles.

**Why it matters:** cruise mode requires the layout
`<source>/<cruise>/<cruise>-PISCO-Profiles/<profile>/`. Cruises with a different layout — e.g.
ATAIR at `/mnt/filer/202604_ATAIR-BSH/202604_ATAIR-BSB_PISCO/<date>/<profile>` — can ONLY be
run through custom-folders mode. The May 2026 ATAIR run used it (its logs landed in the output
dir, which is the custom-folders behaviour); that is no longer reproducible from the CLI.

**How to apply — symlink-tree workaround (no code change):**
```
SRC=/media/veit/T710_data/pisco_src_ATAIR/ATAIR-BSH/ATAIR-BSH-PISCO-Profiles
mkdir -p "$SRC"
while read -r p || [ -n "$p" ]; do ln -sfn "$p" "$SRC/$(basename "$p")"; done < folders_ATAIR.txt
```
then run normal cruise mode with `--source /media/veit/T710_data/pisco_src_ATAIR
--cruise ATAIR-BSH --image-ext .tif`. This works because `find_image_dirs()` appends the
profile dir itself as an image-dir candidate, and `get_adapter_for_cruise()` falls back to
`GenericCruiseAdapter` for unknown cruise names. Verified: "Profiles with images: 3".

**GOTCHA via orchestrate_gpus.py (cost a silent no-op run 2026-08-10):** the orchestrator has
its own `--image-ext` (used only for LPT sizing) and does **NOT** forward it to
`process_pisco_profiles.py`, which then defaults to `.png` and reports "Profiles with images: 0
/ ERROR: No images found" — but the orchestrator still prints `OK` per profile in ~0.2 min, so
it looks like it succeeded. Pass `--image-ext .tif` **in the passthrough too**:
`orchestrate_gpus.py --cruise ATAIR-BSH --image-ext .tif ... -- --image-ext .tif --export-zip --cleanup-intermediate`.
Sanity check: a real run of these profiles is ~15-20 min, not seconds; verify a per-profile
`gpu*.log` says "Profiles with images: 3" and "start segmentation".
NOTE `folders_ATAIR.txt` has **no trailing newline** — a plain `while read` silently drops the
third profile; use `|| [ -n "$p" ]`.

Proper fix (not yet done): restore the custom-folders branch from `02657c0^`, or make cruise
mode accept absolute profile paths. Related: [[pisco-run-with-pisco-conda-env]],
[[choose-deconvolution-model]].
