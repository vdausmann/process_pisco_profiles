# 08 — Repos, reproducibility, known issues

## Commits made this session

**`process_pisco_profiles`** — rebased onto GitLab `main` on 2026-11-13 and
pushed to both remotes. GitLab had **31 commits** of Hub/HDF5 work on top of
`7492061` that the local checkout had not seen (topic 09); this session's
commits now sit on top of those.
| Commit | Change |
|---|---|
| `bf6f5f3` | M181 sample_id = cruise + CTD cast |
| `9227351` | zip multi-part crops in place (inode blowup) |
| `f9de900` | zero-pad object_date/object_time |
| `f0ea603` | fix living/non-living export split (`not-living` label) |
| `405eeb2` | remap deprecated EcoTaxa taxa at export |
| `98e8b5e` | `--dualvit-model` shorthand |
| `7f8dcf2` | `--dualvit-model` falls back to Hugging Face |
| `cc30f6e` | `training/` — dual-ViT recipe |
| `7492061` | reversible scale bar (since lifted into `scale_bar.py` upstream) |
| `75db55a` | this handoff documentation |

**`pisco_segmenter`** (= `pisco_segmenter_py`, branch `main`)
| Commit | Change |
|---|---|
| `99a0d88` | `PISCO_N_CORES` |
| `c1e03cd` | deconv-only recrop mode |
| `8480520` | opt-in crop isolation (`PISCO_ISOLATE_CROPS`) |
| `b805b9a` | **hang fix** (see below) |

**`underwater-lucyd`** — `7091a78` tracked the 18 v2..v5f deconvolution models
(1.8 MB) and a GitLab remote was wired up; previously they existed only on the
training machine, so published results were unreproducible.

## The hang fix (`b805b9a`) — worth knowing about
A profile sat idle for 31 minutes with no GPU load and no log output. Three
defects compounded:
1. `ProcessPool.worker` caught only `queue.Empty`, so any task exception killed
   the worker; once all three died the pool silently stopped consuming.
2. `correct_img` stacked background frames without checking shapes — one
   short-read frame made `np.max` raise (`inhomogeneous shape`).
3. A task that raised never put its result, and the consumer **counts
   results**, so it waited forever.

Fixed and verified with a pool where every third task raises the same error.
The trigger was a transient NFS read, not bad data.

## Reproducibility status
| Artifact | Where | State 2026-11-13 |
|---|---|---|
| Pipeline + training recipe | GitLab + GitHub | in sync |
| Segmenter | GitLab + GitHub | **local and GitHub 9 behind GitLab** |
| 18 deconvolution models | GitLab + GitHub | GitLab `99c2933`, GitHub `7091a78` — **GitHub behind** |
| 4 dual-ViT models | Hugging Face, **public** | published |

`Veit/M181_{binary,multiclass}` (231204 + non-isolated, pipeline default) and
`Veit/ATAIIR2604_NorthSea_{binary,multiclass}` (v5a + isolated). Cards record
the coupling, the pluteus->Rhizaria confusion and the deprecated-taxon remap.

## OPEN ISSUES

### 1. Credentials — RESOLVED
`process_pisco_profiles.config.example.json` carries the EcoTaxa and FTP
credentials. Veit confirms these are **public credentials published by
EcoTaxa**, so it stays committed. Upstream `b5030e8` additionally added the
**local** `process_pisco_profiles.config.json` to `.gitignore`, which is the
right split: example versioned, local ignored. Kept here as a note only so
nobody "fixes" the example again assuming it is secret.

### 2. Local `pisco_segmenter` is behind GitLab — see topic 09
**Still open.** (The sibling problem on `process_pisco_profiles` is now
resolved — see above.) GitLab `main` is at `8eff671`, **9 commits ahead** of
local `b805b9a`. This is
not just packaging: **background estimation changed** and there is no legacy
switch, so pulling changes the output of every future run.
**[Topic 09](09-hub-integration.md) has the analysis.** GitHub `main` is also
behind and has not received those commits.

### 3. Docs vs code — Veit is handling this
`README.md` sections 5-7 still document `--folder/--folders/--folders-file`,
which commit `02657c0` deleted. Non-standard layouts need a symlink tree.

### 4. Orchestrator does not forward `--image-ext` (still present at last check)
`orchestrate_gpus.py` has its own `--image-ext` for LPT sizing but does **not**
pass it to the pipeline, which then defaults to `.png`. A `.tif` cruise silently
produces "Profiles with images: 0" while the orchestrator still prints `OK` per
profile in ~0.2 min. Workaround: put `--image-ext` in the passthrough too.
A real run of 3 ATAIR profiles is ~15-20 min, not seconds — use that as a
sanity check.

### 5. Two GitHub mirrors lag GitLab
`pisco_segmenter` GitHub is at `b805b9a` (9 behind) and `underwater-lucyd`
GitHub at `7091a78` while GitLab has `99c2933`. GitLab is authoritative for
both; the GitHub mirrors are push-only copies and nobody is pushing them.

## New-chat prompt
> Read docs/handoff/08-infrastructure.md. I want to <sync pisco_segmenter and
> underwater-lucyd to GitHub / pull the segmenter's 9 Hub commits (read topic
> 09 first) / fix the orchestrator --image-ext forwarding>.
