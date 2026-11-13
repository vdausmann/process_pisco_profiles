# 08 — Repos, reproducibility, known issues

## Commits made this session

**`process_pisco_profiles`** (GitLab + GitHub, both in sync at `7492061`)
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
| `7492061` | reversible scale bar |

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
| Artifact | Where |
|---|---|
| Pipeline + training recipe | GitLab + GitHub |
| Segmenter | GitLab + GitHub |
| 18 deconvolution models | GitLab + GitHub |
| 4 dual-ViT models | Hugging Face, **public** |

`Veit/M181_{binary,multiclass}` (231204 + non-isolated, pipeline default) and
`Veit/ATAIIR2604_NorthSea_{binary,multiclass}` (v5a + isolated). Cards record
the coupling, the pluteus->Rhizaria confusion and the deprecated-taxon remap.

## OPEN ISSUES

### 1. Credentials — RESOLVED (committed)
`process_pisco_profiles.config.example.json` carries the EcoTaxa and FTP
credentials. Veit confirms these are **public credentials published by
EcoTaxa**, so it is committed as of 2026-11-13. Kept here as a note only so
nobody "fixes" it again assuming they are secret.

### 2. Local `pisco_segmenter` is behind GitLab — see topic 09
GitLab `main` is at `8eff671`, **9 commits ahead** of local `b805b9a`. This is
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

## New-chat prompt
> Read docs/handoff/08-infrastructure.md. I want to <fix the credentials in the
> config / sync pisco_segmenter to GitHub / fix the orchestrator --image-ext
> forwarding>.
