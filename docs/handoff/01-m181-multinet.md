# 01 — M181 multinet stations -> EcoTaxa

**State: closed.** 17 CTD stations coincident with multinet hauls processed and
uploaded; a not-living validation set was drawn and uploaded.

## What was done
* Identified the 17 CTD profiles coinciding with multinet hauls from the M181
  cruise report, processed them on both 7900XTX GPUs via `orchestrate_gpus.py`.
* Uploaded the **living fraction only** to EcoTaxa project **22975**
  (332,644 objects, 17 samples named `M181_CTD-xxx`).
* Drew a **10,000-object stratified random sample of the not-living fraction**
  -> project **22989**, for estimating the missed-living rate.

## Why living-only
EcoTaxa's import queue runs at a **fixed ~52 objects/s regardless of
parallelism** (measured over 8 min; 41 concurrent jobs advanced at the same
aggregate rate). The full 21.6M objects would have taken ~4 days; the living
1.5% took ~1.8 h. The not-living crops stay on disk for later sampling.

## Bugs found and fixed (all pushed)
| Commit | Problem |
|---|---|
| `bf6f5f3` | sample_id was the ship-station id (`216-1`) not `M181_CTD-058` |
| `9227351` | multi-part export copied millions of crops -> inode exhaustion |
| `f9de900` | `object_time` lost its leading zero -> morning casts rejected |
| `f0ea603` | living/non-living export split compared `'non-living'` while the classifier emits `'not-living'`, so **everything** landed in the living export |
| `405eeb2` | `Asteroidea larvae` / `Noctiluca sp.` are deprecated in EcoTaxa -> remapped at export |

## Traps worth remembering
* **Inode exhaustion, not disk space.** One profile can write ~38M files.
  `--cleanup-intermediate` is mandatory.
* **GPU deconvolution is not reproducible**, so the recrop tool cannot
  regenerate crops matching an existing TSV (coverage came out 89.7%/99.3%).
  Re-run the full pipeline with `--export-zip` instead.
* Emptying a project leaves **empty sample shells**; `erase_project` +
  recreate is the clean path.
* Imports stall in state `A` on ambiguous taxa — reply keys are
  `users`/`taxa`, **not** the question's `missing_*`.

## Open
Nothing required. 22989 is uploaded but **not validated** — validating it would
give the missed-living rate for M181.

## New-chat prompt
> M181 multinet stations are in EcoTaxa project 22975 (living only) with a 10k
> not-living validation sample in 22989. Read docs/handoff/01-m181-multinet.md.
> I want to validate 22989 and compute the missed-living rate.
