# Session handoff — index

One long session (2026-07-30 .. 2026-11-13) covering several independent
workstreams. Each file below is self-contained: what was done, where things
live, what is still open, and a prompt to open a new chat with.

| # | Topic | State |
|---|---|---|
| [01](01-m181-multinet.md) | M181 multinet stations -> EcoTaxa | closed |
| [02](02-deconvolution-models.md) | LUCYD deconvolution models + crop isolation | decision pending |
| [03](03-atair-validation.md) | ATAIR-BSH validation + not-living re-screening | closed, awaiting validation |
| [04](04-dualvit-training.md) | Dual-ViT retraining + model publishing | closed |
| [05](05-so308-campaign.md) | SO308 processing campaign | **open — main thread** |
| [06](06-scale-bars.md) | Scale bars on annotator crops | **open — half-finished** |
| [07](07-benchmark-extraction.md) | Benchmark annotation extraction | closed |
| [08](08-infrastructure.md) | Repos, reproducibility, known issues | **open items** |
| [09](09-hub-integration.md) | PIScO Hub integration — background estimation changed | **open — affects all future runs** |

## The one thing to carry into every topic

Deconvolution model + crop isolation + classifier are **one coupled unit**.
Since 2026-09, **background estimation** is a fourth member of that unit (see
[09](09-hub-integration.md)). Changing any of them without retraining the ViT is an out-of-domain shift: a
controlled test measured an 83-93% collapse in the living rate at unchanged
confidence. Every abundance number in this session carries that caveat unless
the classifier was trained on matching crops.

## EcoTaxa projects created or touched

| id | title | note |
|---|---|---|
| 22975 | M181_PIScO_multinet_stations | 17 stations, living only |
| 22988 | M181 - 20k subset | |
| 22989 | M181_PIScO_notliving_validation | 10k not-living sample |
| 23003 | M181 - 2k subset | non-isolated crops |
| 23014 | ATAIR_PIScO_v5d_lucyd | |
| 23021 | ATAIR-BSH_PISCO_v4_lucyd | |
| 23022 | ATAIR-BSH_PISCO_v5a_lucyd | main ATAIR project, isolated crops |
| 23052 | ATAIR v5a subset (20,775) | not-living re-screening |
| 23196 | ATAIR subset (2,500) | re-screening confirmation |
| 23210 | ATAIR_v5a_NorthSeaViT_living_check | precision/recall sample, unvalidated |
| 23263 | PIScO_benchmark_v2 - PETRA BILIC validations | |
| 23286 | SO308_Madagascar_v5a_validation | scale bars, unvalidated |
