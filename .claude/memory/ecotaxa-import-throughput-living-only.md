---
name: ecotaxa-import-throughput-living-only
description: EcoTaxa import runs at ~52 objects/sec regardless of parallelism; filter to living-only before uploading
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-05T14:58:00.322Z
---

EcoTaxa's server-side FileImport queue processes at a **fixed ~52 objects/sec aggregate** (measured 2026-08-05, steady over 8 min). It does **not** parallelize: uploading a profile as many separate zip parts/jobs does not speed up total import — 41 "running" jobs still advanced at ~52/s combined. So a full M181 multinet run (~21.6M objects across 17 profiles) would take **~4+ days** to import.

**How to apply:** only import the **living** objects; the classifier labels ~98.4% of PISCO objects `not-living` (only ~1.5% living, ~332k for the 17 M181 profiles). Living-only imports in **~1.8h** instead of ~4 days. The non-living crops stay preserved in the on-disk deconv zips for later random validation sampling ([[ecotaxa-notliving-validation-sampling]]).

Key mechanics:
- The classifier's category label is exactly `not-living` (hyphen, "not-living") — **not** `non-living`. `calc_abundance_biovolume_pisco.py` filters biovolume on `!= "not-living"`.
- `process_pisco_profiles.py --export-zip` already splits output into `<profile>_Results/EcoTaxa/other/` (living taxa) and `.../non-living/` (the not-living). **Upload the `other/` `deconv_crops*.zip` for the living import.** (A label typo made this split include everything until fixed in commit f0ea603, 2026-08-05.)
- To rescue already-generated zips that contain everything: filter each profile's `deconv_crops` parts to rows where `object_annotation_category != 'not-living'`, repack only referenced crops (crops are stored flat in the zip keyed by the `img_file_name` TSV column). See [[ecotaxa-api-workflow-gotchas]].

M181 living project as of 2026-08-05: **22975** (`M181_PIScO_multinet_stations`), 17 samples named `M181_CTD-xxx`.
