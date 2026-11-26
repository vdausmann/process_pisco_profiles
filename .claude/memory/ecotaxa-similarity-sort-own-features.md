---
name: ecotaxa-similarity-sort-own-features
description: "EcoTaxa similarity sort needs server-side CNN features; own vectors can't be uploaded (not via FTP either); restore ViT predictions afterwards via classify_auto_multiple"
metadata:
  node_type: memory
  type: reference
  originSessionId: 5d4ed56f-4523-4695-82b0-d55466b9ac36
  modified: 2026-11-25T09:41:59.935Z
---

Checked against the live vlfr.fr API spec on 2026-11-25:
- Similarity sort (`POST /object_set/{pid}/similarity_search/{objid}`) runs on deep features that the server computes with the project's "CNN network". No endpoint accepts feature vectors. FTP is only a drop area for import zips, and TSV `object_*` columns become free columns, not deep features.
- `GET /instruments/cnn_networks` has **no PIScO network**. The list covers FlowCam, IFCB, ISIIS, LISST-Holo, Loki, PlanktoScope, UVP5HD/SD, UVP6, ZooCam and Zooscan. To get one, ask the EcoTaxa team; they added lisst_holo_2026-06, so they do add networks.
- An EcoTaxa prediction overwrites status-P objects. Use `ecotaxa_restore_predictions.py`: `snapshot` BEFORE predicting, then `predict` (runs the job and waits), then `restore`, which writes back with `classify_auto_multiple`. Imported P objects have classif_score 1.0; the ViT probability is the `prob_1` free column, and restore writes that.
- For 23286 (2026-11-25, CNN isiis): snapshot at /media/veit/T710_data/pisco_processed/SO308_v5a_madagascar/ecotaxa_23286_vit_snapshot_20261125.tsv. Object histories were empty before the predict, so `revert_to_history` could not have restored anything.
- For 24331 (ATAIR subset, 2026-11-26): the score column is `prediction_score`, so pass `--score-field fre.prediction_score`. Snapshot at /media/veit/T710_data/pisco_processed/ecotaxa_24331_vit_snapshot_20261126.tsv. Check `obj_free_cols` for the score column on each project.

Related: [[ecotaxa-api-workflow-gotchas]], [[pisco-two-routes-hdf5-hub]]
