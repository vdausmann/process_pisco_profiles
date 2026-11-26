---
name: ecotaxa-api-workflow-gotchas
description: "EcoTaxa API/upload gotchas — client deserialization bug, TSV sample_id/object_time rules, fixing metadata after import via /sample_set/update, how to set project managers"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-05T14:58:23.097Z
---

Working notes for the EcoTaxa API (`ecotaxa_py_client`, host in `process_pisco_profiles.config.example.json` under `ecotaxa`; login via `AuthentificationApi.login(LoginReq(...))` → set `conf.access_token`).

**Client deserialization bug:** `ProjectsApi.project_query` and `SamplesApi.search_projects` raise on `bodc_variables None`. Work around by calling the REST endpoints with **raw `urllib`** and a `Bearer` header (e.g. `GET /projects/{id}`, `GET /jobs/?for_admin=false`, `GET /users/{id}`). `SamplesApi.samples_search(project_ids='NN', id_pattern='*')` and `ObjectsApi`/`JobsApi` work fine.

**Uploading:** `upload_to_ecotaxa.py <zip> --config <cfg> --project-id NN` uploads via FTP then triggers `ProjectsApi.import_file`; success prints `job_id=...`. Zip = one `ecotaxa_*.tsv` (2 header rows: names + `[t]`/`[f]` markers) + crop PNGs stored flat, keyed by the `img_file_name` column.

**TSV rules that bite:**
- `sample_id` groups objects into EcoTaxa samples. The M181 adapter must set it to `M181_CTD-xxx` (cruise + CTD cast); a bug once produced ship-station ids like `216-1` (fixed, commit bf6f5f3).
- `object_time` must be zero-padded `HHMMSS` and `object_date` `YYYYMMDD`. Pandas re-reads them as ints and drops leading zeros → morning casts import as `Invalid Time '65300'`. The exporter zero-pads (commit f9de900).

**Emptying vs deleting a project:** `ObjectsApi.get_object_set(pid, ProjectFilters(), fields='obj.objid')` then `erase_object_set` in ~20k batches removes objects but **leaves empty sample shells**. For a truly clean project, `ProjectsApi.erase_project(pid)` + `create_project(CreateProjectReq(title=..., instrument='Other camera', visible=False))`. Instrument **must** be `Other camera` — `PISCO` isn't in EcoTaxa's instrument table (FK violation).

**Fix metadata after an import (no re-import):** free columns are physical `tNN`/`nNN` columns; the names come from `GET /projects/{id}` → `sample_free_cols` / `obj_free_cols` (e.g. `{"cruise":"t01","dship_id":"t02","instrument":"t03","ctd_cast":"t04"}`). Then `POST /sample_set/update` with `{"target_ids":[<sampleid>],"updates":[{"ucol":"t03","uval":"PISCO2"}]}` returns the number updated. Sibling endpoints: `/object_set/update`, `/acquisition_set/update`, `/project_set/update`. Find the sample with `GET /samples/search?project_ids=NN&id_pattern=` — **`id_pattern` is required** (422 without it) and may be empty; that listing returns `free_columns: {}`, so read the values with `GET /sample/{sampleid}`. Sample-level fields are one row for the whole profile, so this fixes all objects at once (done 2026-11-16: project 24269 instrument PISCO3→PISCO2). What it cannot fix is anything baked into `object_id`/`img_file_name` — those come from the frame names and need a re-import.

**Cancel imports:** `JobsApi.erase_job(id)` cancels/removes FileImport jobs. Filter jobs by `type=='FileImport'` and `id >= <first-of-your-batch>` to ignore unrelated/old jobs.

**Add a project manager:** find user via `UsersApi.search_user(by_name='%Surname%')` → id; then raw `GET /projects/{id}`, append `{id,email,name,organisation}` to the `managers` list, `PUT /projects/{id}` with the whole modified body. (Done 2026-08-05: Fanny Brodbek, user 3402, added to 22975/22988/22989.)

Related: [[ecotaxa-import-throughput-living-only]], [[ecotaxa-notliving-validation-sampling]].
