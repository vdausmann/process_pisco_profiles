---
name: ecotaxa-deprecated-taxa-validate-fails
description: "Validating objects to a deprecated EcoTaxa taxon 500s (\"Changes NOT saved\"); remap Asteroidea larvae→Asteroidea and Noctiluca sp.→Noctiluca at upload"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-06T11:34:52.076Z
---

If an object's predicted `object_annotation_category` maps to a **deprecated** EcoTaxa taxon (taxonomy `status='D'`), the UI validate/classify fails with **"Changes NOT saved (see logs)"** and the backend raises HTTP 500 `AssertionError: Cannot classify or validate deprecated taxa` (`/app/BO/ObjectSet.py:979`). Confirmed 2026-08-06 on project 23003.

**Why the logs looked broken (same root cause):** classify/validate is a **synchronous** API call — a failure does **not** create a Job, so there is no per-job log in the UI/API; the referenced "logs" are EcoTaxa's server-internal logs, admin-only. And because every save 500'd, nothing committed, so annotation history / revert-to-history came back **empty** — a symptom, not a second bug. **How to diagnose:** call the endpoint directly (e.g. `ObjectsApi.classify_object_set(ClassifyReq(target_ids=[oid], classifications=[taxon_id], wanted_qualification="V"))`) — the HTTP 500 body carries the full backend traceback the web UI hides. Check a taxon's status via `GET /taxon/{id}` (`status`: A=approved, D=deprecated, N=morpho; `rename_to` is often None so EcoTaxa won't auto-redirect).

**Why:** the PISCO classifier emits some category names that EcoTaxa has since deprecated. Known deprecated → accepted mappings:
- `Asteroidea larvae` (taxon 94682, status D) → **`Asteroidea`** (12877)
- `Noctiluca sp.` → **`Noctiluca`** (58408)

**How to apply:** remap these category names in `object_annotation_category` **at export/upload time** (in the pipeline export, i.e. the `df_ET` / `create_ecotaxa_zips` path, [[ecotaxa-api-workflow-gotchas]]) so future imports land on accepted taxa and are validatable. To fix already-uploaded projects, `reclassify_object_set` the affected objects from the deprecated id to the accepted one (2026-08-06: 2,241 objects on 94682 in each of 22975/22988). Whenever a new cruise/model is uploaded, re-check taxon statuses — the deprecated set can grow.
