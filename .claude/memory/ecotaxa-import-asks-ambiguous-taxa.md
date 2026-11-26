---
name: ecotaxa-import-asks-ambiguous-taxa
description: "Import jobs stall in state 'A' (Asking) when a taxon name matches several EcoTaxa entries; Noctiluca and Asteroidea are ambiguous — answer with the APPROVED ids 58408 / 12877"
metadata:
  node_type: memory
  type: reference
---

An EcoTaxa FileImport job can stop at **state `A` = Asking** with
`progress_msg = "Some users or taxonomic references could not be matched"` and
`job.question = {"missing_users": [], "missing_taxa": [...]}`. Progress stays 0 until answered
— it is NOT a failure and NOT a stuck job. Seen 2026-08-07 on job 250273 / project 23014.

**Cause is AMBIGUITY, not capitalization.** EcoTaxa lowercases names before matching (the same
import resolved `copepoda<multicrustacea`, `cnidaria<animalia`, `not-living` fine). The job log
says it plainly:

    Taxo 'noctiluca'  Found more than once
    Taxo 'asteroidea' Found more than once

Several taxonomy rows share the same *name* and differ only by parent lineage:

| name | id | status | lineage | nb_objects |
|---|---|---|---|---|
| Noctiluca | 14029 | **D** | Noctiluca<Noctiluca lineage | 1,347 |
| Noctiluca | **58408** | **A** | Noctiluca<Noctilucaceae | 414,727 |
| Asteroidea | 26072 | D | Asteroidea<Asteroidea<Asteroidea | 55 |
| Asteroidea | 16709 | **D** | Asteroidea<Asteroidea<Asterozoa | 17 |
| Asteroidea | **12877** | **A** | Asteroidea<Asterozoa | 4,335 |

**Why this matters:** the log shows EcoTaxa's own first-pass resolution picked **14029** and
**16709**, both **deprecated**. If it had auto-resolved rather than asked, every annotation on
those objects would later fail with the HTTP 500 "Cannot classify or validate deprecated taxa"
described in [[ecotaxa-deprecated-taxa-validate-fails]]. The question is a safety net — always
answer it with the status='A' id, never accept a 'D' one.

**How to answer — THE REPLY KEYS DIFFER FROM THE QUESTION KEYS.** The question is posed with
`missing_users` / `missing_taxa`, but the reply must use **`users`** / **`taxa`**. Mirroring the
question's key names looks obvious and is wrong: it kills the job with
`state='E'`, `progress_msg="'users'"` and, in the job log,
`KeyError: 'users'` at `/app/API_operations/imports/Import.py:95`
(`self.last_reply["users"], self.last_reply["taxa"]`). Cost one dead job (250273) on 2026-08-07.

```python
JobsApi(c).reply_job_question(job_id, body={
    "users": {},
    "taxa": {"noctiluca": 58408, "asteroidea": 12877}})     # -> state 'R' Running
```
Recovery after killing a job this way: the zip is still on the FTP server, so just re-trigger
with `ProjectsApi.import_file(pid, ImportReq(source_path="FTP/Ecotaxa_Data_to_import/GEOMAR/<zip>"))`
— no need to re-upload. Check the target project is still empty first (a failed ask-phase job
imports nothing, so it normally is).
Check a candidate first with `GET /taxon/{id}` (`status` A=approved, D=deprecated, N=morpho) or
list candidates via `GET /taxon_set/search?query=<name>`.

**Permanent fix (not yet done):** these two names come from the pipeline's deprecated-taxa remap
(`Noctiluca sp.`->`Noctiluca`, `Asteroidea larvae`->`Asteroidea`, `utils.py` ECOTAXA_TAXON_MAP)
which trades a deprecated-but-unique name for an approved-but-AMBIGUOUS one. Emit the numeric
`object_annotation_category_id` column in the TSV instead of relying on name resolution, and
imports will never ask. Related: [[ecotaxa-api-workflow-gotchas]].
