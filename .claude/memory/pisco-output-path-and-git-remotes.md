---
name: pisco-output-path-and-git-remotes
description: Processed-data output path (T710) and how to push code to both GitHub (SSH) and GitLab
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-05T14:59:06.820Z
---

**Output path:** processed PISCO data for M181 goes to `/media/veit/T710_data/pisco_processed/<CRUISE>` (e.g. `.../M181`). Note it's **T710**, not T705 — `/media/veit/T705_data` does not exist (a path typo bit us on 2026-07-30). Each profile dir holds `<profile>_Results/` with `Data/`, `EcoTaxa/`, and (unless cleaned, [[pisco-intermediate-crops-exhaust-inodes]]) the crop dirs.

**Git remotes — push code changes to BOTH:**
- `gitlab` → `git@git.geomar.de:team-optics/plankton/process_pisco_profiles.git` — `git push gitlab main` works (SSH).
- `origin` → `https://github.com/vdausmann/process_pisco_profiles.git` — the **HTTPS** origin has no credentials in this environment, so `git push origin main` fails with "could not read Username". Push GitHub via the **SSH URL directly**: `git push git@github.com:vdausmann/process_pisco_profiles.git main`.

The user's standing instruction is to commit **and** push every code change to both remotes.
