---
name: ecotaxa-multiple-categories
description: "Which \"multiple\" categories exist in EcoTaxa (checked 2026-11-18) — no taxon-combination categories like copepoda+appendicularia"
metadata:
  node_type: memory
  type: reference
  originSessionId: 24ddbef3-eabf-4edd-9b8f-144f0b3c1485
  modified: 2026-11-18T13:39:24.831Z
---

Active (status A) "multiple" morpho categories in EcoTaxa as of 2026-11-18:
multiple<Copepoda 85044, multiple<other 85079 (under other<Biota), multiple species 96019 (flat, under Biota),
multiple<plastic 85075, multiple<rectangle 85086, multiple<Diplostraca 95204, multiple<Acantharia 92993,
multiple<Chaetoceros 93584, multiple<Hemiaulus 93582.
Combination names ("multiple+copepoda+detritus" etc.) exist only as status N (unvalidated user proposals).
No multiple<Appendicularia and no combination categories.
Public lookup without auth: GET https://ecotaxa.obs-vlfr.fr/api/taxon_set/search?query=... and /api/taxon/{id}
(not /api/taxa/{id}). ATAIR ViT trainset excludes "multiple species" (see [[atair-bsh-writeup]]).
