---
name: ecotaxa-notliving-validation-sampling
description: "How to draw a statistically sound random validation set of not-living predictions, and the sizing logic"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c3619e1-ae89-44dc-90da-391e3c64f238
  modified: 2026-08-05T14:58:52.071Z
---

To validate the classifier's `not-living` predictions (the important error is *living wrongly dumped as not-living* = lost biomass), draw a random subset and re-classify it manually in a **separate** EcoTaxa project (keep the original `M181_CTD-xxx` sample_ids for provenance — importing into the living project 22975 would merge them into the living samples).

**Sizing (95% confidence, population ~21M so finite-population correction is irrelevant):**
- Estimate a proportion (worst-case p=0.5): ±5% → n≈385; ±3% → ≈1,068; ±2% → ≈2,401; ±1% → ≈9,604.
- Bound a *low* miss rate (rule of three, 0 events found): n=300 → <1%; n=1,000 → <0.3%; n=3,000 → <0.1%.
- **~1,000 is the sweet spot** for pure validation. Go **10,000** if you also want to characterize which taxa slip through / reuse labels; **20,000** only if reusing as training/fine-tuning data.

**Done 2026-08-05:** built a 10,000-object set, **stratified proportional to each profile's not-living count** across the 17 profiles (reservoir sampling, seed 42), packed into one EcoTaxa zip → project **22989** (`M181_PIScO_notliving_validation`). A separate 20k subset lives in project **22988**. Sampler script pattern: stream each profile's `deconv_crops` part zips, reservoir-sample `not-living` rows, extract only the sampled crops (flat-keyed by `img_file_name`), write one combined zip. See [[ecotaxa-api-workflow-gotchas]] and [[ecotaxa-import-throughput-living-only]].
