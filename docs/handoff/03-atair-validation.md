# 03 — ATAIR-BSH validation + not-living re-screening

**State: processing closed; validation partly done.** Full publication-ready
writeup already exists at
`~/.claude/projects/.../memory/atair-bsh-writeup.md`.

## The dataset
ATAIR-BSH, April 2026, North Sea. 3 profiles
(`20260422-2051`, `20260423-1100`, `20260423-1705`), 2,464 frames.
Main project **23022** = v5a deconvolution + isolated crops:
7,255 living + a 20,000 random not-living sample.

Note: ATAIR has a non-standard directory layout, so cruise mode needs a
**symlink tree** (custom-folders mode was deleted in commit `02657c0`).

## The headline correction
The classifier's second-largest living class — **2,354 objects called
`Rhizaria`** — turned out on validation to be almost entirely **pluteus
larvae (Echinoidea)**, 2,212 of them. The old ViT has no pluteus class.

## Re-screening the not-living fraction for missed living
Iterative EcoTaxa **deep-feature (SCN/GPU) prediction**, learning from the
growing validated set:

1. Random 20,000 not-living -> project **23052** (20,775 incl. 775 dubious).
2. First prediction had no not-living class in its label set, so everything was
   forced into living classes (18,315 "Copepoda"). A catch-all **`Unknowns`**
   category was created and validated to give the model a fall-back.
3. Iterating collapsed the mass into Unknowns: living candidates
   **17,832 -> 737 -> ~6**.
4. Confirmation: fresh random 2,500 -> project **23196**; predicted 2,494
   Unknowns, validated 2,492 Unknowns + 5 pluteus.

**Conclusion: the living-in-not-living pool is essentially exhausted.**

## Per-pass confusion matrices (validated objects only)
| Pass | n | Agreement |
|---|--:|--:|
| deep pass 1 | 3,473 | 17.2% |
| deep pass 2 | 3,473 | 17.1% |
| deep pass 3 | 1,362 | 36.3% |
| deep pass 3 on the confirmation subset | 2,497 | **99.8%** |

Passes 1-2 look catastrophic for a **structural** reason: they had no
`Unknowns` class, yet 2,588 validated objects were Unknowns — so recall on the
largest true class was necessarily 0%.

## Interactive figure
Published artifact (stacked bars per stage, a measured Sankey of all 20,775
object trails, and the confusion matrices):
<https://claude.ai/code/artifact/75968fb1-9d58-4d9c-ac43-8561f36746d6>

The Sankey is built from **per-object EcoTaxa classification history**
(`/object/{id}/history`, all 20,775 fetched, ~55/s), not inferred from
aggregates.

## Validator caveats (stated, must survive into any publication)
* `Unknowns` is a **hoover** — "very likely correct, but it is a hoover for
  anything of any shape".
* Late-stage validation was fast batch confirmation ("flying over batches of
  100, mostly selecting everything") — possible confirmation bias.
* EcoTaxa's **similar-to** search was used when something classifiable turned
  up, which raises confidence that most were caught.
* The **dubious flag means different things** in 23022 ("unsure it matches the
  predicted class") vs the re-screening projects ("certainly zooplankton, taxon
  unclear"). **Do not pool dubious counts across projects.**

## Open
* 23210 (`ATAIR_v5a_NorthSeaViT_living_check`, 1,000 objects) is **uploaded but
  unvalidated** — see topic 04, it measures the retrained model.
* Sections 01-03 of the artifact use snapshot aggregates; section 04-05 use the
  authoritative history data. Rebuilding the earlier sections from history was
  offered and not done.

## New-chat prompt
> Read docs/handoff/03-atair-validation.md. ATAIR processing and re-screening
> are done. I want to <validate 23210 / rebuild the figures / write this up>.
