# 07 — Benchmark annotation extraction (Petra Bilic)

**State: closed.**

## What was done
Extracted every annotation by **PETRA BILIC (user 4449)** from
`PIScO_benchmark_v2` (project **20649**, 406,047 objects) into a new **private**
project **23263** — `PIScO_benchmark_v2 - PETRA BILIC validations` — with
**Tobias Strickmann (2470)** as co-manager.

**7,902 objects**: 7,828 validated + 74 dubious, across 6 samples. Top
categories: `unidentifiable` 4,282 · `streak` 1,567 · `copepoda<Copepoda` 1,010
· `Appendicularia` 193 · `Copepoda<Multicrustacea` 176 · `tentacle<Cnidaria` 114.

## How
EcoTaxa **Subset** job with `filters={"filt_last_annot": "4449"}`,
`group_type="C"`, `limit_type="P"`, `limit_value=100` (100% per category).
Note `SubsetReq.filters` takes a **plain dict**, not a `ProjectFilters` object.

## Two deliberate choices
* **This is a copy — the source is untouched.** 20649 still has all 406,047
  objects with her 7,902 annotations in place. "Pull out" could have meant
  *move*, but deleting 7,900 validations from a benchmark project is
  destructive and was not done without an explicit instruction.
* Her 74 **dubious** were included alongside the 7,828 validated — a supervisor
  usually wants the cases she flagged as uncertain.

## Known gap
Selection used `filt_last_annot` (objects where she is the **current**
annotator). A further **241 objects she annotated have since been re-annotated
by someone else** (`filt_annot` returns 8,143 vs 7,902). They were excluded
because in the copy they would carry someone else's label, not hers — but if
the point is to review everything she touched, **those 241 disagreements are
arguably the most interesting**. They were never added.

## New-chat prompt
> Read docs/handoff/07-benchmark-extraction.md. Add the 241 objects Petra
> annotated that were later re-annotated by others into project 23263 as a
> separate, distinguishable batch.
