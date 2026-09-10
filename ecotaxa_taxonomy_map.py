#!/usr/bin/env python3
"""Translate ViT class labels into EcoTaxa taxon ids.

The models we have were trained on their own label vocabulary, which is not
EcoTaxa's. Until a model is trained against EcoTaxa's taxonomy directly (see
``docs/`` handover note), predictions have to be translated, and the
translation has to be **explicit and versioned** rather than resolved at
runtime. Three real cases from the current taxonomy show why guessing is
unsafe:

* ``Ctenophora`` matches two active taxa -- ``Ctenophora<Animalia`` (comb
  jellies) and ``Ctenophora<Fragilariaceae`` (**a diatom**). Taking the lowest
  id picks the diatom.
* ``Copepoda`` matches a phylogenetic Class and a morphological category.
* ``living`` matches only ``living<t011`` and ``living<Chydoridae``. Neither is
  a sensible target: it is the binary stage's *gate*, not a classification, so
  it is marked ``skip`` and never written to EcoTaxa.

So this builds a candidate mapping for review, and validates one that exists.
Unresolved or ambiguous labels are left null with their candidates recorded;
loading a map that still has them raises rather than dropping objects.

    # propose a mapping for a model's labels
    ecotaxa_taxonomy_map.py build --model Veit/diplo-vit-plankton \\
        -o ecotaxa_taxonomy_map.json

    # re-check an existing one against the live taxonomy
    ecotaxa_taxonomy_map.py check ecotaxa_taxonomy_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_DSN = os.environ.get(
    "ECOTAXA_DSN",
    "host=127.0.0.1 port=5432 dbname=ecotaxa user=postgres password=mysecretpassword")

# Labels that are pipeline machinery rather than taxa. The binary stage's
# "living" decides which crops reach the taxon stage; it is not an annotation.
GATE_LABELS = {"living"}

# Shapes the label vocabulary uses that the taxonomy does not.
def _variants(label: str) -> list[str]:
    """Spellings to try, most specific first."""
    out = [label]
    if "_" in label:                      # Ctenophora_Metazoa -> Ctenophora<Metazoa
        out.append(label.replace("_", "<"))
    # Strip the ancestor from every "<" form produced so far, not just from the
    # original: the label's ancestor is often not the one EcoTaxa records
    # (Ctenophora_Metazoa is Ctenophora<Animalia there), so the bare name has to
    # be tried too. It may well be ambiguous -- that is for review, not a guess.
    for form in list(out):
        if "<" in form:
            out.append(form.split("<", 1)[0])
    if label.endswith(" sp."):            # Noctiluca sp. -> Noctiluca
        out.append(label[:-4].strip())
    for suffix in (" larvae", " larva"):  # Asteroidea larvae -> Asteroidea
        if label.endswith(suffix):
            out.append(label[: -len(suffix)].strip())
    seen, uniq = set(), []
    for v in out:
        if v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq


def candidates_for(cur, label: str) -> list[dict]:
    """Active taxa matching any spelling of `label`, on name or display name."""
    found: dict[int, dict] = {}
    for variant in _variants(label):
        cur.execute("""
            SELECT id, name, display_name, rank, taxotype
            FROM taxonomy
            WHERE taxostatus = 'A'
              AND (lower(name) = lower(%s) OR lower(display_name) = lower(%s))
            ORDER BY id
        """, (variant, variant))
        for row in cur.fetchall():
            found.setdefault(row[0], {
                "id": row[0], "name": row[1], "display_name": row[2],
                "rank": row[3], "taxotype": row[4], "matched": variant,
            })
        if found:
            # A more specific spelling matched; do not dilute it with looser ones.
            break
    # Prefer phylogenetic entries: a model label naming a taxon means the taxon,
    # not a morphological look-alike category.
    return sorted(found.values(), key=lambda c: (c["taxotype"] != "P", c["id"]))


def labels_from_model(model_path: str) -> list[str]:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    return [config.id2label[i] for i in sorted(config.id2label)]


def build(args) -> int:
    import psycopg

    labels = (labels_from_model(args.model) if args.model
              else [s.strip() for s in args.labels.split(",") if s.strip()])
    print(f"{len(labels)} labels to map")

    entries: dict[str, dict] = {}
    n_auto = n_review = n_none = 0
    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        for label in labels:
            if label.lower() in GATE_LABELS:
                entries[label] = {"id": None, "skip": True,
                                  "note": "pipeline gate, not a taxon"}
                print(f"  {label:24s} SKIP  (gate, not written to EcoTaxa)")
                continue
            found = candidates_for(cur, label)
            if len(found) == 1:
                best = found[0]
                entries[label] = {"id": best["id"], "display_name": best["display_name"]}
                n_auto += 1
                print(f"  {label:24s} ->    {best['id']:>7}  {best['display_name']}")
            elif len(found) > 1:
                entries[label] = {
                    "id": None,
                    "note": "AMBIGUOUS - choose one id from candidates",
                    "candidates": [{"id": c["id"], "display_name": c["display_name"],
                                    "taxotype": c["taxotype"]} for c in found],
                }
                n_review += 1
                print(f"  {label:24s} REVIEW ({len(found)} candidates):")
                for c in found:
                    print(f"      {c['id']:>7}  [{c['taxotype']}] {c['display_name']}")
            else:
                entries[label] = {"id": None, "note": "NOT FOUND - fill in by hand"}
                n_none += 1
                print(f"  {label:24s} NOT FOUND")

    out = {
        "_comment": ("Model label -> EcoTaxa taxon id. Entries with a null id must be "
                     "resolved by hand before use; loading this map will refuse them. "
                     "Rebuild with ecotaxa_taxonomy_map.py build, re-check with check."),
        "source_model": args.model or "(explicit label list)",
        "labels": entries,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{n_auto} resolved, {n_review} ambiguous, {n_none} not found "
          f"-> {args.output}")
    if n_review or n_none:
        print("Fill in the null ids before using this map.")
    return 0


def load_map(path: str | Path) -> dict[str, int]:
    """label -> taxon id, for labels that map to a taxon.

    Raises on a map that still has unresolved entries: a prediction run that
    silently dropped a class would be worse than one that refused to start.
    """
    data = json.loads(Path(path).read_text())
    entries = data.get("labels", {})
    unresolved = [label for label, e in entries.items()
                  if e.get("id") is None and not e.get("skip")]
    if unresolved:
        raise ValueError(
            f"{path}: {len(unresolved)} label(s) still unmapped: "
            f"{', '.join(sorted(unresolved))}. Resolve them before predicting.")
    return {label: e["id"] for label, e in entries.items() if e.get("id") is not None}


def check(args) -> int:
    import psycopg

    data = json.loads(Path(args.mapping).read_text())
    entries = data.get("labels", {})
    problems = 0
    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur:
        for label, entry in sorted(entries.items()):
            if entry.get("skip"):
                print(f"  {label:24s} skipped ({entry.get('note','')})")
                continue
            taxon_id = entry.get("id")
            if taxon_id is None:
                print(f"  {label:24s} UNMAPPED  {entry.get('note','')}")
                problems += 1
                continue
            cur.execute("SELECT display_name, taxostatus, rename_to FROM taxonomy "
                        "WHERE id = %s", (taxon_id,))
            row = cur.fetchone()
            if row is None:
                print(f"  {label:24s} MISSING   id {taxon_id} is not in the taxonomy")
                problems += 1
                continue
            display, status, rename_to = row
            flags = []
            if status != "A":
                flags.append(f"status={status}")
            if rename_to:
                flags.append(f"renamed to {rename_to}")
            if entry.get("display_name") and entry["display_name"] != display:
                flags.append(f"display changed: {entry['display_name']!r} -> {display!r}")
            if flags:
                print(f"  {label:24s} WARN      {taxon_id} {display}  [{'; '.join(flags)}]")
                problems += 1
            else:
                print(f"  {label:24s} ok        {taxon_id} {display}")
    print(f"\n{len(entries)} entries, {problems} needing attention")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="propose a mapping for a model's labels")
    b.add_argument("--model", default=None, help="model directory or HF repo id")
    b.add_argument("--labels", default=None, help="comma-separated labels instead")
    b.add_argument("-o", "--output", required=True)
    b.set_defaults(func=build)

    c = sub.add_parser("check", help="validate a mapping against the live taxonomy")
    c.add_argument("mapping")
    c.set_defaults(func=check)

    args = ap.parse_args()
    if args.command == "build" and not (args.model or args.labels):
        print("ERROR: build needs --model or --labels", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
