#!/usr/bin/env python3
"""Push ViT embeddings into EcoTaxa so its "similar to" filter works.

EcoTaxa's similarity search is plain pgvector: it orders by L2 distance over
``obj_cnn_features_vector.features`` and stores nothing about what produced a
vector. So our own embeddings work there, subject to three constraints:

* the column is **``vector(50)``**, so a 768-dim ViT embedding must be reduced;
* **every** object in the project needs a row -- EcoTaxa's ``find_missing()``
  refuses to search at all if any object lacks one, rather than degrading;
* there is **no REST endpoint** for features (``DeepFeatures.save()`` is
  commented out upstream), so this writes to Postgres directly, exactly as
  ``ecotaxa_ML_back`` does.

The PCA basis is fitted once and **stored**. Two projects reduced under
different bases are not comparable, and the table has no field to tell them
apart -- mixing them corrupts neighbourhoods silently. For the same reason, do
not mix embeddings from different encoders: take them all from the binary
stage, which is the only one that sees every object.

Usage:
    ecotaxa_write_features.py stage1/embeddings.npz --project-id 7 \\
        --pca /mnt/pisco/models/pisco_pca50.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TARGET_DIMS = 50

# Rows reduced and written at a time. A deep profile has ~700k objects, whose
# 768-dim embeddings are 2 GB as float32; a float64 copy of the whole matrix
# on top of that exhausted the 8 GB EcoTaxa machine and took sshd down with it.
CHUNK = 50_000

# fit_pca fits on at most this many rows; an SVD over every object of a deep
# profile would need several times the matrix in memory, and a basis needs far
# fewer samples than that to settle.
FIT_SAMPLE = 100_000


def fit_pca(matrix: np.ndarray, dims: int = TARGET_DIMS) -> dict:
    """Fit a PCA basis by SVD. Returns {mean, components}.

    numpy rather than sklearn: this needs to run in EcoTaxa's own virtualenv,
    which has numpy but no sklearn, and the maths is four lines.
    """
    if matrix.shape[0] > FIT_SAMPLE:
        rows = np.random.default_rng(0).choice(matrix.shape[0], FIT_SAMPLE, replace=False)
        matrix = matrix[np.sort(rows)]
    matrix = matrix.astype(np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return {"mean": mean, "components": vt[:dims]}


def apply_pca(matrix: np.ndarray, basis: dict) -> np.ndarray:
    """Reduce row by row in chunks, so only one chunk is ever float64."""
    reduced = np.empty((matrix.shape[0], basis["components"].shape[0]), np.float32)
    for start in range(0, matrix.shape[0], CHUNK):
        part = matrix[start:start + CHUNK].astype(np.float64)
        reduced[start:start + CHUNK] = (part - basis["mean"]) @ basis["components"].T
    return reduced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("embeddings", help="embeddings.npz from vit_classify_hdf5.py")
    ap.add_argument("--project-id", type=int, default=None,
                    help="EcoTaxa project to attach features to")
    ap.add_argument("--project-id-from", default=None,
                    help="import.json written by ecotaxa_import.py, or the "
                         "directory holding it. Lets a workflow attach features "
                         "to the project the import step just created, since the "
                         "Hub chains tasks by path rather than by value.")
    ap.add_argument("--pca", required=True,
                    help="PCA basis .npz. Fitted and written if absent, "
                         "loaded and reused if present.")
    ap.add_argument("--dsn", default=os.environ.get(
        "ECOTAXA_DSN", "host=127.0.0.1 port=5432 dbname=ecotaxa "
                       "user=postgres password=mysecretpassword"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="write anyway when some objects would have no vector. "
                         "EcoTaxa will refuse to run a similarity search on "
                         "such a project, so this is for debugging only.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import psycopg

    project_id = args.project_id
    if project_id is None:
        if not args.project_id_from:
            print("ERROR: need --project-id or --project-id-from", file=sys.stderr)
            return 1
        marker = Path(args.project_id_from)
        if marker.is_dir():
            marker = marker / "import.json"
        if not marker.exists():
            print(f"ERROR: {marker} not found", file=sys.stderr)
            return 1
        project_id = json.loads(marker.read_text())["project_id"]
        print(f"project id {project_id} read from {marker}")

    payload = np.load(args.embeddings, allow_pickle=False)
    object_ids = [str(v) for v in payload["object_ids"]]
    matrix = payload["embeddings"]
    print(f"embeddings: {matrix.shape[0]} objects x {matrix.shape[1]} dims")

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT o.orig_id, o.objid
                FROM obj_head o
                JOIN acquisitions a ON o.acquisid = a.acquisid
                JOIN samples s ON a.acq_sample_id = s.sampleid
                WHERE s.projid = %s
            """, (project_id,))
            by_orig = {row[0]: row[1] for row in cur.fetchall()}
    print(f"project {project_id}: {len(by_orig)} objects")

    matched = [(by_orig[oid], i) for i, oid in enumerate(object_ids) if oid in by_orig]
    print(f"matched: {len(matched)}")
    if not matched:
        print("ERROR: no embedding matched an object in this project. The "
              "object ids here must equal EcoTaxa's orig_id.", file=sys.stderr)
        return 1

    missing = len(by_orig) - len(matched)
    if missing:
        message = (f"{missing} of {len(by_orig)} objects would have no vector. "
                   "EcoTaxa disables similarity search for the whole project "
                   "when any object is missing features.")
        if not args.allow_partial:
            print(f"ERROR: {message} Re-run the embedding stage over every "
                  "object, or pass --allow-partial to override.", file=sys.stderr)
            return 1
        print(f"WARNING: {message}", file=sys.stderr)

    pca_path = Path(args.pca)
    if pca_path.exists():
        stored = np.load(pca_path)
        basis = {"mean": stored["mean"], "components": stored["components"]}
        if basis["components"].shape[1] != matrix.shape[1]:
            print(f"ERROR: stored basis expects {basis['components'].shape[1]} "
                  f"dims, embeddings have {matrix.shape[1]}. Different encoder?",
                  file=sys.stderr)
            return 1
        print(f"PCA basis: loaded {pca_path}")
    else:
        if matrix.shape[0] < TARGET_DIMS:
            print(f"ERROR: need at least {TARGET_DIMS} objects to fit a "
                  f"{TARGET_DIMS}-dim basis, have {matrix.shape[0]}",
                  file=sys.stderr)
            return 1
        basis = fit_pca(matrix)
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pca_path, **basis)
        print(f"PCA basis: fitted and saved to {pca_path}")

    reduced = apply_pca(matrix, basis)
    print(f"reduced: {reduced.shape[1]} dims "
          f"(range {reduced.min():.2f}..{reduced.max():.2f})")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    del matrix
    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(matched), CHUNK):
                rows = [(objid, "[" + ",".join(f"{v:.6f}" for v in reduced[i]) + "]")
                        for objid, i in matched[start:start + CHUNK]]
                cur.executemany("""
                    INSERT INTO obj_cnn_features_vector (objcnnid, features)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (objcnnid) DO UPDATE SET features = EXCLUDED.features
                """, rows)
                print(f"  {start + len(rows)} / {len(matched)}", flush=True)
            # EcoTaxa deletes a project's vectors whenever cnn_network_id
            # changes, and its settings form saves NULL as ''. A project
            # left at NULL so lost every vector the first time anyone
            # saved its settings (MSM147_003-01). '' is what the form
            # writes, so a later save changes nothing.
            cur.execute("UPDATE projects SET cnn_network_id = '' "
                        "WHERE projid = %s AND cnn_network_id IS NULL", (project_id,))
        conn.commit()
    print(f"wrote {len(matched)} feature vectors")

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*), count(v.objcnnid)
                FROM obj_head o
                JOIN acquisitions a ON o.acquisid = a.acquisid
                JOIN samples s ON a.acq_sample_id = s.sampleid
                LEFT JOIN obj_cnn_features_vector v ON v.objcnnid = o.objid
                WHERE s.projid = %s
            """, (project_id,))
            total, with_features = cur.fetchone()
    print(f"coverage: {with_features}/{total} objects have features"
          + ("  -> similarity search enabled" if with_features == total
             else "  -> STILL INCOMPLETE, search will refuse"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
