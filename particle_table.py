"""The particle table: every object of a profile, measured, in one file.

The EcoTaxa archive is a sample of the profile. It carries the images people
annotate, and to keep those manageable the export draws only a subsample of
the not-living objects (``--subsample-max``). Abundances cannot be computed
from it. The particle table is the complete record next to it: one row per
object the segmenter stored, with the same measurements as the EcoTaxa TSV,
the classifier's verdict, and whether the object went to EcoTaxa — but no
images, so it stays small.

It also lists every frame that was processed, with or without objects, which
is what a sampled volume has to be counted over: frames the segmenter skipped
were not looked at, and frames it looked at and found empty still count.

Layout (HDF5, one 1-D dataset per column, so a reader needs h5py only)::

    /                    @format @format_version @profile @workflow_id ...
    /objects/<column>    (N,)  one per object, in frame order
    /frames/<column>     (F,)  one per processed frame

``object_id`` is the same string the EcoTaxa TSV carries (``<frame>_<n>``),
which EcoTaxa stores as the object's ``orig_id``: that is the key that joins
this table to live annotations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

FORMAT = "pisco-particle-table"
FORMAT_VERSION = 1

_STR = h5py.string_dtype("utf-8")


def table_name(profile: str) -> str:
    return f"particles_{profile}.h5"


class ParticleTableWriter:
    """Appends columns frame by frame; the file appears complete or not at all.

    Columns are created on first sight from the dtype of the arrays given, so
    the writer does not need its own copy of the exporter's column list. Every
    append must carry the same columns.
    """

    def __init__(self, path: Path, attrs: dict):
        self.path = Path(path)
        self.tmp = self.path.with_name(f".{self.path.name}.tmp")
        self.file = h5py.File(self.tmp, "w")
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self.file.attrs[key] = value
        self.file.attrs["format"] = FORMAT
        self.file.attrs["format_version"] = FORMAT_VERSION
        self.file.attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.counts = {"objects": 0, "frames": 0}
        self.buffers: dict = {}
        self.pending: dict = {}

    # Rows are buffered and written a chunk at a time: resizing and writing
    # every compressed column once per frame re-compressed the same chunk
    # hundreds of times and made writing ten times slower than measuring.
    FLUSH_ROWS = 65536

    def _append(self, group_name: str, columns: dict):
        n = None
        buf = self.buffers.setdefault(group_name, {})
        for name, values in columns.items():
            # Pass typed arrays: an empty list would otherwise become float.
            arr = np.asarray(values)
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"{group_name}/{name}: {len(arr)} values, expected {n}")
            buf.setdefault(name, []).append(arr)
        n = n or 0
        self.counts[group_name] += n
        self.pending[group_name] = self.pending.get(group_name, 0) + n
        if self.pending[group_name] >= self.FLUSH_ROWS:
            self._flush(group_name)

    def _flush(self, group_name: str):
        buf = self.buffers.get(group_name) or {}
        group = self.file.require_group(group_name)
        for name, parts in buf.items():
            arr = np.concatenate(parts) if parts else np.asarray([])
            dtype = _STR if arr.dtype.kind in "OUS" else arr.dtype
            if name not in group:
                group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                     chunks=(self.FLUSH_ROWS,), compression="gzip",
                                     compression_opts=4, shuffle=dtype != _STR)
            if not len(arr):
                continue
            ds = group[name]
            start = ds.shape[0]
            ds.resize((start + len(arr),))
            ds[start:] = arr.astype(str).astype(object) if dtype == _STR else arr
        self.buffers[group_name] = {name: [] for name in buf}
        self.pending[group_name] = 0

    def add_objects(self, columns: dict):
        self._append("objects", columns)

    def add_frames(self, columns: dict):
        self._append("frames", columns)

    def close(self, extra_attrs: dict | None = None) -> Path:
        for group_name in list(self.buffers):
            self._flush(group_name)
        for key, value in (extra_attrs or {}).items():
            if value is not None:
                self.file.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        self.file.attrs["n_objects"] = self.counts["objects"]
        self.file.attrs["n_frames"] = self.counts["frames"]
        self.file.close()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self):
        try:
            self.file.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def read_particle_table(path: Path, columns: list[str] | None = None) -> dict:
    """{"attrs": {...}, "objects": {col: array}, "frames": {col: array}}.

    Strings come back as Python str. ``columns`` limits which object columns
    are read; frames are always read whole (they are small).
    """
    out = {"attrs": {}, "objects": {}, "frames": {}}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        for key, value in f.attrs.items():
            out["attrs"][key] = value.item() if hasattr(value, "item") else value
        for group_name in ("objects", "frames"):
            group = f.get(group_name)
            if group is None:
                continue
            wanted = columns if (group_name == "objects" and columns) else list(group)
            for name in wanted:
                if name not in group:
                    continue
                ds = group[name]
                if h5py.check_string_dtype(ds.dtype):
                    out[group_name][name] = ds.asstr()[()]
                else:
                    out[group_name][name] = ds[()]
    return out
