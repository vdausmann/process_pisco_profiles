#!/usr/bin/env python3
"""Compare PIScO and UVP5 casts made on the same CTD rosette.

Pulls both EcoTaxa projects, works out the volume each instrument imaged per
depth bin, and writes abundance (ind/L) and biovolume (mm^3/L) per cast and
taxon group, plus comparison figures.

Sampled volume
--------------
PIScO  frames per bin (from the raw PNG names on the filer, which carry the
       pressure) x volume per frame. The segmenter resizes every frame to
       2560 px and masks a circle of mask_radius (1500) around the centre; that
       circle overhangs the 1280 px half-width, so the imaged area is the
       circle-square intersection (87% of pi r^2), times the depth of field.
UVP5   EcoTaxa only holds the vignettes, not the per-depth image count that
       EcoPart has. Both cameras were on the same rosette and run at a constant
       frame rate (PIScO 4 Hz, UVP ~19 Hz), so the UVP's images per bin follow
       the PIScO timeline: n_uvp(bin) = N_uvp * frames_pisco(bin) / frames_pisco.
       N_uvp = last_img - first_img + 1 from the UVP process metadata; vol/img
       from acq_volimage. A cross-check at a fixed 19 Hz is written alongside.
       That estimate is only right over a whole cast (within 5%): EcoPart keeps
       descending images only, so near the surface (soak, ship heave) the
       timeline proxy is 2-3x too high. Pass --ecopart (the EcoPart "reduced"
       export zip or its unpacked folder) to use EcoPart's sampled volume per
       5 m bin -- always do this when it is available. It also enables the
       all-particle check against the UVP's own LPM counts (lpm_check_*).

Sizes
-----
PIScO  ESD = px * sqrt(4 A / pi), recomputed from area: the object_esd in
       project 20736 was exported at 27 um/px, not the correct 22.5.
UVP5   ESD = 0.088 mm * sqrt(4 A / pi): linear acquisition pixel size (default,
       --uvp-size linear). The aa * A_px^exp calibration (--uvp-size powerlaw)
       is fitted on small particles; on >5 mm vignettes it inflates area ~2x
       and biovolume ~3x. The reference report figure used linear.
Biovolume = pi/6 * ESD^3 (sphere), and an ellipsoid variant in the CSVs.

PIScO stores only objects >= ~0.5 mm ESD, the UVP5 only vignettes >= ~0.9 mm,
so the size-matched comparison (--min-esd-mm, default 1.0) is the fair one.

Usage:
    python compare_pisco_uvp5.py --out /media/veit/T710_data/pisco_processed/M181_PISCO_vs_UVP5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Taxon groups (fixed order = fixed colour). Anything not listed is non-living
# (detritus, artefacts, bubbles, temporary t0xx bins, othertocheck) and dropped.
GROUPS = {
    "Trichodesmium": ["Trichodesmium", "tuft", "puff"],
    "Copepoda": ["Copepoda<Multicrustacea", "Calanoida", "Oithona", "like<Copepoda"],
    "Appendicularia": ["Appendicularia<Tunicata"],
    "Rhizaria": ["Rhizaria"],
    "Cnidaria": ["Cnidaria<Animalia", "tentacle<Cnidaria"],
    "Thaliacea": ["Thaliacea"],
    "Other living": ["Chaetognatha<Animalia", "Crustacea", "Eumalacostraca",
                     "shrimp-like", "Ctenophora<Animalia", "Mollusca",
                     "Polychaeta", "Asteroidea<Asterozoa", "Pyrocystis",
                     "Noctiluca<Noctilucaceae", "fish larvae",
                     "pluteus<Echinodermata", "Bacillariophyceae",
                     "multiple species"],
}
GROUP_OF = {name: g for g, names in GROUPS.items() for name in names}
COLORS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                           "#e87ba4", "#008300", "#6250d6"]))
INSTR_COLORS = {"PIScO": "#2a78d6", "UVP5": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

PISCO_FIELDS = ("obj.orig_id,obj.classif_qual,txo.display_name,fre.area,"
                "fre.major_axis_len,fre.minor_axis_len,fre.pressure,sam.orig_id")
UVP_FIELDS = ("obj.orig_id,obj.depth_min,obj.classif_qual,txo.display_name,"
              "fre.area,fre.major,fre.minor,sam.orig_id")


# --------------------------------------------------------------------- EcoTaxa
class EcoTaxa:
    def __init__(self, config: Path):
        cfg = json.loads(config.read_text())["ecotaxa"]
        self.base = cfg["host"].rstrip("/")
        if not self.base.endswith("/api"):
            self.base += "/api"
        self.token = self.call("/login", {"username": cfg["username"],
                                          "password": cfg["password"]})

    def call(self, path, data=None):
        req = urllib.request.Request(
            self.base + path, method="POST" if data is not None else "GET",
            data=json.dumps(data).encode() if data is not None else None)
        req.add_header("Content-Type", "application/json")
        if getattr(self, "token", None):
            req.add_header("Authorization", "Bearer " + self.token)
        with urllib.request.urlopen(req, timeout=600) as f:
            body = f.read().decode()
            return json.loads(body) if body else None

    def objects(self, pid, fields):
        rows, start = [], 0
        while True:
            r = self.call(f"/object_set/{pid}/query?fields={fields}"
                          f"&window_start={start}&window_size=100000", {})
            rows += r["details"]
            start += len(r["details"])
            if not r["details"] or start >= r["total_ids"]:
                break
        return pd.DataFrame(rows, columns=fields.split(","))

    def uvp_meta(self, pid):
        rows = []
        for s in self.call(f"/samples/search?project_ids={pid}&id_pattern="):
            sid = s["sampleid"]
            sam = self.call(f"/sample/{sid}")
            acq = self.call(f"/acquisitions/search?project_id={pid}")
            acq = next(a for a in acq if a["acq_sample_id"] == sid)
            acq = self.call(f"/acquisition/{acq['acquisid']}")["free_columns"]
            # process ids mirror acquisition ids in UVP imports
            prc = self.call(f"/process/{sid_to_acq(sid)}")["free_columns"]
            rows.append(dict(
                uvp_sample=sam["orig_id"], cast=cast_id(sam["orig_id"]),
                first_img=int(prc["first_img"]), last_img=int(prc["last_img"]),
                volimage_L=float(acq["volimage"]), aa=float(acq["aa"]),
                exp=float(acq["exp"]), profileid=sam["free_columns"].get("profileid")))
        m = pd.DataFrame(rows)
        m["n_img"] = m.last_img - m.first_img + 1
        return m


def sid_to_acq(sampleid: int) -> int:
    # sample 17905000001 -> acquisition/process 179050000001
    s = str(sampleid)
    return int(s[:5] + "0" + s[5:])


def cast_id(name: str) -> str:
    m = re.search(r"CTD-(\d{3})", name) or re.search(r"_(\d{3})$", name)
    return f"CTD-{m.group(1)}"


# ----------------------------------------------------------------- PIScO frames
FRAME_RE = re.compile(r"(\d{8})-(\d{2})(\d{2})(\d{2})(\d{2})_(\d+\.\d+)bar")


def pisco_frames(raw_root: Path, casts):
    out = []
    for c in casts:
        d = glob.glob(str(raw_root / f"*{c}_*"))
        if not d:
            raise SystemExit(f"no raw PIScO folder for {c} under {raw_root}")
        for f in os.listdir(Path(d[0]) / "PNG"):
            m = FRAME_RE.match(f)
            if m:
                t = (pd.Timestamp(m.group(1))
                     + pd.Timedelta(hours=int(m.group(2)), minutes=int(m.group(3)),
                                    seconds=int(m.group(4)) + int(m.group(5)) / 100))
                out.append((c, t, float(m.group(6))))
    return pd.DataFrame(out, columns=["cast", "time", "p_bar"]).sort_values(["cast", "time"])


# --------------------------------------------------------------------- volumes
def masked_area_px(radius, size):
    """Pixels inside the segmenter's circular mask, clipped to the frame."""
    y, x = np.mgrid[0:size, 0:size] + 0.5
    return int(((x - size / 2) ** 2 + (y - size / 2) ** 2 <= radius ** 2).sum())


def volumes(frames, uvp_meta, uvp_obj_dmax, edges, v_frame, atm_dbar, uvp_hz):
    rows = []
    frames = frames.assign(p=frames.p_bar * 10 - atm_dbar)
    for c, g in frames.groupby("cast"):
        g = g.sort_values("time")
        n = len(g)
        counts = pd.cut(g.p, edges, right=False).value_counts(sort=False)
        # UVP kept recording below PIScO's deepest frame (CTD-075): extend the
        # timeline at the descent speed of the last 500 m, at 4 Hz.
        extra, extra_bins = 0.0, pd.Series(0.0, index=counts.index)
        m = uvp_meta.set_index("cast").loc[c]
        dmax_p, dmax_u = g.p.max(), uvp_obj_dmax.get(c, g.p.max())
        if dmax_u > dmax_p + 20:
            tail = g[g.p > dmax_p - 500]
            v = (tail.p.iloc[-1] - tail.p.iloc[0]) / (
                tail.time.iloc[-1] - tail.time.iloc[0]).total_seconds()
            fake = np.linspace(dmax_p, dmax_u, int((dmax_u - dmax_p) / v * 4))
            extra = len(fake)
            extra_bins = pd.cut(pd.Series(fake), edges, right=False).value_counts(sort=False)
        for iv, k in counts.items():
            k_uvp = k + extra_bins.get(iv, 0)
            rows.append(dict(
                cast=c, depth_bin_start=iv.left, depth_bin_end=iv.right,
                pisco_frames=int(k), pisco_volume_L=k * v_frame,
                uvp_images_est=m.n_img * k_uvp / (n + extra),
                uvp_volume_L=m.n_img * k_uvp / (n + extra) * m.volimage_L,
                uvp_volume_L_fixed_rate=k_uvp / 4 * uvp_hz * m.volimage_L))
    return pd.DataFrame(rows)


LPM_CLASSES = [("0.512-1.02 mm", 0.512, 1.02), ("1.02-2.05 mm", 1.02, 2.05),
               ("2.05-4.1 mm", 2.05, 4.1), ("4.1-8.19 mm", 4.1, 8.19)]
LAYERS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000)]


def load_ecopart(path: Path, casts):
    """EcoPart reduced export -> one row per 5 m bin: cast, depth, volume, LPM."""
    import io
    import zipfile
    files = {}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if "_PAR_" in n:
                    files[n] = z.read(n)
    else:
        for f in path.glob("*_PAR_*.tsv"):
            files[f.name] = f.read_bytes()
    parts = []
    for name, raw in files.items():
        c = cast_id(Path(name).stem)
        if c in casts:
            parts.append(pd.read_csv(io.BytesIO(raw), sep="\t", encoding="latin1")
                         .assign(cast=c))
    if not parts:
        raise SystemExit(f"no EcoPart PAR files for {sorted(casts)} in {path}")
    ep = pd.concat(parts, ignore_index=True)
    ep = ep.rename(columns={"Depth [m]": "depth", "Sampled volume [L]": "sampled_volume_L"})
    # EcoPart's Depth is the bin centre of a 5 m bin
    ep["depth"] = ep.depth - 2.5
    return ep


def lpm_check(objs, frames, ep, v_frame, atm_dbar, out):
    """All PIScO objects (any class) vs the UVP's own particle counts (LPM),
    per size class and depth layer: independent of vignettes and classifiers."""
    fr = frames.assign(p=frames.p_bar * 10 - atm_dbar)
    pis = objs[objs.instrument == "PIScO"]
    rows = []
    for lo, hi in LAYERS:
        e = ep[(ep.depth >= lo) & (ep.depth < hi)]
        vu = e.groupby("cast").sampled_volume_L.sum()
        vp = fr[(fr.p >= lo) & (fr.p < hi)].groupby("cast").size() * v_frame
        for name, a, b in LPM_CLASSES:
            nu = (e[f"LPM ({name}) [# l-1]"] * e.sampled_volume_L).groupby(e.cast).sum()
            sel = pis[(pis.depth >= lo) & (pis.depth < hi)
                      & (pis.esd_mm >= a) & (pis.esd_mm < b)]
            npis = sel.groupby("cast").size()
            for c in vu.index:
                rows.append(dict(layer=f"{lo}-{hi} m", layer_top=lo, size_class=name,
                                 cast=c, uvp_volume_L=vu[c], pisco_volume_L=vp.get(c, 0),
                                 uvp_lpm_per_L=nu.get(c, 0) / vu[c],
                                 pisco_all_per_L=npis.get(c, 0) / vp.get(c, np.nan)))
    t = pd.DataFrame(rows)
    big = t[t.size_class != "0.512-1.02 mm"]
    pooled = (big.groupby(["layer", "layer_top", "cast"], as_index=False)
              .agg(uvp_volume_L=("uvp_volume_L", "first"),
                   pisco_volume_L=("pisco_volume_L", "first"),
                   uvp_lpm_per_L=("uvp_lpm_per_L", "sum"),
                   pisco_all_per_L=("pisco_all_per_L", "sum"))
              .assign(size_class=">=1.02 mm"))
    t = pd.concat([t, pooled], ignore_index=True)
    t["ratio_pisco_over_uvp"] = t.pisco_all_per_L / t.uvp_lpm_per_L
    t.to_csv(out / "lpm_check_by_layer.csv", index=False)

    casts = sorted(t.cast.unique())
    panels = ["0.512-1.02 mm", "1.02-2.05 mm", "2.05-4.1 mm", ">=1.02 mm"]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.6), sharey=True,
                             sharex=True)
    for ax, name in zip(axes, panels):
        d = t[t.size_class == name]
        for k, c in enumerate(casts):
            dc = d[d.cast == c].sort_values("layer_top")
            y = np.arange(len(dc))
            ax.plot(dc.ratio_pisco_over_uvp, y, "-o", lw=2, ms=5,
                    color=list(COLORS.values())[k], label=c)
        ax.axvline(1, color=INK, lw=1)
        ax.set_xscale("log")
        ax.set_yticks(np.arange(len(LAYERS)), [f"{a}-{b} m" for a, b in LAYERS])
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("PIScO / UVP5 abundance", color=INK)
        style(ax); ax.xaxis.grid(True, color=GRID, lw=0.8)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("All particles, any class: PIScO objects vs UVP5 LPM counts "
                 "(EcoPart volumes). 1 = agreement", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "lpm_check_ratio.png", dpi=200)
    plt.close(fig)
    piv = t[t.size_class == ">=1.02 mm"].pivot_table(
        index=["layer_top", "layer"], columns="cast", values="ratio_pisco_over_uvp")
    print("\nPIScO/UVP-LPM abundance ratio, all particles >= 1.02 mm:")
    print(piv.round(1).to_string())


# --------------------------------------------------------------------- objects
def prep_pisco(df, px_um, atm_dbar):
    d = pd.DataFrame({
        "cast": df["sam.orig_id"].map(cast_id),
        "depth": df["fre.pressure"] - atm_dbar,
        "category": df["txo.display_name"], "qual": df["obj.classif_qual"],
        "esd_mm": px_um / 1000 * np.sqrt(4 * df["fre.area"] / np.pi),
        "major_mm": df["fre.major_axis_len"] * px_um / 1000,
        "minor_mm": df["fre.minor_axis_len"] * px_um / 1000,
    })
    d["instrument"] = "PIScO"
    return d


def prep_uvp(df, meta, size="linear", pixel_mm=0.088):
    if size == "linear":
        area_mm2 = df["fre.area"] * pixel_mm ** 2
    else:
        m = meta.set_index("uvp_sample")
        aa = df["sam.orig_id"].map(m.aa)
        ex = df["sam.orig_id"].map(m.exp)
        area_mm2 = aa * df["fre.area"] ** ex
    # scale lengths so they stay consistent with the calibrated area
    lin = np.sqrt(area_mm2 / df["fre.area"])
    d = pd.DataFrame({
        "cast": df["sam.orig_id"].map(cast_id), "depth": df["obj.depth_min"],
        "category": df["txo.display_name"], "qual": df["obj.classif_qual"],
        "esd_mm": 2 * np.sqrt(area_mm2 / np.pi),
        "major_mm": df["fre.major"] * lin, "minor_mm": df["fre.minor"] * lin,
    })
    d["instrument"] = "UVP5"
    return d


def add_derived(d):
    d["group"] = d.category.map(GROUP_OF)
    d["biovol_mm3"] = np.pi / 6 * d.esd_mm ** 3
    d["biovol_ellipsoid_mm3"] = np.pi / 6 * d.major_mm * d.minor_mm ** 2
    return d


def summarise(objs, vols, lo, hi, min_esd):
    sel = objs[(objs.depth >= lo) & (objs.depth < hi) & objs.group.notna()]
    if min_esd:
        sel = sel[sel.esd_mm >= min_esd]
    v = vols[(vols.depth_bin_start >= lo) & (vols.depth_bin_end <= hi)]
    vol = pd.concat([
        v.groupby("cast").pisco_volume_L.sum().rename("PIScO"),
        v.groupby("cast").uvp_volume_L.sum().rename("UVP5")], axis=1)
    g = (sel.groupby(["instrument", "cast", "group"])
         .agg(n=("esd_mm", "size"), n_validated=("qual", lambda q: (q == "V").sum()),
              biovol_mm3=("biovol_mm3", "sum"),
              biovol_ellipsoid_mm3=("biovol_ellipsoid_mm3", "sum"),
              median_esd_mm=("esd_mm", "median"))
         .reset_index())
    g["volume_L"] = [vol.loc[c, i] for i, c in zip(g.instrument, g.cast)]
    g["abundance_per_L"] = g.n / g.volume_L
    g["biovolume_mm3_per_L"] = g.biovol_mm3 / g.volume_L
    g["biovolume_ellipsoid_mm3_per_L"] = g.biovol_ellipsoid_mm3 / g.volume_L
    return g, vol


# ----------------------------------------------------------------------- plots
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fmt(v):
    return f"{v:.3g}" if v < 10 else f"{v:.0f}"


def log_limits(values, pad_top=3.0):
    v = np.asarray([x for x in values if x > 0])
    return 10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max() * pad_top))


def paired_bars(ax, casts, vals_by_ins, ylim, labels=True):
    """PIScO and UVP5 side by side per cast on a shared log axis."""
    x = np.arange(len(casts))
    w = 0.38
    for k, ins in enumerate(["PIScO", "UVP5"]):
        vals = vals_by_ins[ins]
        ax.bar(x + (k - 0.5) * w, [v if v > 0 else np.nan for v in vals], w - 0.03,
               color=INSTR_COLORS[ins], label=ins, edgecolor="white", linewidth=1)
        if labels:
            for xi, v in zip(x + (k - 0.5) * w, vals):
                ax.annotate(fmt(v) if v > 0 else "0", (xi, v if v > 0 else ylim[0]),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK if v > 0 else MUTED)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xticks(x, casts)
    style(ax)


def plot_triplet(summ, casts, metric, ylabel, title, subtitle, path):
    """A: totals; B/C: composition in % (same axis); D: one panel per group,
    PIScO vs UVP5. A and every D panel share one log axis."""
    piv = summ.pivot_table(index=["instrument", "cast"], columns="group",
                           values=metric, aggfunc="sum").fillna(0)
    groups = [g for g in GROUPS if g in piv.columns]
    piv = piv.reindex(columns=groups)
    piv = piv.reindex(pd.MultiIndex.from_product([["PIScO", "UVP5"], casts]), fill_value=0)
    tot = piv.sum(axis=1)
    ylim = log_limits(list(tot.values) + list(piv.values.ravel()))

    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 1.0, 0.9, 0.9],
                          hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    paired_bars(ax, casts, {i: [tot[(i, c)] for c in casts] for i in ("PIScO", "UVP5")},
                ylim)
    ax.set_ylabel(ylabel + " (log)", color=INK)
    ax.set_title("A) Total, PIScO vs UVP5", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

    x = np.arange(len(casts))
    share = piv.div(tot.replace(0, np.nan), axis=0) * 100
    for k, ins in enumerate(["PIScO", "UVP5"]):
        a = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        bottom = np.zeros(len(casts))
        for grp in groups:
            vals = share.loc[ins, grp].reindex(casts).fillna(0).values
            a.bar(x, vals, 0.7, bottom=bottom, color=COLORS[grp], label=grp,
                  edgecolor="white", linewidth=1)
            bottom += vals
        a.set_ylim(0, 100)
        a.set_xticks(x, casts)
        a.set_ylabel("% of total", color=INK)
        a.set_title(f"{'BC'[k]}) {ins}, composition", loc="left", fontsize=11,
                    fontweight="bold", color=INK)
        style(a)
        if k == 0:
            h, l = a.get_legend_handles_labels()
    a.legend(h[::-1], l[::-1], frameon=False, fontsize=8.5, loc="center left",
             bbox_to_anchor=(1.0, 0.5))

    for i, grp in enumerate(groups):
        a = fig.add_subplot(gs[2 + i // 4, i % 4])
        paired_bars(a, casts, {ins: [piv.loc[(ins, c), grp] for c in casts]
                               for ins in ("PIScO", "UVP5")}, ylim, labels=False)
        a.set_xticks(x, [c[-3:] for c in casts], fontsize=8)
        a.set_title(("D) " if i == 0 else "") + grp, loc="left", fontsize=10,
                    fontweight="bold" if i == 0 else "normal", color=INK)
        if i % 4 == 0:
            a.set_ylabel(ylabel + " (log)", color=INK, fontsize=9)
    if len(groups) % 4:
        a = fig.add_subplot(gs[2 + len(groups) // 4, 3])
        a.axis("off")
        a.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=10,
                 loc="center left", title="D panels\n(no bar = none found)",
                 title_fontsize=9)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.06, 0.937, subtitle + "\nA and all D panels share one log axis; B and C share 0-100 %.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.07, right=0.86)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Layout of the reference figure (report Fig. 14): six groups, tab20 colours.
ORIG_GROUPS = ["Appendicularia", "Cnidaria", "Copepoda", "Rhizaria", "Thaliacea",
               "Trichodesmium"]
ORIG_COLORS = dict(zip(ORIG_GROUPS, plt.get_cmap("tab20").colors[:6]))
ORIG_INSTR = {"PIScO": "#5b9ec1", "UVP5": "#b5658c"}


def plot_original(summ, casts, metric, what, unit, rng, path, share_lower=False):
    """Reproduce the reference figure: A grouped totals, B/C stacked per group.
    share_lower=True puts B and C on the same y axis."""
    piv = (summ[summ.group.isin(ORIG_GROUPS)]
           .pivot_table(index=["instrument", "cast"], columns="group",
                        values=metric, aggfunc="sum")
           .reindex(pd.MultiIndex.from_product([["PIScO", "UVP5"], casts]))
           .reindex(columns=ORIG_GROUPS).fillna(0))
    tot = piv.sum(axis=1)
    x = np.arange(len(casts))

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1], hspace=0.28, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    w = 0.35
    for k, ins in enumerate(["PIScO", "UVP5"]):
        vals = [tot[(ins, c)] for c in casts]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=ORIG_INSTR[ins],
                      label="PISCO" if ins == "PIScO" else ins)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if v >= 0.1 else f"{v:.2g}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, casts)
    ax.set_xlabel("CTD Station", fontweight="bold", fontsize=11)
    ax.set_ylabel(f"Total {what} {unit}", fontweight="bold", fontsize=11)
    ax.set_title(f"A) Total Plankton {what}: PISCO vs UVP5 ({rng})",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, tot.max() * 1.08)

    ymax = max(tot.max(), 1e-12) * 1.03
    axes = []
    for k, ins in enumerate(["PIScO", "UVP5"]):
        a = fig.add_subplot(gs[1, k], sharey=axes[0] if (share_lower and axes) else None)
        bottom = np.zeros(len(casts))
        for grp in ORIG_GROUPS:
            vals = piv.loc[ins, grp].reindex(casts).values
            a.bar(x, vals, 0.8, bottom=bottom, color=ORIG_COLORS[grp], label=grp)
            bottom += vals
        a.set_xticks(x, casts)
        a.set_xlabel("CTD Station", fontweight="bold", fontsize=11)
        a.set_ylabel(f"{what} {unit}", fontweight="bold", fontsize=11)
        name = "PISCO" if ins == "PIScO" else ins
        a.set_title(f"{'BC'[k]}) {name} - Category Distribution", fontweight="bold",
                    fontsize=12)
        a.grid(axis="y", alpha=0.3)
        a.set_axisbelow(True)
        if share_lower:
            a.set_ylim(0, ymax)
        if k == 0:
            a.legend(loc="upper left", fontsize=10)
        axes.append(a)
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.98)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spectra(objs, vol, lo, hi, path):
    """Abundance size spectrum (ind / L / mm), all living groups, 0-100 m."""
    edges = np.logspace(np.log10(0.4), np.log10(20), 26)
    mids = np.sqrt(edges[1:] * edges[:-1])
    width = np.diff(edges)
    casts = sorted(vol.index)
    fig, axes = plt.subplots(1, len(casts), figsize=(3.0 * len(casts), 3.6),
                             sharey=True, sharex=True)
    for ax, c in zip(axes, casts):
        for ins in ("PIScO", "UVP5"):
            d = objs[(objs.instrument == ins) & (objs.cast == c) & objs.group.notna()
                     & (objs.depth >= lo) & (objs.depth < hi)]
            h, _ = np.histogram(d.esd_mm, edges)
            y = h / vol.loc[c, ins] / width
            ok = h > 0
            ax.plot(mids[ok], y[ok], "-o", color=INSTR_COLORS[ins], lw=2, ms=4,
                    label=ins)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(c, fontsize=10, color=INK)
        ax.set_xlabel("ESD (mm)", color=INK)
        style(ax); ax.xaxis.grid(True, color=GRID, lw=0.8)
    axes[0].set_ylabel("ind L$^{-1}$ mm$^{-1}$", color=INK)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Size spectra of living objects, {lo:g}-{hi:g} m "
                 "(dashed: 1 mm size-matching cut)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profiles(objs, vols, path, step, zmax=500):
    """Living abundance (>= 1 mm) vs depth, both instruments, per cast."""
    casts = sorted(vols.cast.unique())
    fig, axes = plt.subplots(1, len(casts), figsize=(3.0 * len(casts), 5.5), sharey=True,
                             sharex=True)
    for ax, c in zip(axes, casts):
        v = vols[vols.cast == c].set_index("depth_bin_start")
        for ins, col in (("PIScO", "pisco_volume_L"), ("UVP5", "uvp_volume_L")):
            d = objs[(objs.instrument == ins) & (objs.cast == c) & objs.group.notna()
                     & (objs.esd_mm >= 1.0)]
            n = (d.depth // step * step).value_counts()
            vv = v[col]
            y = (n.reindex(vv.index).fillna(0) / vv).where(vv > 0)
            y = y[y.index < zmax].replace(0, np.nan)
            ax.plot(y.values, y.index + step / 2, "-o", color=INSTR_COLORS[ins],
                    lw=2, ms=3, label=ins)
        ax.set_xscale("log")
        ax.set_title(c, fontsize=10, color=INK)
        ax.set_xlabel("ind L$^{-1}$", color=INK)
        style(ax); ax.xaxis.grid(True, color=GRID, lw=0.8)
    axes[0].set_ylim(zmax, 0)
    axes[0].set_ylabel("depth (m)", color=INK)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Living objects >= 1 mm ESD, 0-{zmax:g} m, {step:g} m bins "
                 "(empty bins omitted)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pisco-project", type=int, default=20736)
    ap.add_argument("--uvp-project", type=int, default=17905)
    ap.add_argument("--config", type=Path,
                    default=HERE / "process_pisco_profiles.config.example.json")
    ap.add_argument("--pisco-raw-root", type=Path,
                    default=Path("/mnt/filer/M181/M181-PISCO-Profiles"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-range", default="0,100", help="top,bottom in m/dbar")
    ap.add_argument("--min-esd-mm", type=float, default=1.0,
                    help="size-matching cut for the 'matched' figures (0 = off)")
    ap.add_argument("--pixel-um", type=float, default=22.5)
    ap.add_argument("--uvp-size", choices=("linear", "powerlaw"), default="linear",
                    help="UVP5 vignette size: linear pixel (default) or aa*A^exp")
    ap.add_argument("--uvp-pixel-mm", type=float, default=0.088)
    ap.add_argument("--mask-radius", type=float, default=1500,
                    help="PIScO mask radius in px (M181 runs used 1500, two used 1550)")
    ap.add_argument("--frame-px", type=int, default=2560,
                    help="frame edge after the segmenter's resize")
    ap.add_argument("--dof-mm", type=float, default=500,
                    help="PIScO depth of field (same as calc_abundance_biovolume_pisco)")
    ap.add_argument("--atm-dbar", type=float, default=10.0,
                    help="PIScO pressure is absolute; subtract this to get sea pressure")
    ap.add_argument("--uvp-hz", type=float, default=19.0,
                    help="UVP frame rate for the fixed-rate cross-check only")
    ap.add_argument("--ecopart", type=Path,
                    help="EcoPart reduced export (zip or folder with *_PAR_<profile>.tsv); "
                         "replaces the timeline estimate of the UVP volume")
    ap.add_argument("--profile-step", type=float, default=10)
    ap.add_argument("--original-layout-dir", type=Path,
                    help="also write the reference-figure layout (all objects, no "
                         "size cut) into this folder")
    ap.add_argument("--validated-only", action="store_true",
                    help="keep only human-validated objects (status V) on both "
                         "instruments; use a separate --out")
    ap.add_argument("--cache-dir", type=Path,
                    help="where the downloaded tables live (default <out>/cache)")
    ap.add_argument("--refresh", action="store_true", help="re-download from EcoTaxa")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    lo, hi = (float(x) for x in args.depth_range.split(","))

    paths = {k: cache / f"{k}.parquet" for k in ("pisco", "uvp", "uvp_meta", "frames")}
    if args.refresh or not all(p.exists() for p in paths.values()):
        et = EcoTaxa(args.config)
        print("downloading PIScO objects ...")
        et.objects(args.pisco_project, PISCO_FIELDS).to_parquet(paths["pisco"])
        print("downloading UVP5 objects ...")
        et.objects(args.uvp_project, UVP_FIELDS).to_parquet(paths["uvp"])
        et.uvp_meta(args.uvp_project).to_parquet(paths["uvp_meta"])
        meta = pd.read_parquet(paths["uvp_meta"])
        print("listing PIScO frames ...")
        pisco_frames(args.pisco_raw_root, sorted(meta.cast)).to_parquet(paths["frames"])

    meta = pd.read_parquet(paths["uvp_meta"])
    frames = pd.read_parquet(paths["frames"])
    pisco = add_derived(prep_pisco(pd.read_parquet(paths["pisco"]), args.pixel_um,
                                   args.atm_dbar))
    uvp = add_derived(prep_uvp(pd.read_parquet(paths["uvp"]), meta,
                                args.uvp_size, args.uvp_pixel_mm))
    objs = pd.concat([pisco, uvp], ignore_index=True)
    if args.validated_only:
        objs = objs[objs.qual == "V"].reset_index(drop=True)
        print("validated objects only:", objs.groupby("instrument").size().to_dict())

    v_frame = masked_area_px(args.mask_radius, args.frame_px) \
        * (args.pixel_um / 1000) ** 2 * args.dof_mm / 1e6
    print(f"PIScO volume per frame: {v_frame:.3f} L")
    step = args.profile_step
    dmax = max(objs.depth.max(), frames.p_bar.max() * 10) + step
    edges = np.arange(0, dmax + step, step)
    if lo not in edges or hi not in edges:
        edges = np.unique(np.concatenate([edges, [lo, hi]]))
    uvp_dmax = uvp.groupby("cast").depth.max().to_dict()
    vols = volumes(frames, meta, uvp_dmax, edges, v_frame, args.atm_dbar, args.uvp_hz)
    ep = None
    if args.ecopart:
        ep = load_ecopart(args.ecopart, set(meta.cast))
        ev = (ep.assign(depth_bin_start=(ep.depth // step) * step)
              .groupby(["cast", "depth_bin_start"]).sampled_volume_L.sum())
        vols = vols.rename(columns={"uvp_volume_L": "uvp_volume_L_timeline_est"})
        vols["uvp_volume_L"] = [ev.get((c, z), 0.0) for c, z in
                                zip(vols.cast, vols.depth_bin_start)]
        print("UVP volume: EcoPart")
    else:
        print("UVP volume: PIScO-timeline estimate (pass --ecopart for real volumes)")
    vols.to_csv(out / "sampled_volume_by_depth.csv", index=False)

    casts = sorted(meta.cast)
    full = hi >= max(objs.depth.max(), frames.p_bar.max() * 10)
    rng = "full water column" if full else f"{lo:g}-{hi:g} m"
    tag = f"{lo:g}-fullm".replace("-fullm", "-full") if full else f"{lo:g}-{hi:g}m"
    for label, cut in (("all", 0.0), (f"esd{args.min_esd_mm:g}mm", args.min_esd_mm)):
        summ, vol = summarise(objs, vols, lo, hi, cut)
        summ.to_csv(out / f"summary_{tag}_{label}.csv", index=False)
        if label == "all" and args.original_layout_dir:
            od = args.original_layout_dir
            od.mkdir(parents=True, exist_ok=True)
            orng = "full water column" if full else f"{lo:g}-{hi:g} m/dbar"
            if args.validated_only:
                orng += ", validated only"
            for metric, what, unit, stem in (
                    ("abundance_per_L", "Abundance", "(objects per litre)", "abundance"),
                    ("biovolume_mm3_per_L", "Biovolume", "(mm³ per litre)", "biovolume")):
                for share, suffix in ((False, "own_axes"), (True, "same_axes")):
                    plot_original(summ, casts, metric, what, unit, orng,
                                  od / f"{stem}_{tag}_{suffix}.png", share_lower=share)
            sel = "validated_only" if args.validated_only else "all_objects"
            summ.to_csv(od / f"summary_{tag}_{sel}.csv", index=False)
        sizes = (f"objects >= {cut:g} mm ESD on both instruments" if cut else
                 "all stored objects (PIScO >= ~0.5 mm, UVP5 >= ~0.9 mm ESD)")
        sub = (f"{rng}, living objects only, {sizes}.\nVolume imaged {rng}: PIScO "
               f"{vol.PIScO.min():.0f}-{vol.PIScO.max():.0f} L, UVP5 "
               f"{vol.UVP5.min():.0f}-{vol.UVP5.max():.0f} L per cast.")
        plot_triplet(summ, casts, "abundance_per_L", "Objects per litre",
                     f"Plankton abundance, PIScO vs UVP5 ({rng})", sub,
                     out / f"abundance_{tag}_{label}.png")
        plot_triplet(summ, casts, "biovolume_mm3_per_L", "Biovolume (mm³ per litre)",
                     f"Plankton biovolume, PIScO vs UVP5 ({rng})", sub,
                     out / f"biovolume_{tag}_{label}.png")
        print(f"\n== {label}: totals per cast ==")
        t = summ.groupby(["cast", "instrument"])[
            ["n", "abundance_per_L", "biovolume_mm3_per_L"]].sum().unstack()
        print(t.round(3).to_string())
        print(vol.round(0).to_string())
    plot_spectra(objs, vol, lo, hi, out / f"size_spectra_{tag}.png")
    if ep is not None:
        lpm_check(objs, frames, ep, v_frame, args.atm_dbar, out)
    plot_profiles(objs, vols, out / "profiles_living_esd1mm.png", step)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
