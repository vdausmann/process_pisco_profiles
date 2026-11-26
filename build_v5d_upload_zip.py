"""Build ONE EcoTaxa upload zip for the v5d ATAIR run: all living objects plus a
stratified random sample of the not-living ones, for annotation -> ViT retraining.

Why not-living matters here: the ViT was trained on OLD-deconvolution crops, so on v5d
crops it dumps most real plankton into `not-living` (measured: living rate falls 83-93%,
and a controlled same-object test flips 14 living -> not-living with 0 reverse flips).
The misclassified plankton is therefore IN the not-living pile, which is exactly the
material that needs hand-annotation to retrain on.

Crops come from `deconv_crops.zip` (the v5d-deconvolved crops the retrained ViT will see),
NOT `crops.zip` (which holds the cleaned/bg-corrected crops by design).
"""
import os, sys, csv, random, zipfile

BASE = "/media/veit/T710_data/pisco_processed/ATAIR-BSH_v5dlucyd"
PROFILES = ["20260422-2051", "20260423-1100", "20260423-1705"]
N_NOTLIVING = 20000          # sized for reuse as fine-tuning data
SEED = 42
OUT = "/media/veit/T710_data/ATAIR_v5d_lucyd_upload.zip"


def group_zip(profile, group):
    return os.path.join(BASE, profile, f"{profile}_Results", "EcoTaxa", group,
                        "deconv_crops.zip")


def read_tsv(zf):
    name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
    lines = zf.read(name).decode("utf-8", "replace").splitlines()
    header, markers, rows = lines[0], lines[1], lines[2:]
    return header, markers, [r for r in rows if r.strip()]


def main():
    rng = random.Random(SEED)

    # --- pass 1: count not-living per profile for proportional allocation ---
    counts = {}
    for p in PROFILES:
        with zipfile.ZipFile(group_zip(p, "non-living")) as z:
            _, _, rows = read_tsv(z)
            counts[p] = len(rows)
    total_nl = sum(counts.values())
    alloc = {p: int(round(N_NOTLIVING * counts[p] / total_nl)) for p in PROFILES}
    print("not-living population and allocation:")
    for p in PROFILES:
        print(f"  {p}: {counts[p]:>9,}  -> sample {alloc[p]:,}")

    header = markers = None
    out_rows = []
    to_copy = []            # (source_zip_path, member_name)
    n_living = 0

    for p in PROFILES:
        # all living
        zp = group_zip(p, "other")
        with zipfile.ZipFile(zp) as z:
            h, m, rows = read_tsv(z)
            header, markers = header or h, markers or m
            assert h == header, f"TSV column mismatch in {zp}"
            members = {os.path.basename(n): n for n in z.namelist() if n.endswith(".png")}
            fi = header.split("\t").index("img_file_name")
            for r in rows:
                fn = r.split("\t")[fi]
                if fn in members:
                    out_rows.append(r); to_copy.append((zp, members[fn])); n_living += 1

        # stratified random not-living
        zp = group_zip(p, "non-living")
        with zipfile.ZipFile(zp) as z:
            h, m, rows = read_tsv(z)
            assert h == header, f"TSV column mismatch in {zp}"
            members = {os.path.basename(n): n for n in z.namelist() if n.endswith(".png")}
            fi = header.split("\t").index("img_file_name")
            usable = [r for r in rows if r.split("\t")[fi] in members]
            pick = rng.sample(usable, min(alloc[p], len(usable)))
            for r in pick:
                out_rows.append(r); to_copy.append((zp, members[r.split("\t")[fi]]))
        print(f"  {p}: living so far {n_living:,}, total rows {len(out_rows):,}")

    # --- integrity checks before writing ---
    cols = header.split("\t")
    fi, oi = cols.index("img_file_name"), cols.index("object_id")
    fns = [r.split("\t")[fi] for r in out_rows]
    oids = [r.split("\t")[oi] for r in out_rows]
    assert len(set(fns)) == len(fns), f"duplicate img_file_name ({len(fns)-len(set(fns))})"
    assert len(set(oids)) == len(oids), f"duplicate object_id ({len(oids)-len(set(oids))})"
    print(f"\nunique img_file_name and object_id OK ({len(fns):,} rows)")

    # --- write the combined zip: crops stored FLAT, keyed by img_file_name ---
    tsv = "\n".join([header, markers] + out_rows) + "\n"
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        out.writestr("ecotaxa_ATAIR_v5d_lucyd.tsv", tsv)
        cache = {}
        for i, (zp, member) in enumerate(to_copy):
            if zp not in cache:
                cache[zp] = zipfile.ZipFile(zp)
            out.writestr(os.path.basename(member), cache[zp].read(member))
            if i % 2500 == 0:
                print(f"  packed {i:,}/{len(to_copy):,}", end="\r")
        for z in cache.values():
            z.close()
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"  living {n_living:,} | not-living {len(out_rows)-n_living:,} | total {len(out_rows):,}")


if __name__ == "__main__":
    main()
