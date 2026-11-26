"""Controlled test: does swapping the deconvolution model alone flip the binary ViT?

Everything except the deconvolution model is held fixed. For a set of ATAIR frames we
rebuild the cleaned frame, deconvolve it with the OLD and the V5D model, cut crops at the
IDENTICAL bounding boxes (taken from the old run's metadata), and run the same binary ViT
on both crop sets. Same objects, same boxes, same classifier -> any difference in the
living rate is attributable to the deconvolution model.

This mirrors the pipeline faithfully: it deconvolves the FULL frame and then crops, which
is what process_pisco_profiles does (crops are cut from the deconvolved frame).
"""
import os, sys, shutil, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
sys.path.insert(0, "/home/veit/PIScO_dev/process_pisco_profiles")
sys.path.insert(0, "/home/veit/PIScO_dev/underwater-lucyd")
sys.path.insert(0, "/home/veit/PIScO_dev/underwater-lucyd/sim")
import numpy as np
import pandas as pd
import cv2 as cv
import torch
import utils as ap
import lib_bgcorr
from lucyd import LUCYD

PROFILE = "20260423-1100"
OLD_RESULTS = f"/media/veit/T710_data/pisco_processed/ATAIR-BSH/{PROFILE}/{PROFILE}_Results"
META = f"{OLD_RESULTS}/EcoTaxa/{PROFILE}_crops_metadata.csv"
RAW = f"/mnt/filer/202604_ATAIR-BSH/202604_ATAIR-BSB_PISCO/20260423/{PROFILE}"
BINARY_MODEL = "/home/veit/PIScO_dev/ViT_custom_size_sensitive_binary/best_model"
MODELS = {
    "old": "/home/veit/PIScO_dev/pisco_segmenter/pisco_segmenter/models/lucyd-edof-plankton_231204.pth",
    "v5d": "/home/veit/PIScO_dev/underwater-lucyd/models/lucyd-edof-plankton_v5d_20260807.pth",
}
N_FRAMES = 12
OUT = "/media/veit/T710_data/ctrl_deconv_vit_clipped"


def load_lucyd(path, dev):
    m = LUCYD(num_res=1).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    return m


WHITE_CLIP = 245        # pipeline saves crops with >=245 snapped to 255 (verified MAE 0.002)


def deconv_full(model, cleaned, dev, clip=True):
    """Same as the segmenter: feed the full cleaned frame, no padding (2560 % 16 == 0).

    `clip` reproduces the crop-saving step: everything >= 245 becomes pure white. Without it
    the ViT is fed an image the pipeline never produces (raw output has ~0% pure-255 pixels,
    real crops have 40-50%), so the living rates come out unrepresentative.
    """
    x = torch.from_numpy(cleaned / 255.0).unsqueeze(0).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        y, _, _ = model(x)
    out = np.clip(y.cpu().numpy()[0, 0] * 255, 0, 255).astype(np.uint8)
    if clip:
        out[out >= WHITE_CLIP] = 255
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(META, low_memory=False,
                     usecols=["object_full_path", "object_x", "object_y",
                              "object_annotation_category", "object_area"])
    df["frame"] = df.object_full_path.map(lambda p: os.path.basename(p).rsplit("_", 1)[0])

    files = sorted(f for f in os.listdir(RAW) if f.endswith(".tif"))
    names = [f.rsplit(".tif", 1)[0] for f in files]
    # frames in the middle of the profile that carry objects
    cand = [n for n in names if n in set(df.frame)]
    picks = cand[len(cand) // 2: len(cand) // 2 + N_FRAMES]
    print(f"{len(cand)} frames with objects; using {len(picks)}")

    models = {k: load_lucyd(v, dev) for k, v in MODELS.items()}
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    dirs = {k: os.path.join(OUT, k) for k in MODELS}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # rebuild cleaned frames with the neighbour window, deconvolve, cut identical boxes
    idx = {n: i for i, n in enumerate(names)}
    win = lib_bgcorr.BG_WIN
    lo = max(0, idx[picks[0]] - win)
    hi = min(len(files), idx[picks[-1]] + win + 1)
    block = [os.path.join(RAW, f) for f in files[lo:hi]]
    kept = 0
    per_obj = []
    for fn, cleaned in lib_bgcorr.clean_frames(block):
        name = os.path.basename(fn).rsplit(".tif", 1)[0]
        if name not in set(picks):
            continue
        sub = df[df.frame == name]
        if sub.empty:
            continue
        dec = {k: deconv_full(m, cleaned, dev) for k, m in models.items()}
        for _, r in sub.iterrows():
            crop_png = r.object_full_path
            base = os.path.basename(crop_png)
            ref = cv.imread(crop_png, 0)          # old run's cleaned crop -> gives the box size
            if ref is None:
                continue
            h, w = ref.shape
            x, y = int(r.object_x), int(r.object_y)
            if y + h > 2560 or x + w > 2560:
                continue
            for k in MODELS:
                cv.imwrite(os.path.join(dirs[k], base), dec[k][y:y + h, x:x + w])
            per_obj.append({"filename": base, "area": r.object_area,
                            "old_run_category": r.object_annotation_category})
        kept += 1
        print(f"  frames done {kept}/{len(picks)}, objects {len(per_obj)}", end="\r")
    print()
    meta = pd.DataFrame(per_obj)
    print(f"{len(meta)} objects cut from {kept} frames")

    # run the SAME binary ViT over both crop sets
    res = {}
    for k, d in dirs.items():
        fnames = set(os.listdir(d))
        ds = ap.load_unclassified_images(d, filenames=fnames)
        ds = ds.with_transform(ap.process_batch)
        f, preds, probs, ent, ood = ap.get_predictions_with_entropy_ood_binary(
            ds, BINARY_MODEL, entropy_threshold=0.5, temperature=1.5, batch_size=64)
        res[k] = pd.DataFrame({"filename": [os.path.basename(x) for x in f],
                               f"{k}_top1": [p[0] for p in preds],
                               f"{k}_prob": [p[0] for p in probs]})
        print(f"{k}: {(res[k][f'{k}_top1'] == 'living').sum()} / {len(res[k])} living "
              f"({100 * (res[k][f'{k}_top1'] == 'living').mean():.2f}%)")

    m = res["old"].merge(res["v5d"], on="filename").merge(meta, on="filename", how="left")
    m.to_csv("/home/veit/PIScO_dev/process_pisco_profiles/ctrl_deconv_vit_clipped.csv", index=False)
    print("\n=== confusion (same object, same ViT, different deconvolution) ===")
    print(pd.crosstab(m.old_top1, m.v5d_top1, rownames=["old deconv"], colnames=["v5d deconv"]))
    flipped = m[(m.old_top1 == "living") & (m.v5d_top1 == "not-living")]
    print(f"\nliving under old -> not-living under v5d: {len(flipped)}")
    if len(flipped):
        print(f"  their old-run categories: "
              f"{flipped.old_run_category.value_counts().head(6).to_dict()}")
        print(f"  median area {flipped.area.median():.0f} px")


if __name__ == "__main__":
    main()
