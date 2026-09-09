"""Reversible scale bars for crops shown to annotators.

Lifted out of ``utils.py`` unchanged so that callers which only need a scale
bar do not pay for that module's imports -- it pulls in torch, transformers,
pandas and matplotlib at module scope, none of which are installed on the
processing workers. This module needs nothing but Pillow.

``utils`` re-exports these names, so existing callers are unaffected.
"""

# Rows appended below a crop to carry the scale bar. Fixed, so the strip is
# removed by dropping this many rows (see strip_scale_bar).
SCALE_BAR_STRIP_H = 34

# Bar lengths in micrometres, longest first. 500 um is the floor: a bar shorter
# than that is not a useful reference, so a crop too narrow to hold one is
# padded (always on the right) rather than given a smaller bar.
_SCALE_BAR_STEPS_UM = (1000, 500)
_SCALE_BAR_MIN_UM = 500

# Original size is recorded here so strip_scale_bar can undo padding as well as
# the strip; without it a right-padded crop is indistinguishable from one that
# genuinely has white on the right.
_ORIG_SIZE_KEY = "pisco_orig_size"


def _scale_bar_geometry(width, pixel_resolution):
    """Return (bar_um, bar_px, padded_width) for a crop of `width` px."""
    margin = 3
    for cand in _SCALE_BAR_STEPS_UM:
        if cand / pixel_resolution <= max(1, width - 2 * margin):
            return cand, int(round(cand / pixel_resolution)), width
    # Too narrow even for the floor bar: widen to the right until it fits.
    bar_px = int(round(_SCALE_BAR_MIN_UM / pixel_resolution))
    return _SCALE_BAR_MIN_UM, bar_px, bar_px + 2 * margin


def add_scale_bar_strip(img, pixel_resolution=23, strip_h=SCALE_BAR_STRIP_H):
    """Append a scale bar below `img`, padding right only if the bar needs it.

    The crop's own pixels are never altered, and the original size is stored in
    the PNG so `strip_scale_bar` restores it exactly. That matters because these
    crops are also classifier input: a burnt-in bar is an appearance change the
    ViT never saw in training.

    The bar is 1 mm where it fits and 500 um otherwise; a crop too narrow for a
    500 um bar is padded on the right, so the object stays left-aligned and in
    the same place from crop to crop.

    Returns a new PIL image, `strip_h` rows taller and possibly wider.
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    bar_um, bar_px, new_w = _scale_bar_geometry(w, pixel_resolution)

    out = _Image.new("L", (new_w, h + strip_h), 255)
    out.paste(img, (0, 0))                      # left-aligned: padding is on the right
    draw = _ImageDraw.Draw(out)

    bar_h = 3
    bar_x = (new_w - bar_px) // 2
    bar_y = h + strip_h - bar_h - 3
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + bar_h], fill=0)

    label = "1 mm" if bar_um >= 1000 else f"{bar_um} um"
    try:
        tw = draw.textlength(label)
    except AttributeError:                       # very old Pillow
        tw = len(label) * 6
    if tw <= new_w:
        draw.text(((new_w - tw) / 2, h + 2), label, fill=0)

    out.info[_ORIG_SIZE_KEY] = f"{w}x{h}"
    return out


def save_crop_with_scale_bar(img, fp, pixel_resolution=23):
    """Write `img` with a scale bar, recording its original size in the PNG."""
    from PIL import PngImagePlugin
    out = add_scale_bar_strip(img, pixel_resolution=pixel_resolution)
    meta = PngImagePlugin.PngInfo()
    meta.add_text(_ORIG_SIZE_KEY, out.info[_ORIG_SIZE_KEY])
    out.save(fp, format="PNG", pnginfo=meta)
    return out


def strip_scale_bar(img, strip_h=SCALE_BAR_STRIP_H):
    """Inverse of `add_scale_bar_strip`, undoing the strip and any right padding.

    Uses the size recorded by `save_crop_with_scale_bar` when present. Without
    it only the strip can be removed, since right padding is not detectable from
    the pixels alone - so read crops with PIL (which keeps PNG text chunks)
    rather than through a converter that drops them.
    """
    w, h = img.size
    recorded = (img.info or {}).get(_ORIG_SIZE_KEY)
    if recorded:
        try:
            ow, oh = (int(v) for v in recorded.split("x"))
            return img.crop((0, 0, ow, oh))
        except (ValueError, TypeError):
            pass
    if h <= strip_h:
        return img
    return img.crop((0, 0, w, h - strip_h))
