"""The crop a downstream consumer should see, derived from what the segmenter stored.

The C++ segmenter stores each object's crop as its padded bounding box -
30 px of real frame on every side, which is what the deconvolution model
needs for context - and a mask made from the object's own contour. Handed
on as-is, that box shows neighbours, and it is not what the classifier was
trained on. The legacy python route saved a *tight* box with everything
outside the contour set to white, and the ViTs learned on those.

``tight_masked`` turns the stored crop into that: cut to the mask's bounding
box, everything outside the mask set to the ground value. Used by the EcoTaxa
exporter and the classifier so the two never disagree about what a crop
looks like.
"""
from __future__ import annotations

import numpy as np


def tight_masked(crop: np.ndarray, mask: np.ndarray, margin: int = 0,
                 fill: int = 255) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Cut ``crop`` and ``mask`` to the mask's bounding box and blank the rest.

    ``crop`` is dark on light (already inverted), ``mask`` is nonzero on the
    object. ``margin`` keeps that many pixels of real crop around the mask's
    box; 0 reproduces the python route. Returns ``(crop, mask, x0, y0)`` where
    ``x0, y0`` is the cut's offset within the input, so callers can shift the
    object's frame coordinates. An empty mask returns the inputs unchanged.
    """
    on = mask > 0
    if not on.any():
        return crop, mask, 0, 0
    ys, xs = np.where(on)
    y0, y1 = max(0, int(ys.min()) - margin), min(mask.shape[0], int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(mask.shape[1], int(xs.max()) + 1 + margin)
    cut_mask = mask[y0:y1, x0:x1]
    cut_crop = np.where(cut_mask > 0, crop[y0:y1, x0:x1], fill).astype(crop.dtype)
    return cut_crop, np.ascontiguousarray(cut_mask), x0, y0
