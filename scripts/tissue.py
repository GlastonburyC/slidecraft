"""
Slidecraft's tissue detection, as it runs outside the browser.

A port of ``src/ml/tissue.ts``, kept deliberately close to it: same overview
size, same score, same Otsu, same two thresholds, same open-bind-grow order.
The point is that a patch chosen here is a patch Slidecraft would have chosen.
Anything simpler -- a percentile, a fixed saturation cut -- puts the cluster's
patches somewhere the browser's "Detect tissue" does not agree with, and the
two views of the same slide then disagree about where the tissue was.

Where the two must differ they differ only in mechanics: the browser walks its
masks in hand-written loops, this leans on scipy. The numbers are the same.

If you change one, change the other. ``test_tissue.py`` pins the parts that
can be checked without a slide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Longest side of the working overview, in pixels. src/ml/tissue.ts:76
MAX_OVERVIEW = 2048

EDGE_BIAS = 0.22
WEAK_BIAS = 0.1
SEAM_RADIUS = 1
BIND_RADIUS = 3
MAX_ASPECT = 10.0
MIN_WIDTH_UM = 400.0
MAX_HOLE_UM2 = 200_000.0
MIN_AREA_UM2 = 250_000.0


def otsu(histogram: np.ndarray, total: int) -> int:
    """
    Otsu's method over a 256-bin histogram, returning the middle of a tied run.

    Where several thresholds tie, the tie is the empty valley between glass and
    tissue -- every cut through it separates the two classes equally well. The
    first such cut sits hard against the glass, where sensor noise then reads as
    tissue; the middle of the valley is the stable choice.
    """
    levels = np.arange(256, dtype=np.float64)
    total_sum = float((levels * histogram).sum())

    sum_b = 0.0
    w_b = 0
    first = last = 0
    best = -1.0
    for t in range(256):
        w_b += int(histogram[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * float(histogram[t])
        m_b = sum_b / w_b
        m_f = (total_sum - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        # A relative tolerance: the variance is a product of large numbers and
        # exact equality across a flat valley is a floating-point accident.
        if between > best * (1 + 1e-9):
            best = between
            first = last = t
        elif between >= best * (1 - 1e-9):
            last = t
    return int(round((first + last) / 2))


@dataclass
class OverviewScore:
    score: np.ndarray          # uint8, (h, w)
    auto: int                  # Otsu's split, at the centre of the valley
    glass: int                 # modal score below the split: the blank slide
    span: int
    relaxed: int               # where tissue is taken to start
    weak_level: int            # where growth into faint tissue stops


def score_overview(rgb: np.ndarray,
                   edge_bias: float = EDGE_BIAS,
                   weak_bias: float = WEAK_BIAS) -> OverviewScore:
    """
    The colour-only reading of an overview: a per-cell score and its levels.

    Tissue score is how far a pixel is from blank slide, taking the stronger of
    two independent cues. Saturation alone finds well-stained tissue but misses
    faded H&E, which is pale yet still darker than glass; darkness alone catches
    faded tissue but also shadows and coverslip edges. The maximum lets either
    cue claim a pixel, so a section with both strong and faded areas comes out
    whole rather than in patches.
    """
    a = rgb.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    saturation = np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1e-6) * 255.0)
    darkness = 255.0 - (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    score = np.rint(np.maximum(saturation, darkness)).clip(0, 255).astype(np.uint8)

    histogram = np.bincount(score.ravel(), minlength=256)

    # One occupied bin means there is nothing to separate: Otsu's between-class
    # variance never goes positive, `auto` falls to 0, and the levels below --
    # anchored on a span of 1 and floored at 6 -- land UNDER the score of plain
    # glass, so a featureless image comes back as tissue everywhere. An image of
    # a single colour has no tissue in it; say so rather than thresholding noise
    # that is not there.
    if int((histogram > 0).sum()) < 2:
        return OverviewScore(score, 0, 0, 1, 256, 256)

    auto = otsu(histogram, score.size)

    # Blank slide is the tallest peak below the automatic threshold, so the
    # histogram says where "certainly not tissue" ends without being told.
    # Anchoring both working levels on the span from glass up to Otsu is what
    # makes them stable: a plain fraction of Otsu moves with wherever that
    # threshold lands, and on a slide with little tissue slips under the glass
    # -- at which point the scan's own sensor noise is tissue.
    glass = int(np.argmax(histogram[: auto + 1])) if auto >= 0 else 0
    span = max(1, auto - glass)

    relaxed = max(6, int(round(glass + span * edge_bias)))
    weak_level = max(glass + 2, int(round(glass + span * weak_bias)))
    return OverviewScore(score, auto, glass, span, relaxed, weak_level)


def _square(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode then dilate: deletes anything thinner than the element."""
    if radius <= 0:
        return mask
    s = _square(radius)
    return ndimage.binary_dilation(ndimage.binary_erosion(mask, s), s)


def closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate then erode: bridges speckled gaps without growing the outline."""
    if radius <= 0:
        return mask
    s = _square(radius)
    return ndimage.binary_erosion(ndimage.binary_dilation(mask, s), s)


def label_and_grow(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """
    Label the connected fragments of ``strong``, then grow each into ``weak``.

    Hysteresis and labelling in one pass, because they are the same pass. The
    strict threshold decides where tissue certainly is, the looser one how far
    it plausibly extends, and loose cells are accepted only where they join
    something already certain -- so a pale fragment survives whole while equally
    pale noise alone on the glass is still rejected, having nothing to anchor to.

    Growth proceeds one ring at a time with every fragment advancing together,
    so two fragments growing toward each other meet at a boundary instead of
    merging. The count of objects is fixed by the confident mask.
    """
    labels, _ = ndimage.label(strong, structure=ndimage.generate_binary_structure(2, 1))
    cross = ndimage.generate_binary_structure(2, 1)

    # One simultaneous ring per iteration is the breadth-first frontier: a cell
    # is claimed by whichever fragment reaches it first, and ties inside a ring
    # go to the lower label rather than to iteration order.
    while True:
        frontier = ndimage.binary_dilation(labels > 0, cross) & weak & (labels == 0)
        if not frontier.any():
            break
        # Lowest neighbouring label wins. Max-filtering the negated labels is
        # how to get a minimum over the non-zero ones in a single pass.
        big = np.where(labels > 0, labels, np.iinfo(np.int32).max)
        nearest = ndimage.minimum_filter(big, footprint=cross, mode="constant",
                                         cval=np.iinfo(np.int32).max)
        labels = np.where(frontier, nearest, labels)
    return labels


def fill_holes(mask: np.ndarray, max_cells: int) -> np.ndarray:
    """
    Fill background regions that do not touch the border and are small enough.

    A lumen or a pale patch inside a section is still that section. But an
    unbounded fill swallows whatever blank space happens to be enclosed by a
    ring of fragments, inventing tissue where there is none -- hence the cap.
    """
    background, n = ndimage.label(~mask, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return mask
    out = mask.copy()

    # Everything reachable from the border is outside, never a hole.
    edge = np.concatenate([background[0, :], background[-1, :],
                           background[:, 0], background[:, -1]])
    outside = set(np.unique(edge).tolist()) - {0}

    sizes = np.bincount(background.ravel())
    for comp in range(1, n + 1):
        if comp in outside:
            continue
        if sizes[comp] <= max_cells:
            out[background == comp] = True
    return out


def tissue_mask(rgb: np.ndarray, um_per_cell: float | None) -> np.ndarray:
    """
    The boolean tissue mask of an overview, at overview resolution.

    ``um_per_cell`` is the width one overview cell covers on the slide; without
    it the physical-size guards fall back to fractions of the grid.
    """
    scored = score_overview(rgb)

    raw = scored.score > scored.relaxed
    # Open before closing. Opening deletes structures thinner than the element
    # -- scanner seams and coverslip edges are a cell or two wide at overview
    # scale, real sections an order of magnitude wider. Order matters: closing
    # first would bridge a seam to the fragment beside it, and they would
    # survive as one region with a spur that no shape test can then separate.
    opened = opening(raw, SEAM_RADIUS)
    weak = opening(scored.score > scored.weak_level, SEAM_RADIUS)

    # Bind each fragment before it identifies anything. Thresholded real tissue
    # is speckled -- stroma and fat fall below the confident level while the
    # nuclei around them clear it -- so the raw mask of ONE section is dozens of
    # islands, and seeding identity from that shatters it into dozens of
    # objects. The gap between two sections is an order of magnitude wider than
    # the gaps inside one, so it survives the same closing untouched.
    seeds = closing(opened, BIND_RADIUS)
    labels = label_and_grow(seeds, weak)

    if labels.max() == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)

    cell_um2 = (um_per_cell ** 2) if um_per_cell else None
    min_cells = max(4, MIN_AREA_UM2 / cell_um2) if cell_um2 else 16
    max_hole = (max(1, round(MAX_HOLE_UM2 / cell_um2)) if cell_um2
                else max(1, round(labels.size * 0.002)))

    keep = np.zeros(labels.shape, dtype=bool)
    objects = ndimage.find_objects(labels)
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        piece = labels[sl] == i
        if piece.sum() < min_cells:
            continue
        h_cells = max(1, sl[0].stop - sl[0].start)
        w_cells = max(1, sl[1].stop - sl[1].start)
        # Drop long thin strips. Scanner seams run the height of a scan row at
        # near-constant width; tissue does not. The absolute-width guard keeps a
        # needle core -- thin, but far wider than a seam -- out of this.
        aspect = max(w_cells, h_cells) / min(w_cells, h_cells)
        if aspect >= MAX_ASPECT:
            if um_per_cell is None or min(w_cells, h_cells) * um_per_cell < MIN_WIDTH_UM:
                continue
        keep[sl] |= piece

    return fill_holes(keep, max_hole)
