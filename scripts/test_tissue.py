"""
Checks for the tissue port's pure parts.

    .venv-export/bin/python -m pytest scripts/test_tissue.py -q

This is a port of src/ml/tissue.ts, and a port is only worth having if it keeps
agreeing with its original. What is pinned here is the behaviour the TypeScript
comments claim -- the plateau centre, the glass anchoring, hysteresis that keeps
fragments apart, the hole cap -- because those are the parts that can go subtly
wrong and still return a plausible mask.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tissue import (  # noqa: E402
    closing, fill_holes, label_and_grow, opening, otsu, score_overview, tissue_mask,
)


def test_otsu_returns_the_middle_of_a_tied_run():
    """
    Two spikes with an empty valley between them. Every cut through the valley
    separates them equally well, so the tie runs its whole width -- and the
    threshold belongs at its centre, not hard against the glass where sensor
    noise would then read as tissue.
    """
    h = np.zeros(256, dtype=np.int64)
    h[10] = 1000
    h[200] = 1000
    t = otsu(h, 2000)
    assert 100 <= t <= 110, t


def test_otsu_is_not_dragged_to_the_first_tie():
    h = np.zeros(256, dtype=np.int64)
    h[0] = 500
    h[100] = 500
    assert otsu(h, 1000) > 10


def _canvas(w=200, h=120, value=245):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_glass_is_the_modal_level_below_otsu():
    img = _canvas()
    img[40:80, 40:120] = (120, 60, 130)  # one stained block
    s = score_overview(img)
    # Blank slide at 245 grey scores ~10; the tissue block scores far higher.
    assert s.glass < s.auto
    assert s.relaxed > s.glass
    assert s.weak_level >= s.glass + 2
    # Both levels sit on the span between glass and Otsu, not above it.
    assert s.glass < s.weak_level <= s.relaxed < s.auto


def test_the_weak_level_sits_below_the_confident_one():
    img = _canvas()
    img[30:90, 30:170] = (150, 80, 160)
    s = score_overview(img)
    assert s.weak_level < s.relaxed


def test_saturation_and_darkness_each_can_claim_a_pixel():
    """Faded-but-dark and pale-but-coloured both have to count as tissue."""
    img = _canvas()
    img[10:20, 10:20] = (110, 110, 110)   # grey: dark, no saturation
    img[30:40, 30:40] = (250, 180, 250)   # pale: saturated, not dark
    s = score_overview(img)
    assert s.score[15, 15] > s.score[0, 0]
    assert s.score[35, 35] > s.score[0, 0]


def test_opening_deletes_a_line_thinner_than_the_element():
    m = np.zeros((40, 40), dtype=bool)
    m[20, :] = True                      # one cell tall
    assert opening(m, 1).sum() == 0


def test_opening_keeps_a_block_wider_than_the_element():
    m = np.zeros((40, 40), dtype=bool)
    m[10:30, 10:30] = True
    assert opening(m, 1).sum() > 0


def test_closing_bridges_a_speckled_gap():
    m = np.zeros((40, 40), dtype=bool)
    m[10:30, 10:30] = True
    m[10:30, 19:21] = False              # a two-cell slot through it
    assert closing(m, 3)[20, 20]


def test_hysteresis_grows_a_fragment_into_weak_cells():
    strong = np.zeros((30, 30), dtype=bool)
    strong[14:16, 14:16] = True
    weak = np.zeros((30, 30), dtype=bool)
    weak[10:20, 10:20] = True
    grown = label_and_grow(strong, weak)
    assert grown[11, 11] == 1            # reached, because it joins the seed
    assert (grown > 0).sum() == 100


def test_weak_cells_touching_nothing_confident_are_rejected():
    """Pale noise alone on the glass anchors to nothing, so it stays out."""
    strong = np.zeros((30, 30), dtype=bool)
    strong[2:4, 2:4] = True
    weak = np.zeros((30, 30), dtype=bool)
    weak[2:4, 2:4] = True
    weak[20:25, 20:25] = True            # a separate pale island
    grown = label_and_grow(strong, weak)
    assert grown[22, 22] == 0


def test_two_fragments_stay_two_objects_when_they_grow_together():
    """
    The count of objects is fixed by the confident mask. Two fragments growing
    toward each other have to meet at a boundary, not merge -- otherwise one
    section and its neighbour become a single object and nothing downstream can
    tell them apart again.
    """
    strong = np.zeros((20, 60), dtype=bool)
    strong[9:11, 8:12] = True
    strong[9:11, 48:52] = True
    weak = np.zeros((20, 60), dtype=bool)
    weak[8:12, :] = True                 # a pale bridge across the whole width
    grown = label_and_grow(strong, weak)
    assert grown.max() == 2
    assert {1, 2} <= set(np.unique(grown).tolist())


def test_fill_holes_fills_a_lumen_but_not_the_slide():
    m = np.zeros((60, 60), dtype=bool)
    m[10:50, 10:50] = True
    m[28:32, 28:32] = False              # a small lumen, 16 cells
    assert fill_holes(m, max_cells=100)[30, 30]
    # With a cap below the hole's size it stays open.
    assert not fill_holes(m, max_cells=4)[30, 30]


def test_fill_holes_leaves_the_background_alone():
    m = np.zeros((40, 40), dtype=bool)
    m[5:15, 5:15] = True
    out = fill_holes(m, max_cells=10_000)
    assert not out[35, 35]               # outside touches the border


def test_a_blank_slide_finds_no_tissue():
    assert not tissue_mask(_canvas(), um_per_cell=30.0).any()


def test_a_stained_block_is_found():
    img = _canvas()
    img[30:90, 40:160] = (130, 70, 140)
    m = tissue_mask(img, um_per_cell=30.0)
    assert m[60, 100]
    assert not m[5, 5]


def test_a_scanner_seam_is_dropped_and_the_section_beside_it_is_not():
    """
    A seam runs the height of a scan at near-constant width; tissue does not.
    The section has to survive the filter that removes the seam.
    """
    img = _canvas(w=400, h=300)
    img[:, 5:8] = (90, 90, 95)           # full-height hairline
    img[80:220, 150:330] = (140, 75, 150)
    m = tissue_mask(img, um_per_cell=30.0)
    assert m[150, 240]                   # the section
    assert not m[150, 6]                 # the seam


@pytest.mark.parametrize("value", [255, 250, 240])
def test_glass_of_any_brightness_yields_nothing(value):
    """
    The levels are anchored on the span from glass up to Otsu precisely so a
    slide with no tissue does not have its own sensor noise promoted.
    """
    rng = np.random.default_rng(0)
    img = np.clip(
        np.full((120, 200, 3), value, dtype=np.int16)
        + rng.integers(-3, 4, (120, 200, 3)), 0, 255,
    ).astype(np.uint8)
    assert tissue_mask(img, um_per_cell=30.0).mean() < 0.02
