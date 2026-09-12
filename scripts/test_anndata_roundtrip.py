"""
Check that the browser's AnnData export is readable by AnnData.

    SLIDECRAFT_ANNDATA_OUT=/tmp/sc.zarr.zip npx vitest run src/__tests__/anndata.test.ts
    .venv-export/bin/python -m pytest scripts/test_anndata_roundtrip.py -q

The TypeScript tests check the bytes against the spec as written down. This
checks them against the library that has to read them, which is a different
question and the one that matters: a layout can satisfy every assertion about
its own structure and still open as a bare zarr hierarchy instead of an
AnnData.

Skipped when the fixture has not been written, so the suite still runs without
a browser build.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

FIXTURE = Path(os.environ.get("SLIDECRAFT_ANNDATA_FIXTURE", "/tmp/sc-demo.zarr.zip"))

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}; see this file's docstring"
)


@pytest.fixture(scope="module")
def adata(tmp_path_factory):
    anndata = pytest.importorskip("anndata")
    out = tmp_path_factory.mktemp("zarr") / "store.zarr"
    with zipfile.ZipFile(FIXTURE) as z:
        # A corrupt central directory reads fine through some tools and not
        # others, so the archive is validated before anything is extracted.
        assert z.testzip() is None, "archive is damaged"
        z.extractall(out)
    return anndata.read_zarr(out)


def test_it_opens_as_an_anndata(adata):
    assert adata.shape == (7, 4)
    assert adata.X.dtype.name == "float32"


def test_the_matrix_is_row_major_over_patches(adata):
    import numpy as np
    # Patch i, gene g was written as i + g/10.
    np.testing.assert_allclose(np.asarray(adata.X)[2], [2.0, 2.1, 2.2, 2.3], rtol=1e-5)


def test_genes_and_patches_keep_their_names(adata):
    assert list(adata.var_names) == ["EPCAM", "MUC2", "COL1A1", "PTPRC"]
    # A patch is identified by where it is, so it survives concatenation.
    assert list(adata.obs_names)[:3] == ["0_50", "100_50", "200_50"]


def test_obs_carries_coordinates_and_the_tissue_mask(adata):
    assert list(adata.obs["x"])[:3] == [0, 100, 200]
    assert list(adata.obs["y"])[:3] == [50, 50, 50]
    assert list(adata.obs["in_tissue"])[:2] == [0, 1]


def test_spatial_holds_patch_centres(adata):
    import numpy as np
    assert adata.obsm["spatial"].shape == (7, 2)
    # Patch 0 is at (0, 50), side 100 — so its centre is (50, 100).
    np.testing.assert_allclose(adata.obsm["spatial"][0], [50, 100])


def test_provenance_travels_with_the_numbers(adata):
    def one(key):
        v = adata.uns[key]
        return str(v[0]) if hasattr(v, "__len__") and not isinstance(v, str) else str(v)

    assert one("model_id") == "deepspot-m-scgpt-4"
    assert one("slide") == "demo.svs"
    # These are predictions. The file has to say so wherever it ends up.
    assert "not a measurement" in one("provenance").lower()


def test_it_survives_the_usual_scanpy_pipeline(adata):
    sc = pytest.importorskip("scanpy")
    a = adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=3)
    assert "X_pca" in a.obsm


def test_it_can_be_written_back_out_as_h5ad(adata, tmp_path):
    out = tmp_path / "out.h5ad"
    adata.write_h5ad(out)
    assert out.stat().st_size > 0
