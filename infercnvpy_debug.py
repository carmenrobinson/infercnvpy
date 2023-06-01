import infercnvpy as cnv
import scanpy as sc
import anndata as ad

import itertools
import re
from multiprocessing import cpu_count
from typing import Sequence, Tuple, Union

import numpy as np
import scipy.ndimage
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from infercnvpy._util import _ensure_array


adata = ad.read("/Users/carmenrobinson/Documents/GitHub/infercnvpy/docs/notebooks/adata_snRNAseq2")
sc.pp.log1p(adata)

sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

var = adata.var.loc[:, ["chromosome", "start", "end"]]
var


def infercnv(
    adata: AnnData,
    *,
    reference_key: Union[str, None] = None,
    reference_cat: Union[None, str, Sequence[str]] = None,
    reference: Union[np.ndarray, None] = None,
    lfc_clip: float = 3,
    window_size: int = 100,
    step: int = 10,
    dynamic_threshold: Union[float, None] = 1.5,
    exclude_chromosomes: Union[Sequence[str], None] = ("chrX", "chrY"),
    chunksize: int = 5000,
    n_jobs: Union[int, None] = None,
    inplace: bool = True,
    layer: Union[str, None] = None,
    key_added: str = "cnv",
) -> Union[None, Tuple[dict, scipy.sparse.csr_matrix]]:
    """Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions.

    This method is heavily inspired by `infercnv <https://github.com/broadinstitute/inferCNV/>`_
    but more computationally efficient. The method is described in more detail
    in on the :ref:`infercnv-method` page.

    There, you can also find instructions on how to :ref:`prepare input data <input-data>`.

    Parameters
    ----------
    adata
        annotated data matrix
    reference_key
        Column name in adata.obs that contains tumor/normal annotations.
        If this is set to None, the average of all cells is used as reference.
    reference_cat
        One or multiple values in `adata.obs[reference_key]` that annotate
        normal cells.
    reference
        Directly supply an array of average normal gene expression. Overrides
        `reference_key` and `reference_cat`.
    lfc_clip
        Clip log fold changes at this value
    window_size
        size of the running window (number of genes in to include in the window)
    step
        only compute every nth running window where n = `step`. Set to 1 to compute
        all windows.
    dynamic_threshold
        Values `< dynamic threshold * STDDEV` will be set to 0, where STDDEV is
        the stadard deviation of the smoothed gene expression. Set to `None` to disable
        this step.
    exclude_chromosomes
        List of chromosomes to exclude. The default is to exclude genosomes.
    chunksize
        Process dataset in chunks of cells. This allows to run infercnv on
        datasets with many cells, where the dense matrix would not fit into memory.
    n_jobs
        Number of jobs for parallel processing. Default: use all cores.
        Data will be submitted to workers in chunks, see `chunksize`.
    inplace
        If True, save the results in adata.obsm, otherwise return the CNV matrix.
    layer
        Layer from adata to use. If `None`, use `X`.
    key_added
        Key under which the cnv matrix will be stored in adata if `inplace=True`.
        Will store the matrix in `adata.obsm["X_{key_added}"] and additional information
        in `adata.uns[key_added]`.

    Returns
    -------
    Depending on inplace, either return the smoothed and denoised gene expression
    matrix sorted by genomic position, or add it to adata.
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns) != set():
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and " "`end` columns in `adata.var`. "
        )

    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        logging.warning(
            f"Skipped {np.sum(var_mask)} genes because they don't have a genomic position annotated. "
        )  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)
    tmp_adata = adata[:, ~var_mask]

    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]

    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()

    reference = cnv.tl._get_reference(tmp_adata, reference_key, reference_cat, reference)

    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]  # type: ignore

    chr_pos, chunks = zip(
        *process_map(
            cnv.tl._infercnv._infercnv_chunk,
            [expr[i: i + chunksize, :] for i in range(0, adata.shape[0], chunksize)],
            itertools.repeat(var),
            itertools.repeat(reference),
            itertools.repeat(lfc_clip),
            itertools.repeat(window_size),
            itertools.repeat(step),
            itertools.repeat(dynamic_threshold),
            tqdm_class=tqdm,
            max_workers=cpu_count() if n_jobs is None else n_jobs,
        )
    )
    res = scipy.sparse.vstack(chunks)

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos[0]}

    else:
        return chr_pos[0], res

cnv.tl.infercnv(adata,reference_key="cell_type",reference_cat=["B-cells", "Endothelial cells"],window_size=100,step=1)
cnv.pl.chromosome_heatmap(adata, groupby="cell_type", dendrogram=True)
