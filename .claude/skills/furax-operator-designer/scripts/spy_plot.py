"""Render a dense matrix as a structure ("spy") plot for furax operator design docs.

Importable:
    from spy_plot import spy
    spy(op.as_matrix(), "figs/structure.png", title="P (pointing)")

CLI:
    python spy_plot.py matrix.npy figs/structure.png --title "P (pointing)"

The plot is value-aware: nonzero entries are colored on a diverging colormap
centered at zero (so sign is visible), exact zeros are left blank so the sparsity
pattern reads at a glance, and small matrices get gridlines and value annotations.

It is intentionally decoupled from furax: it takes any array-like, so the same
helper renders ``op.as_matrix()``, a hand-built reference, or their difference.
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib

matplotlib.use('Agg')  # headless: doc generation has no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def spy(
    matrix: Any,
    path: str | None = None,
    *,
    title: str | None = None,
    ax: Any = None,
    annotate: bool | str = 'auto',
    annotate_limit: int = 12,
    cmap: str = 'RdBu_r',
    dpi: int = 130,
) -> Any:
    """Draw the structure of a 2-D matrix and optionally save it.

    Args:
        matrix: Any array-like; converted with ``np.asarray``. Must be 2-D.
        path: If given (and ``ax`` is None), the figure is saved here.
        title: Optional title drawn above the plot.
        ax: Draw into an existing Axes instead of creating a figure.
        annotate: Write each entry's value in its cell. ``"auto"`` does so only
            when both dimensions are <= ``annotate_limit`` (keeps big matrices legible).
        annotate_limit: Size threshold for ``annotate="auto"``.
        cmap: Diverging colormap; the scale is centered at zero so sign is visible.
        dpi: Resolution when saving.

    Returns:
        The Axes the matrix was drawn into.
    """
    m = np.asarray(matrix)
    if m.ndim != 2:
        raise ValueError(f'spy expects a 2-D matrix, got shape {m.shape}.')
    m = m.astype(float)
    n_rows, n_cols = m.shape

    # Center the colormap at zero and leave exact zeros blank so sparsity reads.
    vmax = float(np.max(np.abs(m))) or 1.0
    masked = np.ma.masked_where(m == 0.0, m)

    created = ax is None
    if created:
        w = float(np.clip(n_cols * 0.45 + 1.5, 3.0, 12.0))
        h = float(np.clip(n_rows * 0.45 + 1.2, 2.5, 12.0))
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')  # zeros -> blank
    im = ax.imshow(masked, cmap=cmap_obj, vmin=-vmax, vmax=vmax, aspect='equal')

    do_annot = annotate is True or (
        annotate == 'auto' and n_rows <= annotate_limit and n_cols <= annotate_limit
    )
    if do_annot:
        for i in range(n_rows):
            for j in range(n_cols):
                v = m[i, j]
                if v == 0.0:
                    continue
                ax.text(
                    j,
                    i,
                    f'{v:g}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black' if abs(v) < 0.6 * vmax else 'white',
                )

    if max(n_rows, n_cols) <= 40:  # cell gridlines only stay legible when small
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='0.85', linewidth=0.5)
        ax.tick_params(which='minor', length=0)

    ax.set_xlabel('input index (column)')
    ax.set_ylabel('output index (row)')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='entry value')

    if path is not None and created:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return ax


def _main() -> None:
    p = argparse.ArgumentParser(description='Render a dense matrix (.npy) as a structure plot.')
    p.add_argument('matrix', help='Path to a .npy file holding a 2-D array.')
    p.add_argument('output', help='Output image path (e.g. figs/structure.png).')
    p.add_argument('--title', default=None)
    p.add_argument('--annotate', choices=['auto', 'always', 'never'], default='auto')
    args = p.parse_args()
    annotate: bool | str = {'auto': 'auto', 'always': True, 'never': False}[args.annotate]
    spy(np.load(args.matrix), args.output, title=args.title, annotate=annotate)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    _main()
