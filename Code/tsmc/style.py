"""Figure style: every temporal plot shows values only at points of the time scale.

Scattered points are markers (and stems for graininess); curves appear only over dense
intervals of T, which are additionally shaded.  Figures are drawn at the text width of the
manuscript (160 mm = 6.3 in), so the font sizes below are the printed sizes.

Colours: an 8-hue categorical palette validated on a white surface (lightness band, chroma
floor, CVD separation of adjacent slots, normal-vision floor, contrast >= 3:1 for every slot)
and a perceptually uniform colour map for graininess.  Output: vector PDF and a high-resolution
PNG (PNG_DPI below).
"""
import logging
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.transforms import Bbox

FIG_W = 6.3                 # text width of the manuscript in inches
PNG_DPI = 1000              # PNG resolution (line-art standard)

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#199e70', '#c98500'
MAGENTA, GREEN, VIOLET, RED = '#d55181', '#008300', '#6250d6', '#e34948'
INK, INK2, MUTED, GRID, AXIS = '#111111', '#3a3936', '#77756f', '#e2e1db', '#3a3936'
AQUA_DARK, YELLOW_DARK, MAGENTA_DARK = '#117a55', '#a46c00', '#b23a69'
DENSE = '#199e70'           # dense components of T
DENSE_SHADE = '#dcf2e8'     # background tint of dense components


def _truncate(name, lo, hi, n=256):
    base = mpl.colormaps[name]
    return LinearSegmentedColormap.from_list(f'{name}_{lo}_{hi}', base(np.linspace(lo, hi, n)))


GRAIN_CMAP = _truncate('plasma', 0.0, 0.8)       # graininess (stops at orange: >= 3:1 on white)
LEVEL_CMAP = _truncate('viridis', 0.0, 0.9)      # generation / index
TIME_CMAP = _truncate('turbo', 0.08, 0.92)       # memory rays


def setup():
    logging.getLogger('fontTools').setLevel(logging.ERROR)   # silence font-subsetting notices
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
        'font.size': 9.2, 'axes.titlesize': 9.6, 'axes.labelsize': 9.6, 'legend.fontsize': 8.2,
        'xtick.labelsize': 8.6, 'ytick.labelsize': 8.6,
        'axes.edgecolor': AXIS, 'axes.linewidth': .85, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.major.width': .85, 'ytick.major.width': .85, 'xtick.minor.width': .6, 'ytick.minor.width': .6,
        'xtick.major.size': 3.4, 'ytick.major.size': 3.4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 3,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .55, 'grid.alpha': 1.0,
        'axes.axisbelow': True, 'legend.frameon': True, 'legend.framealpha': .96,
        'legend.edgecolor': '#cfcec7', 'legend.fancybox': True, 'legend.borderpad': .45,
        'legend.handletextpad': .45, 'legend.labelspacing': .35,
        'lines.linewidth': 1.8, 'lines.markersize': 6, 'patch.linewidth': .6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': PNG_DPI, 'lines.solid_capstyle': 'round',
    })


def full_width_bbox(fig, pad=.03):
    """Tight bounding box of the drawn content (inches), widened symmetrically to exactly FIG_W.

    Every figure is included in LaTeX at the text width FIG_W, so a file of width FIG_W is
    reproduced at scale 1: fonts and line widths appear at their design size in every figure."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, x1, y1 = tb.x0 - pad, tb.y0 - pad, tb.x1 + pad, tb.y1 + pad
    extra = FIG_W - (x1 - x0)
    if extra < -1e-9:
        raise ValueError(f'figure content is {x1 - x0:.3f} in wide; at most FIG_W = {FIG_W} in is allowed')
    return Bbox([[x0 - extra / 2, y0], [x1 + extra / 2, y1]])


def save(fig, name, root):
    """Write Figures/<name>.pdf (vector) and Figures/<name>.png (PNG_DPI), both FIG_W wide, atomically."""
    out = Path(root) / 'Figures'
    out.mkdir(exist_ok=True)
    bbox = full_width_bbox(fig)
    pdf_tmp = out / f'{name}.pdf.part'
    fig.savefig(pdf_tmp, format='pdf', bbox_inches=bbox, metadata={'CreationDate': None, 'ModDate': None})
    pdf_tmp.replace(out / f'{name}.pdf')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=PNG_DPI, bbox_inches=bbox)
    buf.seek(0)
    png_tmp = out / f'{name}.png.part'
    # opaque RGB on white (the alpha channel carries no information) with maximal lossless compression
    Image.open(buf).convert('RGB').save(png_tmp, format='PNG', optimize=True, dpi=(PNG_DPI, PNG_DPI))
    png_tmp.replace(out / f'{name}.png')
    plt.close(fig)


def label(ax, text, x=-.02, y=1.02):
    """Panel letter such as (a), set in bold Computer Modern (mathtext)."""
    ax.text(x, y, r'$\mathbf{%s}$' % text, transform=ax.transAxes, fontsize=10.5, va='bottom', ha='right',
            color=INK)


def shade_dense(ax, intervals, color=DENSE_SHADE):
    for a, b in intervals:
        ax.axvspan(a, b, color=color, lw=0, zorder=0)


def barcode(ax, rs_t, rs_mu, intervals=(), norm=None, cmap=GRAIN_CMAP, height=.8, lw=1.6,
            special=None, special_label=None, ylabel=r'$\mathbb{T}$', marker_size=None):
    """Time-scale strip: dense intervals as thick bars, right-scattered points as ticks coloured by
    their graininess, optional special points (accumulation / left-dense) as diamonds."""
    rs_t = np.asarray(rs_t, float)
    rs_mu = np.asarray(rs_mu, float)
    if norm is None and len(rs_mu):
        norm = Normalize(vmin=float(rs_mu.min()), vmax=float(rs_mu.max()) if rs_mu.max() > rs_mu.min() else float(rs_mu.min()) + 1)
    for a, b in intervals:
        ax.plot([a, b], [0, 0], color=DENSE, lw=7.5, solid_capstyle='butt', zorder=2)
    if len(rs_t):
        ax.vlines(rs_t, -height / 2, height / 2, colors=cmap(norm(rs_mu)), lw=lw, zorder=3)
    if special is not None and len(special):
        ax.scatter(special, np.zeros(len(special)), marker='D', s=marker_size or 30, color=ORANGE,
                   edgecolor='white', linewidth=.6, zorder=4, label=special_label)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, rotation=0, labelpad=10, va='center')
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    return norm


def stems(ax, t, y, colors, lw=1.3, ms=22, base=0.0, zorder=3, marker='o', label=None, alpha=1.0):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ax.vlines(t, base, y, colors=colors, lw=lw, zorder=zorder, alpha=alpha)
    ax.scatter(t, y, s=ms, c=colors, marker=marker, zorder=zorder + 1, linewidths=0, label=label, alpha=alpha)


def points(ax, t, y, color=BLUE, size=26, marker='o', label=None, zorder=4, edge='white', lw=.5, **kw):
    return ax.scatter(t, y, s=size, color=color, marker=marker, label=label, zorder=zorder,
                      edgecolors=edge, linewidths=lw, **kw)


def colorbar(fig, ax, cmap, norm, text, orientation='vertical', **kw):
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation=orientation, **kw)
    cb.set_label(text)
    cb.outline.set_linewidth(.6)
    cb.ax.tick_params(labelsize=8.2)
    return cb


__all__ = ['setup', 'save', 'label', 'barcode', 'stems', 'points', 'shade_dense', 'colorbar',
           'GRAIN_CMAP', 'LEVEL_CMAP', 'TIME_CMAP', 'LogNorm', 'Normalize', 'FIG_W', 'PNG_DPI']
