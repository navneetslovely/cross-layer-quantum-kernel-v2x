"""Publication plotting themes for the qkv2x papers (paper).

Three colour-blind-aware palettes are provided; pick one and call ``apply()`` at the top
of any plotting script to set a consistent, publication-friendly look (font sizes, clean
spines, grid, 300-dpi vector-friendly output, and the chosen colour cycle).

    from qkv2x.utils.style import apply, PALETTES, CMAP
    apply("okabe_ito")                 # sets rcParams (colour cycle + look)
    colors = PALETTES["okabe_ito"]["colors"]
    cmap   = CMAP["okabe_ito"]         # sequential cmap for scatter colour-by-value

Palettes
--------
- ``okabe_ito``    : the de-facto journal standard; 8-colour, fully colour-blind-safe.
- ``tol_bright``   : Paul Tol's "bright" set; colour-blind-safe, a touch more saturated.
- ``brand_indigo`` : cohesive set anchored on the project's existing #534AB7 accent.
"""
from __future__ import annotations

PALETTES = {
    "okabe_ito": {
        "colors": ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                   "#E69F00", "#56B4E9", "#F0E442", "#000000"],
        "grey": "#999999",
        "desc": "Okabe-Ito — journal-standard, fully colour-blind-safe (recommended).",
    },
    "tol_bright": {
        "colors": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"],
        "grey": "#BBBBBB",
        "desc": "Paul Tol 'bright' — colour-blind-safe, modern/saturated.",
    },
    "brand_indigo": {
        "colors": ["#534AB7", "#EF6F6C", "#2EC4B6", "#FFB703", "#5C6B8A", "#B5179E"],
        "grey": "#8A8A8A",
        "desc": "Brand indigo — cohesive with the existing #534AB7 figures.",
    },
}

# perceptually-uniform, CVD-safe sequential maps (for scatter colour-by-ROC etc.)
CMAP = {"okabe_ito": "cividis", "tol_bright": "viridis", "brand_indigo": "viridis"}


def apply(name: str = "okabe_ito", *, dpi: int = 300, base_fontsize: int = 10):
    """Set matplotlib rcParams for a clean, publication-ready look. figsize is left to
    each figure (IEEE single column ~3.5in, double ~7.16in)."""
    import matplotlib as mpl
    from cycler import cycler
    if name not in PALETTES:
        raise ValueError(f"unknown palette {name!r}; choose from {list(PALETTES)}")
    colors = PALETTES[name]["colors"]
    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=colors),
        "figure.dpi": 120, "savefig.dpi": dpi, "savefig.bbox": "tight",
        "font.size": base_fontsize, "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize, "legend.fontsize": base_fontsize - 2,
        "xtick.labelsize": base_fontsize - 1, "ytick.labelsize": base_fontsize - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "legend.frameon": False, "figure.autolayout": True,
    })
    return colors
