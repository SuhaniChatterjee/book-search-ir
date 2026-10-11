"""Shared matplotlib style for report figures."""
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, one per model
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#ffffff"


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
