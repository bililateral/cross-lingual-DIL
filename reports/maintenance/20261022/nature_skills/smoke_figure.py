"""Handwritten tool fixture only: these points are not research observations."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.83, wspace=0.30)
x = [1, 2, 3, 4]
for ax, y, label, title, color in zip(
    axes, ([1, 2, 3, 4], [4, 3, 2, 1]), ("a", "b"),
    ("Increasing fixture", "Decreasing fixture"), ("#4477AA", "#CC6677")
):
    ax.plot(x, y, marker="o", markersize=3, color=color)
    ax.set(xlim=(0.7, 4.3), ylim=(0.5, 4.5), xticks=x, yticks=x,
           xlabel="Input index", ylabel="Fixture value", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.13, 1.08, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom")
fig.text(0.5, 0.06, "TOOL CHECK ONLY — handwritten inputs; no experimental data",
         ha="center", fontsize=6)
fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig, json_out=OUT / "figure.alignment.json",
    overlay_svg=OUT / "figure.alignment.svg", require_panel_labels=True,
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
)
fig.savefig(OUT / "figure.pdf")
fig.savefig(OUT / "figure.svg")
fig.savefig(OUT / "figure.png", dpi=600)
fig.savefig(OUT / "figure.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
