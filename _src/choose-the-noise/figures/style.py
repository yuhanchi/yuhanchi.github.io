"""Shared matplotlib style for the figures of "Choose the Noise, Keep the Flow".

Fonts: EB Garamond (SIL OFL), set FONT_DIR to a folder holding EBGaramond.ttf and
EBGaramond-Italic.ttf; otherwise matplotlib's default serif font is used.
Each figure is written as SVG and PNG for the post and as PDF for the LaTeX version.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_WEB = os.path.join(REPO, "assets", "img", "posts", "choose-the-noise")
OUT_PDF = os.path.join(HERE, "..", "paper", "figures")
FONT_DIR = os.environ.get("FONT_DIR", "/tmp/choose-noise-figs/downloads/ebgaramond")

family = "serif"
if os.path.exists(os.path.join(FONT_DIR, "EBGaramond.ttf")):
    for f in ("EBGaramond.ttf", "EBGaramond-Italic.ttf"):
        fm.fontManager.addfont(os.path.join(FONT_DIR, f))
    family = "EB Garamond"

BG = "#ffffff"
TEXT = "#1f2a3a"
MUTED = "#5d6675"
LINE = "#d9d6ce"
NAVY = "#2c4a6e"
ACCENT = "#b0562f"   # value / reward
LIGHT = "#c9d3df"
DEC_FILL = "#eef1f5"

plt.rcParams.update({
    "font.family": family,
    "font.size": 11,
    "mathtext.fontset": "custom",
    "mathtext.rm": family,
    "mathtext.it": f"{family}:italic",
    "mathtext.bf": family,
    "mathtext.cal": "cmsy10",
    "mathtext.fallback": "stix",
    "text.color": TEXT,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
})

def panel_label(ax_or_fig, x, y, s, **kw):
    """Bold panel tag such as (a), placed in figure or axes coordinates."""
    tgt = ax_or_fig
    tr = tgt.transAxes if hasattr(tgt, "transAxes") else tgt.transFigure
    tgt.text(x, y, s, transform=tr, fontsize=12, fontweight="bold", color=TEXT,
             ha="left", va="bottom", **kw)

def save(fig, name):
    os.makedirs(OUT_WEB, exist_ok=True); os.makedirs(OUT_PDF, exist_ok=True)
    fig.savefig(os.path.join(OUT_WEB, f"{name}.svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT_WEB, f"{name}.png"), dpi=200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT_PDF, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
