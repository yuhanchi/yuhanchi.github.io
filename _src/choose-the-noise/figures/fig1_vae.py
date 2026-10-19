"""VAE correspondence: encoder -> latent -> decoder, for a VAE and for noise-space RL."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig = plt.figure(figsize=(7.0, 2.45))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10); ax.set_ylim(0.2, 3.7); ax.axis("off")

def box(x, y, w, h, top, sub, fill=BG, edge=MUTED, lw=0.8, ls="-", tc=TEXT):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=lw, ls=ls))
    ax.text(x, y + 0.12, top, ha="center", va="center", fontsize=12, color=tc)
    ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=10, color=MUTED, style="italic")

def arrow(p, q, color=MUTED, lw=0.9, ms=9, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", linestyle=ls))

def t(x, y, s, fs=12, color=TEXT, ha="center", style="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, style=style)

X_IN, X_ENC, X_LAT, X_DEC, X_OUT = 0.45, 2.1, 3.85, 5.6, 8.5
W_ENC, W_DEC, W_OUT = 1.85, 1.95, 2.75
H = 0.62
rows = {
    "vae": dict(y=2.92, name="(a)  VAE",
                inp=r"$x$", enc=(r"encoder $q_\phi(z\,|\,x)$", "trained"),
                lat=r"$z$", dec=(r"decoder $p_\theta(x\,|\,z)$", "trained jointly"),
                out=(r"$\log p_\theta(x\,|\,z)$", "reconstruction")),
    "rl": dict(y=1.02, name="(b)  Noise-space RL",
               inp=r"$s$", enc=(r"proposer $q_\phi(w\,|\,s)$", "small, trained by RL"),
               lat=r"$w$", dec=(r"flow $\psi_\theta(s,w)$", "pretrained, frozen"),
               out=(r"$Q(s,a)\,/\,\beta$", "return replaces reconstruction")),
}
for key, r in rows.items():
    y = r["y"]
    ax.text(0.12, y + 0.62, r["name"], ha="left", va="center", fontsize=12, color=TEXT, fontweight="bold")
    t(X_IN, y, r["inp"], fs=14)
    frozen = key == "rl"
    box(X_ENC, y, W_ENC, H, *r["enc"])
    ax.add_patch(Circle((X_LAT, y), 0.25, fc=BG, ec=MUTED, lw=0.8))
    t(X_LAT, y, r["lat"], fs=14)
    box(X_DEC, y, W_DEC, H, *r["dec"], fill=DEC_FILL if frozen else BG, edge=LIGHT if frozen else MUTED)
    box(X_OUT, y, W_OUT, H, *r["out"], edge=ACCENT, tc=ACCENT)
    arrow((X_IN + 0.18, y), (X_ENC - W_ENC / 2, y))
    arrow((X_ENC + W_ENC / 2, y), (X_LAT - 0.25, y))
    arrow((X_LAT + 0.25, y), (X_DEC - W_DEC / 2, y))
    arrow((X_DEC + W_DEC / 2, y), (X_OUT - W_OUT / 2, y))
    if key == "rl":
        t((X_DEC + W_DEC / 2 + X_OUT - W_OUT / 2) / 2, y + 0.13, r"$a$", fs=11.5, color=MUTED)
        # the state also conditions the decoder
        ax.plot([X_IN, X_IN, X_DEC - 0.2], [y - 0.18, y - 0.62, y - 0.62], color=MUTED, lw=0.8)
        arrow((X_DEC - 0.2, y - 0.62), (X_DEC - 0.2, y - H / 2 - 0.02))
# shared prior between the two rows
ym = (rows["vae"]["y"] + rows["rl"]["y"]) / 2
t(X_LAT, ym, r"prior $\mathcal{N}(0,I)$", fs=11.5, color=MUTED)
for yy, lab in ((rows["vae"]["y"], r"$-\mathrm{KL}$"), (rows["rl"]["y"], r"$-\mathrm{KL}$")):
    sgn = 1 if yy > ym else -1
    y_a, y_b = ym + sgn * 0.17, yy - sgn * 0.25
    ax.plot([X_LAT, X_LAT], [y_a, y_b], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    t(X_LAT + 0.1, (y_a + y_b) / 2, lab, fs=10.5, color=MUTED, ha="left")
save(fig, "vae-correspondence")
