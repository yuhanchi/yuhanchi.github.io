"""Schematic: five routes that carry action-space value back to noise space."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROW_H = 1.18
NROW = 5
TOP = NROW * ROW_H + 0.55
fig = plt.figure(figsize=(7.0, 0.92 * TOP))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10); ax.set_ylim(0, TOP); ax.axis("off")

XL, XM, XR = 1.35, 5.0, 8.65       # column centres
WL, WM, WR = 2.3, 2.1, 2.3          # box widths
BH = 0.44

def box(x, y, w, text, fill=BG, edge=MUTED, color=TEXT, fs=12, ls="-", lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=lw, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)

def arrow(p, q, color=MUTED, lw=0.8, rad=0.0, ms=8, ls="-", z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))

def label(x, y, s, color=ACCENT, fs=11, ha="center", va="center", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, style=style)

# column headers
hy = TOP - 0.28
label(XL, hy, r"noise space $w$", color=TEXT, fs=12.5)
label(XM, hy, r"frozen decoder $\psi_\theta$", color=TEXT, fs=12.5)
label(XR, hy, r"action space $a$", color=TEXT, fs=12.5)
ax.plot([0.15, 9.85], [TOP - 0.52, TOP - 0.52], color=LINE, lw=0.9)

rows = [
    ("1", "Route 1:  noise critic", "DSRL-NA, LP-DS", ""),
    ("2", "Route 2:  inversion", "FlowDAgger, UniSteer, FRS", ""),
    ("3", "Route 3:  backpropagation through the sampler", "D-Flow", ""),
    ("4", "Route 4:  one-step decoder", "LPS, Noise Hypernetworks, ReNO", ""),
    ("5", "Route 5:  likelihood ratio", "Noise PPO, GoRL", ""),
]

def node(x, y, w, text, fs=12):
    box(x, y, w, text, fill=BG, edge=MUTED, color=TEXT, fs=fs)

def bypass(y_lane, label_text):
    # accent arrow from the action box, under the decoder, up into the noise box
    xs, xe = XR - WR / 2 + 0.35, XL + WL / 2 - 0.35
    ytop = y_lane + 0.18
    ax.plot([xs, xs, xe], [ytop, y_lane, y_lane], color=ACCENT, lw=1.3, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=4)
    arrow((xe, y_lane), (xe, ytop + 0.005), color=ACCENT, lw=1.3, ms=10, z=4)
    label(XM, y_lane - 0.15, label_text, fs=10.5)

def through(yc, label_text, y_label):
    arrow((XR - WR / 2, yc), (XL + WL / 2, yc), color=ACCENT, lw=1.3, ms=10, z=5)
    label(XM, y_label, label_text, fs=10.5)

def forward(yc):
    arrow((XL + WL / 2, yc), (XM - WM / 2, yc), ls=(0, (2, 2)))
    arrow((XM + WM / 2, yc), (XR - WR / 2, yc), ls=(0, (2, 2)))
    label((XL + WL / 2 + XM - WM / 2) / 2, yc + 0.13, r"$w$", color=MUTED, fs=11)
    label((XM + WM / 2 + XR - WR / 2) / 2, yc + 0.13, r"$a$", color=MUTED, fs=11)

for i, (num, name, who, price) in enumerate(rows):
    y0 = TOP - 0.55 - (i + 1) * ROW_H        # bottom of the row
    yt = y0 + ROW_H - 0.12                     # heading line
    yc = y0 + 0.64                            # node centre line
    if i > 0:
        ax.plot([0.15, 9.85], [y0 + ROW_H, y0 + ROW_H], color=LINE, lw=0.6)
    ax.text(0.18, yt, name, fontsize=12, color=TEXT, va="center", ha="left", fontweight="bold")
    ax.text(9.82, yt, who, fontsize=10.5, color=MUTED, va="center", ha="right", style="italic")
    y_low = yc - BH / 2 - 0.17

    if num == "1":
        node(XL, yc, WL, r"noise critic $Q^{W}(s,w)$", fs=11.5)
        box(XM, yc, WM, r"$\psi_\theta(s,\cdot)$", fill=DEC_FILL, edge=LIGHT)
        node(XR, yc, WR, r"action critic $Q^{A}(s,a)$", fs=11.5)
        forward(yc)
        bypass(y0 + 0.27, r"regress $Q^{W}(s,w)$ onto $Q^{A}(s,\psi_\theta(s,w))$")
    elif num == "2":
        node(XL, yc, WL, r"$w^\star \approx \psi_\theta^{-1}(s,a^\star)$")
        box(XM, yc, WM, "", fill=DEC_FILL, edge=LIGHT)
        label(XM, yc + 0.12, r"reverse ODE", color=TEXT, fs=11)
        node(XR, yc, WR, r"corrected action $a^\star$", fs=11.5)
        through(yc - 0.06, "actions become noise-space data", y_low)
    elif num == "3":
        node(XL, yc, WL, r"$\nabla_w Q$")
        K = 5; sw = WM / K
        for k in range(K):
            xk = XM - WM / 2 + sw * (k + 0.5)
            ax.add_patch(FancyBboxPatch((xk - sw / 2 + 0.04, yc - BH / 2), sw - 0.08, BH,
                         boxstyle="round,pad=0.0,rounding_size=0.05", fc=DEC_FILL, ec=LIGHT, lw=0.8))
        node(XR, yc, WR, r"$\nabla_a Q(s,a)$")
        through(yc, r"$\nabla_w Q=(\partial\psi_\theta/\partial w)^{\top}\nabla_a Q$, a product of $K$ step Jacobians", y_low)
    elif num == "4":
        node(XL, yc, WL, r"$\nabla_w Q$")
        box(XM, yc, 0.5, "", fill=DEC_FILL, edge=LIGHT)
        node(XR, yc, WR, r"$\nabla_a Q(s,a)$")
        through(yc, "one Jacobian: the reparameterization trick through a frozen decoder", y_low)
    elif num == "5":
        node(XL, yc, WL, r"$\nabla_\phi \log q_\phi(w\,|\,s)\cdot A$", fs=11.5)
        box(XM, yc, WM, r"$\psi_\theta(s,\cdot)$", fill=DEC_FILL, edge=LIGHT)
        node(XR, yc, WR, "return, advantage $A$", fs=11.5)
        forward(yc)
        bypass(y0 + 0.27, r"no gradient through $\psi_\theta$; only $\log q_\phi$ is needed")

save(fig, "five-routes")
