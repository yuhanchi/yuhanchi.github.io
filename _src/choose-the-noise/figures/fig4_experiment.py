"""Figures 4 and 5: the trained 2D experiment (see ../experiment/noise_routes.py)."""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment"))
from style import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import torch
import toy
from toy import psi, student, Q, M, S, goal, dev

RES = os.path.join(HERE, "..", "experiment", "results")
toy.load(os.path.join(RES, "decoders.pt"))
data = json.load(open(os.path.join(RES, "results.json")))
EVAL_PHI = np.array(data["log"]["eval_phi"])
IDX = int(os.environ.get("STATE_IDX", 9))
phi = float(EVAL_PHI[IDX])
print(f"display state: phi = {math.degrees(phi):.2f} deg")
BETA = toy.BETA
rng = torch.Generator(device=dev).manual_seed(0)

cm = LinearSegmentedColormap.from_list("n", ["#f3f5f8", "#c2cedb", "#7489a4"])
LEV = [.01, .04, .12, .3, .6, 1.01]
COLS = [cm(x) for x in np.linspace(0, 1, len(LEV) - 1)]
L, LA = 3.4, (-3.4, 3.4)

# noise grid: exact tilted posterior for the learned decoder
g = np.linspace(-L, L, 201)
WX, WY = np.meshgrid(g, g)
Wg = torch.tensor(np.stack([WX.ravel(), WY.ravel()], 1), dtype=torch.float32, device=dev)
with torch.no_grad():
    Ag = {"psi": psi(Wg), "student": student(Wg)}
def qstar(dec):
    lq = -0.5 * (WX ** 2 + WY ** 2) + (Q(torch.full((len(Wg),), phi, device=dev), Ag[dec]) / BETA).cpu().numpy().reshape(WX.shape)
    q = np.exp(lq - lq.max()); return q / q.max()
mode_lab = ((Ag["psi"][:, None] - M[None]) ** 2).sum(-1).argmin(1).cpu().numpy().reshape(WX.shape)

# action grid: tilted mixture the flow was trained to imitate
ga = np.linspace(*LA, 201)
AX, AY = np.meshgrid(ga, ga)
Agr = np.stack([AX.ravel(), AY.ravel()], 1)
Mn = M.cpu().numpy(); gl = goal(torch.tensor(phi)).numpy()
base = sum(np.exp(-0.5 * ((Agr - m) ** 2).sum(1) / S ** 2) for m in Mn)
tilt = (base * np.exp(-0.5 * ((Agr - gl) ** 2).sum(1) / BETA)).reshape(AX.shape); tilt /= tilt.max()
Qgrid = -0.5 * ((Agr - gl) ** 2).sum(1).reshape(AX.shape)

def style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color(LINE); sp.set_linewidth(0.8)

def draw_pair(ax_w, ax_a, mu, ls, dec="psi", n=500, shade=True):
    mu = np.array(mu); sig = np.exp(np.array(ls))
    w = torch.tensor(mu, device=dev) + torch.tensor(sig, device=dev) * torch.randn(n, 2, device=dev, generator=rng)
    with torch.no_grad():
        a = (psi(w.float()) if dec == "psi" else student(w.float())).cpu().numpy()
    w = w.cpu().numpy()
    ax_w.contourf(WX, WY, qstar(dec), levels=LEV, colors=COLS)
    for k in range(4):
        ax_w.contour(WX, WY, (mode_lab == k).astype(float), levels=[0.5], colors="#98a1ae", linewidths=0.5)
    ax_w.scatter(w[:, 0], w[:, 1], s=1.4, color=TEXT, lw=0, alpha=0.55, rasterized=True)
    ax_w.add_patch(Ellipse(mu, 4 * sig[0], 4 * sig[1], fill=False, ec=ACCENT, lw=1.0))
    ax_w.set_xlim(-L, L); ax_w.set_ylim(-L, L)
    ax_a.contourf(AX, AY, tilt, levels=LEV, colors=COLS)
    ax_a.contour(AX, AY, Qgrid, levels=[-4, -2, -0.5], colors=ACCENT, linewidths=0.6,
                 linestyles=[(0, (3, 2))], alpha=0.8)
    ax_a.plot(*gl, marker="*", color=ACCENT, ms=7, mec="none")
    ax_a.scatter(a[:, 0], a[:, 1], s=1.4, color=TEXT, lw=0, alpha=0.55, rasterized=True)
    ax_a.set_xlim(*LA); ax_a.set_ylim(*LA)
    for ax in (ax_w, ax_a):
        style_ax(ax)

def get(route, seed=0):
    return next(r for r in data["results"] if r["route"] == route and r["seed"] == seed)

def arrows(fig, axes, label_first=True):
    for j in range(axes.shape[1]):
        b0 = axes[0, j].get_position(); b1 = axes[1, j].get_position()
        x = b0.x0 + b0.width / 2
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (x, b0.y0 - 0.01), (x, b1.y1 + 0.01), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.7))
        if j == 0 and label_first:
            fig.text(x + 0.01, (b0.y0 + b1.y1) / 2, r"frozen $\psi_\theta$", fontsize=10,
                     color=MUTED, va="center", ha="left")

import matplotlib
# ---------------------------------------------------------------- Figure 4: training snapshots (R3)
r3 = get("R3")
its = sorted(int(k) for k in r3["snapshots"])
fig, axes = plt.subplots(2, len(its), figsize=(7.0, 3.35))
fig.subplots_adjust(left=0.06, right=0.995, top=0.91, bottom=0.02, wspace=0.06, hspace=0.32)
for j, it in enumerate(its):
    mu, ls = r3["snapshots"][str(it)]
    draw_pair(axes[0, j], axes[1, j], mu[IDX], ls[IDX])
    axes[0, j].set_title("initial" if it == 0 else f"update {it}", fontsize=11.5, pad=3)
axes[0, 0].set_ylabel(r"noise $w$", fontsize=11.5)
axes[1, 0].set_ylabel(r"action $a$", fontsize=11.5)
arrows(fig, axes)
save(fig, "experiment-training")

# ---------------------------------------------------------------- Figure 5: five routes, final proposers + curves
routes = ["R1", "R2", "R3", "R4", "R5"]
names = ["1: noise critic", "2: inversion", "3: backprop", "4: one-step", "5: likelihood ratio"]
fig = plt.figure(figsize=(7.0, 5.2))
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15], left=0.075, right=0.995, top=0.93,
                      bottom=0.085, wspace=0.06, hspace=0.36)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(2)])
for j, rt in enumerate(routes):
    r = get(rt)
    mu, ls = r["snapshots"][str(max(int(k) for k in r["snapshots"]))]
    draw_pair(axes[0, j], axes[1, j], mu[IDX], ls[IDX], dec="student" if rt == "R4" else "psi")
    axes[0, j].set_title(f"Route {j + 1}", fontsize=11.5, pad=3)
axes[0, 0].set_ylabel(r"noise $w$", fontsize=11.5)
axes[1, 0].set_ylabel(r"action $a$", fontsize=11.5)
arrows(fig, axes, label_first=False)
axc = fig.add_subplot(gs[2, :])
COL = {"R1": "#2c4a6e", "R2": "#8a8f98", "R3": "#b0562f", "R4": "#d49a3a", "R5": "#5f8f6e"}
for j, rt in enumerate(routes):
    cs = np.array([[g_ for _, g_ in r["curve"]] for r in data["results"] if r["route"] == rt])
    xs = np.array([i for i, _ in get(rt)["curve"]])
    m = cs.mean(0)
    axc.plot(xs, m, color=COL[rt], lw=1.3, label="Route " + names[j])
    axc.fill_between(xs, cs.min(0), cs.max(0), color=COL[rt], alpha=0.15, lw=0)
floor = json.load(open(os.path.join(RES, "gaussian_floor.json")))["gap_mean"]
axc.axhline(floor, color=TEXT, lw=0.8, ls=(0, (2, 2)), label="best Gaussian per state")
axc.set_yscale("log")
axc.set_yticks([0.2, 0.5, 1, 2]); axc.set_yticklabels(["0.2", "0.5", "1", "2"])
axc.minorticks_off()
axc.set_xlabel("proposer update", fontsize=11)
axc.set_ylabel(r"$\mathrm{KL}(q_\phi\,\|\,q^\star)$", fontsize=11)
axc.spines[["top", "right"]].set_visible(False)
axc.legend(frameon=False, fontsize=9.5, ncol=2, borderaxespad=0.2, loc="upper right", handlelength=1.4, columnspacing=1.2)
axc.set_ylim(0.15, 4)
b = axes[0, 0].get_position(); panel_label(fig, 0.005, b.y1 + 0.012, "(a)")
b = axc.get_position(); panel_label(fig, 0.005, b.y1 + 0.012, "(b)")
axc.set_xlim(0, xs.max())
save(fig, "experiment-routes")

# ---------------------------------------------------------------- numbers for Table 3
for rt in routes:
    rs = [r for r in data["results"] if r["route"] == rt]
    gap = np.array([r["gap"] for r in rs]); ms = np.array([r["ms_per_update"] for r in rs])
    gi = np.array([r["gap_per_state"][IDX] for r in rs])
    print(f"{rt}: gap {gap.mean():.3f} ± {gap.std():.3f}   gap@state {gi.mean():.3f}   "
          f"E[Q]/b {np.mean([r['EQ_over_beta'] for r in rs]):.3f}  KL {np.mean([r['KL'] for r in rs]):.3f}  "
          f"{ms.mean():.1f} ms/update")
