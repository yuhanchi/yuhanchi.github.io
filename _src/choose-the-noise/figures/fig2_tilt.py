"""Tilting the noise tilts the policy: 2D toy with an exact flow-matching ODE map.

Base policy: 4-component GMM on a ring.  The flow psi is the probability-flow ODE of
linear-interpolation flow matching from N(0,I) to that GMM, whose marginal velocity
field has a closed form.  We integrate it with RK4, so psi_# N(0,I) ~= GMM.
For each beta, the tilted noise q*(w) ∝ N(w;0,I) exp(Q(psi(w))/beta) is computed
exactly on a grid (density) and sampled by importance resampling from N(0,I).
Its pushforward is compared with pi_base(a) exp(Q(a)/beta) computed analytically.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(0)
R, S = 2.2, 0.38
ang = np.deg2rad([20, 110, 200, 290])
M = np.stack([R * np.cos(ang), R * np.sin(ang)], 1)        # mode means
PI = np.full(4, 0.25)
G = np.array([2.9, 1.6])                                    # Q peak
def Q(a):
    return -0.125 * np.sum((a - G) ** 2, -1)

def velocity(x, t):
    s2t = (1 - t) ** 2 + t ** 2 * S ** 2
    c = (t * S ** 2 - (1 - t)) / s2t
    d = x[:, None, :] - t * M[None]                         # N,K,2
    logw = np.log(PI)[None] - 0.5 * np.sum(d ** 2, -1) / s2t
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    u = M[None] + c * d                                      # N,K,2
    return np.einsum("nk,nkd->nd", w, u)

def psi(x, steps=200):
    x = x.copy(); h = 1.0 / steps
    for i in range(steps):
        t = i * h
        k1 = velocity(x, t); k2 = velocity(x + .5 * h * k1, t + .5 * h)
        k3 = velocity(x + .5 * h * k2, t + .5 * h); k4 = velocity(x + h * k3, t + h)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

def gmm_pdf(a):
    d = a[..., None, :] - M
    return np.sum(PI * np.exp(-0.5 * np.sum(d ** 2, -1) / S ** 2) / (2 * np.pi * S ** 2), -1)

# --- samples from the base noise and their decoded actions
N = 400_000
import os
CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "choose-the-noise-fig2-cache.npz")
if os.path.exists(CACHE):
    _c = np.load(CACHE); W, A, Ag_cached = _c["W"], _c["A"], _c["Ag"]
else:
    W = rng.standard_normal((N, 2))
    A = np.concatenate([psi(W[i:i + 50_000]) for i in range(0, N, 50_000)])
    Ag_cached = None
QA = Q(A)

# --- grids
L = 3.4
g = np.linspace(-L, L, 241)
WX, WY = np.meshgrid(g, g)
Wg = np.stack([WX.ravel(), WY.ravel()], 1)
Ag = psi(Wg) if Ag_cached is None else Ag_cached
if Ag_cached is None: np.savez(CACHE, W=W, A=A, Ag=Ag)
QWg = Q(Ag).reshape(WX.shape)
logN = -0.5 * (WX ** 2 + WY ** 2)
mode_lab = np.argmin(((Ag[:, None, :] - M) ** 2).sum(-1), 1).reshape(WX.shape)

ga = np.linspace(-3.6, 4.0, 241)
AX, AY = np.meshgrid(ga, ga)
Agrid = np.stack([AX.ravel(), AY.ravel()], 1)
base_a = gmm_pdf(Agrid).reshape(AX.shape)
Qa = Q(Agrid).reshape(AX.shape)

# --- sanity: pushforward of N(0,I) vs GMM (mode masses)
lab = np.argmin(((A[:, None, :] - M) ** 2).sum(-1), 1)
print("base mode masses (pushforward):", np.bincount(lab, minlength=4) / N)

betas = [np.inf, 4.0, 1.0, 0.1]
cm_noise = LinearSegmentedColormap.from_list("n", ["#f0f2f6", "#c2cedb", "#7489a4"])
LEV = [.01, .04, .12, .3, .6, 1.01]
COLS = [cm_noise(x) for x in np.linspace(0, 1, len(LEV) - 1)]

fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.95))
fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.03, wspace=0.06, hspace=0.30)
n_show = 650
for j, beta in enumerate(betas):
    # exact tilted noise density on grid
    lq = logN + (0 if np.isinf(beta) else QWg / beta)
    q = np.exp(lq - lq.max()); q /= q.sum()
    # exact tilted target in action space
    lt = np.log(base_a + 1e-300) + (0 if np.isinf(beta) else Qa / beta)
    tgt = np.exp(lt - lt.max()); tgt /= tgt.sum()
    # importance resampling from N(0,I): q* ∝ N · exp(Q∘psi/β)
    lw = np.zeros(N) if np.isinf(beta) else QA / beta
    wts = np.exp(lw - lw.max()); wts /= wts.sum()
    ess = 1 / np.sum(wts ** 2)
    idx = rng.choice(N, n_show, p=wts)
    # check: mode masses, weighted pushforward vs analytic target
    push_mass = np.bincount(lab, weights=wts, minlength=4)
    tl = np.argmin(((Agrid[:, None, :] - M) ** 2).sum(-1), 1)
    tgt_mass = np.bincount(tl, weights=tgt.ravel(), minlength=4)
    # TV on a coarse histogram
    edges = np.linspace(-3.6, 4.0, 39)
    H, _, _ = np.histogram2d(A[:, 0], A[:, 1], bins=[edges, edges], weights=wts)
    from scipy.stats import norm
    # analytic bin masses by fine-grid integration
    Ht, _, _ = np.histogram2d(Agrid[:, 0], Agrid[:, 1], bins=[edges, edges], weights=tgt.ravel())
    tv = 0.5 * np.abs(H / H.sum() - Ht / Ht.sum()).sum()
    print(f"beta={beta}: ESS={ess:.0f}  push mode mass={np.round(push_mass,3)}  "
          f"target mode mass={np.round(tgt_mass,3)}  TV(hist)={tv:.3f}")

    ax = axes[0, j]
    ax.contourf(WX, WY, q / q.max(), levels=LEV, colors=COLS, rasterized=True)
    for k in range(4):
        ax.contour(WX, WY, (mode_lab == k).astype(float), levels=[0.5], colors="#98a1ae", linewidths=0.55)
    ax.scatter(W[idx[:400], 0], W[idx[:400], 1], s=1.0, color=TEXT, lw=0, alpha=0.5, rasterized=True)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)

    ax = axes[1, j]
    ax.contourf(AX, AY, tgt / tgt.max(), levels=LEV, colors=COLS, rasterized=True)
    ax.contour(AX, AY, Qa, levels=[-3, -1.5, -0.5], colors=ACCENT, linewidths=0.6,
               linestyles=[(0, (3, 2))], alpha=0.8)
    ax.plot(*G, marker="*", color=ACCENT, ms=6, mec="none")
    ax.scatter(A[idx, 0], A[idx, 1], s=1.0, color=TEXT, lw=0, alpha=0.5, rasterized=True)
    ax.set_xlim(-3.6, 4.0); ax.set_ylim(-3.6, 4.0)

    title = r"$\beta=\infty$  (base)" if np.isinf(beta) else rf"$\beta={beta:g}$"
    axes[0, j].set_title(title, fontsize=13, pad=4)

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color(LINE); sp.set_linewidth(0.8)

axes[0, 0].set_ylabel(r"noise $w$", fontsize=12.5)
axes[1, 0].set_ylabel(r"action $a$", fontsize=12.5)
# frozen-decoder arrow between rows
for j in range(4):
    b0 = axes[0, j].get_position(); b1 = axes[1, j].get_position()
    x = b0.x0 + b0.width / 2
    fig.patches.append(matplotlib.patches.FancyArrowPatch(
        (x, b0.y0 - 0.008), (x, b1.y1 + 0.008), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.8))
    if j == 0:
        fig.text(x + 0.012, (b0.y0 + b1.y1) / 2, r"frozen $\psi_\theta$", fontsize=10.5,
                 color=MUTED, va="center", ha="left")
save(fig, "tilt-noise-tilts-policy")
