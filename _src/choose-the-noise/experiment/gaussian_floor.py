"""Best single Gaussian per state (no amortization): the floor for every proposer in Table 3.

Optimizes a separate mean and log-std for each of the 64 evaluation states by
reparameterized gradients through the 10-step decoder, then reports KL(q || q*).
"""
import json, math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy import *

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
load(os.path.join(RES, "decoders.pt"))
torch.manual_seed(0)
PHI = torch.linspace(0, 2 * math.pi, 65, device=dev)[:-1]
n = 1024
mu = torch.zeros(64, 2, device=dev, requires_grad=True)
ls = torch.zeros(64, 2, device=dev, requires_grad=True)
opt = torch.optim.Adam([mu, ls], 1e-2)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
ph = PHI.repeat_interleave(n)
for it in range(3000):
    w = (mu[:, None] + torch.exp(ls)[:, None] * torch.randn(64, n, 2, device=dev)).reshape(-1, 2)
    kl = 0.5 * (mu ** 2 + torch.exp(2 * ls) - 1 - 2 * ls).sum(-1)
    loss = (-Q(ph, psi(w)).view(64, n).mean(1) / BETA + kl).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
with torch.no_grad():
    m = 16384
    eq, lz = [], []
    for i in range(64):
        w = mu[i] + torch.exp(ls[i]) * torch.randn(m, 2, device=dev)
        eq.append(Q(PHI[i].expand(m), psi(w)).mean() / BETA)
        a = psi(torch.randn(1 << 17, 2, device=dev))
        lz.append(torch.logsumexp(Q(PHI[i].expand(1 << 17), a) / BETA, 0) - math.log(1 << 17))
    kl = 0.5 * (mu ** 2 + torch.exp(2 * ls) - 1 - 2 * ls).sum(-1)
    gap = torch.stack(lz) - (torch.stack(eq) - kl)
print(f"best per-state Gaussian: KL(q||q*) mean {gap.mean():.3f}, min {gap.min():.3f}, max {gap.max():.3f}")
json.dump({"gap_mean": gap.mean().item(), "gap_per_state": gap.tolist(),
           "mu": mu.tolist(), "log_sigma": ls.tolist()}, open(os.path.join(RES, "gaussian_floor.json"), "w"))
