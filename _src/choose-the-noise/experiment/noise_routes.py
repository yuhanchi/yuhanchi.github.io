"""Five routes for carrying action-space value into noise space, on a 2D toy.

A flow-matching policy is trained once on a 4-mode Gaussian mixture and then frozen.
Its decoder is a K-step Euler sampler, a = psi(w), w ~ N(0, I).  A state is a goal
angle phi; the action-value is Q(phi, a) = -0.5 * ||a - g(phi)||^2, known exactly.
Each route trains the same Gaussian proposer q(w | phi) = N(mu(phi), diag sigma(phi)^2):

  R1  noise critic:   regress Q^W(phi, w) onto Q(phi, psi(w)), w ~ N(0, I); ascend Q^W
  R2  inversion:      invert corrected actions a* ~ pi*(.|phi) to noises; fit q by MLE
  R3  backprop:       reparameterized gradient through all K Euler steps
  R4  one step:       distil a one-step decoder psi_1 ~ psi, backprop through psi_1
  R5  likelihood:     score-function gradient with a leave-one-out baseline

R1, R3, R4, R5 maximize J = E_q[Q]/beta - KL(q || N(0, I)).  For each state,
  log Z(phi) - J(phi) = KL(q || q*),   q*(w) ∝ N(w; 0, I) exp(Q(phi, psi(w)) / beta),
so the gap to log Z, estimated by Monte Carlo, measures how far q is from the exact
tilted noise posterior.  R2 never sees Q; its corrections are exact samples from the
tilted policy of the analytic mixture, standing in for an expert.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--updates", type=int, default=4000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy import *
ckpt = os.path.join(args.out, "decoders.pt")
log = {}
if os.path.exists(ckpt):
    sd = torch.load(ckpt); vnet.load_state_dict(sd["v"]); student.load_state_dict(sd["s"])
else:
    opt = torch.optim.Adam(vnet.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
    for it in range(20000):                            # conditional flow matching
        a1 = sample_base(8192); w0 = torch.randn_like(a1)
        t = torch.rand(8192, 1, device=dev)
        loss = ((v((1 - t) * w0 + t * a1, t) - (a1 - w0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    opt = torch.optim.Adam(student.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
    for it in range(20000):                            # one-step distillation
        w = torch.randn(8192, 2, device=dev)
        with torch.no_grad(): tgt = psi(w)
        loss = ((student(w) - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    torch.save({"v": vnet.state_dict(), "s": student.state_dict()}, ckpt)
for prm in list(vnet.parameters()) + list(student.parameters()):
    prm.requires_grad_(False)

with torch.no_grad():
    w = torch.randn(200000, 2, device=dev)
    a = psi(w)
    lab = ((a[:, None] - M[None]) ** 2).sum(-1).argmin(1)
    log["base_mode_mass"] = (torch.bincount(lab, minlength=4).float() / len(a)).tolist()
    log["student_rmse"] = ((student(w) - a) ** 2).sum(-1).mean().sqrt().item()
    # inversion accuracy on corrected actions
    phi = torch.rand(20000, device=dev) * 2 * math.pi
    astar = sample_tilted(phi)
    for iters in (1, 20):
        err = (psi(invert(astar, iters=iters)) - astar).norm(dim=-1)
        log[f"inversion_err_median_iters{iters}"] = err.median().item()
        log[f"inversion_err_p90_iters{iters}"] = err.quantile(0.9).item()
print(json.dumps(log, indent=1))

# ---------------------------------------------------------------- evaluation
EVAL_PHI = torch.linspace(0, 2 * math.pi, 65, device=dev)[:-1]

@torch.no_grad()
def log_Z(dec, phi, n=1 << 17):
    out = []
    for p_ in phi:
        a = dec(torch.randn(n, 2, device=dev))
        out.append(torch.logsumexp(Q(p_.expand(n), a) / BETA, 0) - math.log(n))
    return torch.stack(out)

LOGZ = {"psi": log_Z(psi, EVAL_PHI), "student": log_Z(student, EVAL_PHI)}

class Proposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(2, 4, h=128, n=2)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, phi):
        o = self.net(feat(phi)); return o[:, :2], o[:, 2:].clamp(-5, 2)
    def kl(self, mu, ls):                                   # KL(q || N(0, I))
        return 0.5 * (mu ** 2 + torch.exp(2 * ls) - 1 - 2 * ls).sum(-1)
    def logq(self, w, mu, ls):
        return (-0.5 * ((w - mu) / torch.exp(ls)) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)

@torch.no_grad()
def evaluate(prop, dec_name, n=4096):
    dec = psi if dec_name == "psi" else student
    mu, ls = prop(EVAL_PHI)
    eq = []
    for i, p_ in enumerate(EVAL_PHI):
        w = mu[i] + torch.exp(ls[i]) * torch.randn(n, 2, device=dev)
        eq.append(Q(p_.expand(n), dec(w)).mean() / BETA)
    J = torch.stack(eq) - prop.kl(mu, ls)
    gap = LOGZ[dec_name] - J
    return gap.mean().item(), gap.cpu().numpy(), torch.stack(eq).mean().item(), prop.kl(mu, ls).mean().item()

SNAPS = (0, 25, 100, 400)
B_S, N_W = 256, 4      # states per update, noises per state: 1024 decoder calls per update

def run(route, seed):
    torch.manual_seed(1000 + seed)
    prop = Proposer().to(dev)
    opt = torch.optim.Adam(prop.parameters(), 1e-3)
    dec_name = "student" if route == "R4" else "psi"
    if route == "R1":
        qw = mlp(4, 1).to(dev)
        qopt = torch.optim.Adam(qw.parameters(), 1e-3)
    curve, times, snaps = [], [], {}
    for it in range(args.updates + 1):
        if it % 100 == 0 or it in (25, 50):
            curve.append((it, evaluate(prop, dec_name)[0]))
        if it in SNAPS or it == args.updates:
            with torch.no_grad():
                mu_, ls_ = prop(EVAL_PHI)
            snaps[it] = (mu_.cpu().tolist(), ls_.cpu().tolist())
        if it == args.updates:
            break
        torch.cuda.synchronize(); t0 = time.perf_counter()
        phi = (torch.rand(B_S, device=dev) * 2 * math.pi).repeat_interleave(N_W)
        mu, ls = prop(phi)
        eps = torch.randn(len(phi), 2, device=dev)
        if route == "R1":
            with torch.no_grad():
                w0 = torch.randn(len(phi), 2, device=dev)
                y = Q(phi, psi(w0)) / BETA
            ql = ((qw(torch.cat([feat(phi), w0], -1)).squeeze(-1) - y) ** 2).mean()
            qopt.zero_grad(); ql.backward(); qopt.step()
            w = mu + torch.exp(ls) * eps
            loss = (-qw(torch.cat([feat(phi), w], -1)).squeeze(-1) + prop.kl(mu, ls)).mean()
        elif route == "R2":
            with torch.no_grad():
                what = invert(sample_tilted(phi))
            loss = -prop.logq(what, mu, ls).mean()
        elif route in ("R3", "R4"):
            dec = psi if route == "R3" else student
            w = mu + torch.exp(ls) * eps
            loss = (-Q(phi, dec(w)) / BETA + prop.kl(mu, ls)).mean()
        elif route == "R5":
            with torch.no_grad():
                w = mu + torch.exp(ls) * eps
                r = (Q(phi, psi(w)) / BETA).view(B_S, N_W)
                b = (r.sum(1, keepdim=True) - r) / (N_W - 1)          # leave-one-out
                adv = (r - b).view(-1)
            loss = (-prop.logq(w, mu, ls) * adv + prop.kl(mu, ls)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    gap, gaps, eq, kl = evaluate(prop, dec_name)
    with torch.no_grad():                                      # samples for figures
        mu, ls = prop(EVAL_PHI)
    res = dict(route=route, seed=seed, gap=gap, EQ_over_beta=eq, KL=kl,
               ms_per_update=1000 * float(np.median(times[50:])), curve=curve,
               gap_per_state=gaps.tolist(), snapshots=snaps)
    return res

results = []
for route in ("R1", "R2", "R3", "R4", "R5"):
    for seed in range(args.seeds):
        r = run(route, seed)
        print(f"{route} seed {seed}: gap {r['gap']:.3f}  E[Q]/b {r['EQ_over_beta']:.3f}  "
              f"KL {r['KL']:.3f}  {r['ms_per_update']:.2f} ms/update", flush=True)
        results.append(r)
log["logZ_mean_psi"] = LOGZ["psi"].mean().item()
log["logZ_mean_student"] = LOGZ["student"].mean().item()
log["eval_phi"] = EVAL_PHI.cpu().tolist()
log["config"] = dict(K=K, beta=BETA, fixed_point_iters=20, updates=args.updates, states_per_update=B_S,
                     noises_per_state=N_W, seeds=args.seeds)
json.dump(dict(log=log, results=results), open(os.path.join(args.out, "results.json"), "w"))
