"""Shared 2D toy: base mixture, value Q, frozen flow decoder, one-step student, inversion."""
import math
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
BETA, K = 1.0, 10

R, S = 2.2, 0.38
ANG = torch.deg2rad(torch.tensor([20., 110., 200., 290.]))
M = torch.stack([R * torch.cos(ANG), R * torch.sin(ANG)], 1).to(dev)   # mode means
RG = 2.2

def goal(phi):
    return torch.stack([RG * torch.cos(phi), RG * torch.sin(phi)], -1)

def Q(phi, a):
    return -0.5 * ((a - goal(phi)) ** 2).sum(-1)

def sample_base(n):
    k = torch.randint(0, 4, (n,), device=dev)
    return M[k] + S * torch.randn(n, 2, device=dev)

def sample_tilted(phi):
    """Exact samples from pi_GMM(a) exp(Q(phi, a)/beta): again a Gaussian mixture."""
    c = 1.0 / BETA
    g = goal(phi)                                          # n,2
    s2 = 1.0 / (1.0 / S ** 2 + c)
    logw = -0.5 * ((M[None] - g[:, None]) ** 2).sum(-1) * c / (1 + c * S ** 2)
    k = torch.distributions.Categorical(logits=logw).sample()
    mean = s2 * (M[k] / S ** 2 + c * g)
    return mean + math.sqrt(s2) * torch.randn_like(mean)

def mlp(i, o, h=256, n=3):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))

def temb(t):
    f = torch.arange(1, 5, device=t.device) * math.pi
    return torch.cat([t, torch.sin(f * t), torch.cos(f * t)], -1)

def feat(phi):
    return torch.stack([torch.cos(phi), torch.sin(phi)], -1)

# ---------------------------------------------------------------- frozen decoders
torch.manual_seed(0)
vnet = mlp(2 + 9, 2).to(dev)

def v(x, t):
    return vnet(torch.cat([x, temb(t.expand(x.shape[0], 1))], -1))

def psi(w, k=K):
    x, h = w, 1.0 / k
    for i in range(k):
        x = x + h * v(x, torch.full((1, 1), i * h, device=dev))
    return x

def invert(a, k=K, iters=20):
    """Reverse the Euler sampler: x_i = x_{i+1} - h v(x_i, t_i), solved by fixed point."""
    x, h = a, 1.0 / k
    for i in reversed(range(k)):
        t = torch.full((1, 1), i * h, device=dev)
        y = x
        for _ in range(iters):
            y = x - h * v(y, t)
        x = y
    return x

student = mlp(2, 2).to(dev)

def load(path):
    sd = torch.load(path, map_location=dev)
    vnet.load_state_dict(sd["v"]); student.load_state_dict(sd["s"])
    for prm in list(vnet.parameters()) + list(student.parameters()):
        prm.requires_grad_(False)
