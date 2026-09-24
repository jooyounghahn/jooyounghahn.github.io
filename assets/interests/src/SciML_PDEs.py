"""SciML_PDEs: neural-network solvers for the eikonal equation with vanishing viscosity.

Left : 1-D eikonal |u'| = 1 on (-1, 1), u(+-1) = 0.  Many functions satisfy it
       almost everywhere (grey zigzags).  A small network trained on the viscous
       problem -eps u'' + |u'| = 1 while eps is reduced (sky -> steel) ends on the
       viscosity solution 1 - |x| (carmine).
Right: 2-D travel time from a small source through a medium with a slow region,
       computed by a network trained the same way (vanishing viscosity).
       Contours are wavefronts; they bend around the slow region and meet
       behind it in a kink.
Synthetic toy problems only.  Runs in about a minute on a CPU.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
import torch

sys.path.insert(0, _SRCDIR)
import figstyle as fs

OUT = _FIGDIR + "/SciML_PDEs"
torch.set_num_threads(4)
torch.manual_seed(1)
np.random.seed(1)


def mlp(din, w, depth):
    layers, d = [], din
    for _ in range(depth):
        layers += [torch.nn.Linear(d, w), torch.nn.Tanh()]
        d = w
    layers += [torch.nn.Linear(d, 1)]
    return torch.nn.Sequential(*layers)


# ---------------------------------------------------------------- 1-D problem
def train_1d():
    net = mlp(1, 32, 2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xb = torch.tensor([[-1.0], [1.0]])
    xs = torch.linspace(-1, 1, 401)[:, None]
    stages = [(0.5, 1200), (0.25, 1200), (0.1, 1200), (0.002, 2000)]
    snaps = []
    for eps, n_it in stages:
        for _ in range(n_it):
            x = (torch.rand(256, 1) * 2 - 1).requires_grad_(True)
            u = net(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
            res = -eps * d2u + du.abs() - 1.0
            loss = (res ** 2).mean() + 10.0 * (net(xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            snaps.append(net(xs).numpy().ravel())
    return xs.numpy().ravel(), snaps


# ---------------------------------------------------------------- 2-D problem
SRC = torch.tensor([0.0, -1.15]); R0 = 0.07
BLOB = torch.tensor([0.0, -0.15])
BOX = (1.0, 1.5)                                   # half-width, half-height


def speed(p):
    return 1.0 - 0.75 * torch.exp(-((p - BLOB) ** 2).sum(1, keepdim=True) / 0.09)


class TravelTime(torch.nn.Module):
    """T(x) = (|x - src| - r0) * softplus(N(x)): zero on the source circle."""
    def __init__(self):
        super().__init__()
        self.f = mlp(2, 48, 3)

    def forward(self, p):
        d = torch.sqrt(((p - SRC) ** 2).sum(1, keepdim=True)) - R0
        return d * torch.nn.functional.softplus(self.f(p) + 0.5)


def train_2d(n_it=3500):
    net = TravelTime()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 1200, 0.5)
    scale = torch.tensor(BOX)
    for it in range(n_it):
        eps = max(0.05 * 0.5 ** (it / 450), 0.001)          # vanishing viscosity
        p = (torch.rand(1024, 2) * 2 - 1) * scale
        p = p[((p - SRC) ** 2).sum(1) > R0 ** 2].requires_grad_(True)
        T = net(p)
        g = torch.autograd.grad(T.sum(), p, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, k].sum(), p, create_graph=True)[0][:, k]
                  for k in range(2))
        res = -eps * lap + speed(p).squeeze() * g.norm(dim=1) - 1.0
        loss = (res ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    nx, ny = 161, 241
    X, Y = np.meshgrid(np.linspace(-BOX[0], BOX[0], nx), np.linspace(-BOX[1], BOX[1], ny))
    P = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        T = net(P).numpy().reshape(ny, nx)
        C = speed(P).numpy().reshape(ny, nx)
    return X, Y, T, C


def main():
    fs.setup()
    x, snaps = train_1d()
    X, Y, T, C = train_2d()

    fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
    axL = fig.add_axes([0.03, 0.08, 0.46, 0.84])
    axR = fig.add_axes([0.56, 0.04, 0.41, 0.92])

    # left: a.e. solutions vs vanishing-viscosity network
    grey = dict(color=fs.MUTED, lw=1.6, ls=(0, (3, 2)), alpha=0.9)
    axL.plot(x, np.abs(x) - 1, **grey)                               # V
    saw = lambda x, k: (1 / k) * (1 - np.abs(((x + 1) * k) % 2 - 1))   # k teeth
    axL.plot(x, saw(x, 2), **grey)
    axL.plot(x, -saw(x, 3), **grey)
    for u, col in zip(snaps[:3], [fs.CMAP_SEQ(0.48), fs.CMAP_SEQ(0.62), fs.CMAP_SEQ(0.76)]):
        axL.plot(x, u, color=col, lw=2.2)
    axL.plot(x, snaps[-1], color=fs.CARMINE, lw=2.8)
    axL.axhline(0, color=fs.INK, lw=1.0)
    axL.set_xlim(-1.08, 1.08); axL.set_ylim(-1.1, 1.18)
    fs.clean(axL)
    axL.text(-1.02, 1.05, "|u′| = 1", fontsize=14, color=fs.INK, va="center")
    axL.annotate("", xy=(0.0, 0.95), xytext=(0.0, 0.50),
                 arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=1.6, mutation_scale=14))
    axL.text(0.16, 1.04, "ε → 0", fontsize=14, color=fs.INK, va="center")

    # right: travel time around a slow region
    mask = np.hypot(X - SRC[0].item(), Y - SRC[1].item()) < R0
    axR.imshow(1 - C, extent=[-BOX[0], BOX[0], -BOX[1], BOX[1]], origin="lower",
               cmap=fs.CMAP_SEQ, vmin=-0.05, vmax=1.05)
    axR.contour(X, Y, np.ma.array(T, mask=mask), levels=np.arange(0.17, T[-4].min() - 0.03, 0.24),
                colors=fs.INK, linewidths=1.6)
    axR.plot(SRC[0], SRC[1], "o", ms=7, color=fs.INK)
    axR.set_aspect("equal"); fs.clean(axR)
    axR.set_xlim(-BOX[0], BOX[0]); axR.set_ylim(-BOX[1], BOX[1])

    fs.save(fig, OUT + ".png")


if __name__ == "__main__":
    main()
