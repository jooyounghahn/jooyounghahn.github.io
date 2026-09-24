"""ROM_PINN: proper orthogonal decomposition and Operator Inference on a toy reacting plume.

Full model: 2-D advection-diffusion-reaction of a scalar c (e.g. a fuel or
heat marker) injected at the bottom of a channel, carried upward and swayed
sideways by a time-periodic cross-flow, and consumed by a first-order
reaction.  48 x 96 finite-volume grid (4608 unknowns), upwind advection,
Heun time stepping.

Reduced model: POD basis from the SVD of the snapshot matrix (training window
only), then Operator Inference: a 10-equation ODE system
    dz/dt = c + A z + sin(wt)(b1 + N1 z) + cos(wt)(b2 + N2 z)
fitted by regularized least squares to the projected snapshots, and
integrated beyond the training window.

Default figure ("pair"): the same instant t = 12, which lies beyond the
training window 3 <= t <= 9, computed by
  left  : the full model (4608 unknowns),
  middle: the first 3 POD modes (columns of V from the SVD of the training
          snapshots), signed, on a steel / white / sand diverging map with a
          symmetric scale (drawn as sign(m)|m|^0.6 so weak lobes stay visible),
  right : the 10-equation reduced model (lifted back to the grid).
Other variants (argv[1]): "sv" -> singular values of the snapshot matrix;
"probe" -> c at one probe point, full vs reduced model, training vs forecast.
Synthetic toy only.  Runs in a few seconds.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.colors import PowerNorm, LinearSegmentedColormap

sys.path.insert(0, _SRCDIR)
import figstyle as fs

OUT = _FIGDIR + "/ROM_PINN"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "pair"     # "pair" (homepage), "sv", "probe"
R, REG = 10, 1e-2
# signed map for the POD modes: steel <- white -> sand (carmine is kept for the reduced model)
CMAP_MODE = LinearSegmentedColormap.from_list(
    "rom_modes", [fs.STEEL, fs.SKY, "#F7F7F7", "#E6D3A8", fs.SAND])

# ------------------------------------------------------------ full model
NX, NY, LX, LY = 48, 96, 2.0, 3.0
DX, DY = LX / NX, LY / NY
XC = -1 + (np.arange(NX) + 0.5) * DX
YC = (np.arange(NY) + 0.5) * DY
X, Y = np.meshgrid(XC, YC)
V0, AMP, PERIOD = 1.0, 0.5, 1.6
OM = 2 * np.pi / PERIOD
KY = 0.8 * OM / V0
DIFF, KAPPA = 0.006, 0.25
INFLOW = np.exp(-(XC / 0.16) ** 2)


def rhs(c, t):
    u = AMP * np.sin(OM * t - KY * Y)
    g = np.zeros((NY + 2, NX + 2)); g[1:-1, 1:-1] = c
    g[0, 1:-1] = 2 * INFLOW - c[0]          # inflow c = profile at the bottom
    g[-1, 1:-1] = c[-1]                      # outflow at the top
    g[1:-1, 0] = -c[:, 0]; g[1:-1, -1] = -c[:, -1]   # c = 0 on the side walls
    C = g[1:-1, 1:-1]
    dcy = (C - g[:-2, 1:-1]) / DY
    dcx = np.where(u > 0, (C - g[1:-1, :-2]) / DX, (g[1:-1, 2:] - C) / DX)
    lap = (g[1:-1, 2:] - 2 * C + g[1:-1, :-2]) / DX ** 2 + (g[2:, 1:-1] - 2 * C + g[:-2, 1:-1]) / DY ** 2
    return -V0 * dcy - u * dcx + DIFF * lap - KAPPA * C


def simulate(T=16.0, dt=0.004, every=10):
    c = np.zeros((NY, NX)); snaps, ts = [], []
    for n in range(int(round(T / dt))):
        t = n * dt
        k1 = rhs(c, t); k2 = rhs(c + dt * k1, t + dt)
        c = c + 0.5 * dt * (k1 + k2)
        if (n + 1) % every == 0:
            snaps.append(c.copy()); ts.append((n + 1) * dt)
    return np.array(snaps), np.array(ts)


# ------------------------------------------------------------ POD + Operator Inference
def features(z, t):
    s, co = np.sin(OM * t), np.cos(OM * t)
    return np.vstack([np.ones_like(s)[None], z, s[None], co[None], s * z, co * z])


def reduce_and_forecast(S, t, t0=3.0, t1=9.0):
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    Q = S.reshape(len(S), -1).T
    qbar = Q[:, i0:i1].mean(1, keepdims=True)
    V, sv, _ = np.linalg.svd(Q[:, i0:i1] - qbar, full_matrices=False)
    Vr = V[:, :R]
    Z = Vr.T @ (Q - qbar)
    dZ = np.gradient(Z, t[1] - t[0], axis=1)
    D = features(Z[:, i0:i1], t[i0:i1]).T
    O = np.linalg.solve(D.T @ D + REG * np.eye(D.shape[1]), D.T @ dZ[:, i0:i1].T).T
    f = lambda tt, z: O @ features(z[:, None], np.array([tt])).ravel()
    sol = solve_ivp(f, (t[i0], t[-1]), Z[:, i0], t_eval=t[i0:], rtol=1e-8, atol=1e-10)
    Qrom = qbar + Vr @ sol.y
    return sv, Vr, i0, i1, Q[:, i0:], Qrom


def main():
    fs.setup()
    S, t = simulate()
    sv, Vr, i0, i1, Q, Qrom = reduce_and_forecast(S, t)
    err = np.linalg.norm(Qrom - Q, axis=0) / np.linalg.norm(Q, axis=0)
    print("relative error: training %.3f, forecast %.3f" % (err[: i1 - i0].max(), err[i1 - i0:].max()))

    fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
    PW = 0.34 if VARIANT == "pair" else 0.40
    axL = fig.add_axes([0.01 if VARIANT == "pair" else 0.02, 0.03, PW, 0.94])
    k_show = np.searchsorted(t, 12.0)
    axL.imshow(S[k_show], origin="lower", extent=[-1, 1, 0, LY], cmap=fs.CMAP_SEQ,
               norm=PowerNorm(0.55, vmin=0, vmax=0.95), interpolation="bilinear")
    fs.clean(axL)
    for s_ in axL.spines.values():
        s_.set_visible(True); s_.set_color(fs.MUTED); s_.set_linewidth(1.0)

    if VARIANT == "pair":
        k_rom = k_show - i0
        axR = fig.add_axes([0.65, 0.03, PW, 0.94])
        axR.imshow(Qrom[:, k_rom].reshape(NY, NX), origin="lower", extent=[-1, 1, 0, LY], cmap=fs.CMAP_SEQ,
                   norm=PowerNorm(0.55, vmin=0, vmax=0.95), interpolation="bilinear")
        fs.clean(axR)
        for s_ in axR.spines.values():
            s_.set_visible(True); s_.set_color(fs.CARMINE); s_.set_linewidth(1.6)
        # middle column: the first 3 POD modes, each on a symmetric signed scale
        # (shown as sign(m)|m|^0.6, like the PowerNorm of the fields, so the weaker lobes stay visible)
        for j, y0 in enumerate((0.625, 0.33, 0.035)):
            m = Vr[:, j].reshape(NY, NX)
            m = np.sign(m) * np.abs(m) ** 0.6; v = np.abs(m).max()
            axM = fig.add_axes([0.43375, y0, 0.1325, 0.265])  # 2:3 box, same shape as the domain
            axM.imshow(m, origin="lower", extent=[-1, 1, 0, LY], cmap=CMAP_MODE, vmin=-v, vmax=v,
                       interpolation="bilinear")
            fs.clean(axM)
            for s_ in axM.spines.values():
                s_.set_visible(True); s_.set_color(fs.MUTED); s_.set_linewidth(1.0)
        fig.text(0.50, 0.978, "modes", ha="center", va="top", fontsize=13, color=fs.INK)
        axL.text(0.5, 0.97, "4608 unknowns", transform=axL.transAxes, ha="center", va="top", fontsize=13)
        axR.text(0.5, 0.97, "10 equations", transform=axR.transAxes, ha="center", va="top", fontsize=13,
                 color=fs.CARMINE)
    elif VARIANT == "sv":
        axR = fig.add_axes([0.55, 0.13, 0.42, 0.78])
        k = np.arange(1, 31); y = sv[:30] / sv[0]
        axR.semilogy(k[R:], y[R:], "o", ms=6.5, mfc="white", mec=fs.MUTED, mew=1.6)
        axR.semilogy(k[:R], y[:R], "o", ms=7.5, color=fs.CARMINE)
        axR.set_xlim(0, 31); axR.set_ylim(3e-5, 2)
        axR.set_yticks([1, 1e-4]); axR.set_yticklabels(["1", "10⁻⁴"], fontsize=13)
        axR.minorticks_off()
        axR.set_xticks([1, 10, 30]); axR.set_xticklabels(["1", "10", "30"], fontsize=13)
        axR.tick_params(length=4, width=1.2, colors=fs.MUTED)
        for s_ in axR.spines.values():
            s_.set_linewidth(1.2)
        axR.text(0.97, 0.97, "singular values", transform=axR.transAxes, ha="right", va="top",
                 fontsize=13, color=fs.INK)
    else:
        py, px = 45, 30
        axL.plot(XC[px], YC[py], "o", ms=8, color=fs.STEEL, mec="white", mew=1.5)
        p = py * NX + px
        tt = t[i0:]
        show = tt >= 5.4
        axR = fig.add_axes([0.49, 0.18, 0.49, 0.64])
        axR.axvspan(5.4, t[i1], color=fs.PALE, lw=0)
        axR.plot(tt[show], Q[p, show], color=fs.STEEL, lw=2.4)
        fc = np.arange(len(tt)) >= i1 - i0
        axR.plot(tt[fc], Qrom[p, fc], color=fs.CARMINE, lw=2.2, ls=(0, (2.0, 1.3)))
        axR.set_xlim(5.4, tt[-1]); fs.clean(axR)
        lo, hi = Q[p].min(), Q[p].max()
        axR.set_ylim(lo - 0.25 * (hi - lo), hi + 0.55 * (hi - lo))
        axR.text(0.5 * (5.4 + t[i1]), hi + 0.3 * (hi - lo), "training", ha="center", fontsize=14)
        axR.text(0.5 * (t[i1] + tt[-1]), hi + 0.3 * (hi - lo), "forecast", ha="center", fontsize=14,
                 color=fs.CARMINE)
    fs.save(fig, OUT + ("" if VARIANT == "pair" else "_" + VARIANT) + ".png")


if __name__ == "__main__":
    main()
