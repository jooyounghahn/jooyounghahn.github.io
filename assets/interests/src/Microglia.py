"""Microglia: Chan-Vese level-set segmentation of a synthetic branched cell.

Left : a synthetic microglia-like cell (soma + tapering, branching processes)
       with Gaussian noise, drawn as a microscope image would be.
Right: the level-set function phi after Chan-Vese evolution (started from a
       small circle in the soma).  Carmine: the zero level phi = 0, i.e. the
       segmented cell boundary.  Blue lines: other level lines of phi
       (signed distance, spacing 9 px), fading away from the boundary.
Faint, thin branch tips stay outside phi = 0: the hard part of the problem.
Everything is synthetic; no microscopy data are used.  Runs in a few seconds.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
import skfmm
from scipy import ndimage as ndi

sys.path.insert(0, _SRCDIR)
import figstyle as fs

OUT = _FIGDIR + "/Microglia"
SEED = 25
MU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08       # boundary-length weight
NOISE = 0.15


# ------------------------------------------------------------ synthetic cell
def polyline(rng, start, ang, length, turn):
    pts, a, da = [np.array(start, float)], ang, 0.0
    for _ in range(int(length)):
        da = 0.85 * da + rng.normal(0, turn)          # smooth, persistent bending
        a += da
        a = float(np.clip(a, ang - 0.9, ang + 0.9))   # bounded total turn: no loops
        pts.append(pts[-1] + np.array([np.cos(a), np.sin(a)]))
    return np.array(pts)


def synth_cell(n=256, seed=SEED):
    rng = np.random.default_rng(seed)
    c = np.array([n / 2 + rng.uniform(-6, 6), n / 2 + rng.uniform(-4, 4)])
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r = np.hypot((xx - c[0]) / 1.2, (yy - c[1]) / 0.95)
    img = np.clip(1.25 * (1 - (r / 10) ** 2), 0, 1.0)                    # soma
    segs = []
    nb = 7
    angs = np.sort(np.linspace(0, 2 * np.pi, nb, endpoint=False) + rng.uniform(-0.3, 0.3, nb))
    for a in angs:                                                          # primary processes
        P = polyline(rng, c + 7 * np.array([np.cos(a), np.sin(a)]), a, rng.uniform(45, 80), 0.035)
        segs.append((P, 3.0, 1.0, 1.0, rng.uniform(0.45, 0.65)))
        for _ in range(rng.integers(1, 4)):                                 # side branches
            j = rng.integers(int(0.25 * len(P)), int(0.85 * len(P)))
            d = P[j + 1] - P[j]
            a2 = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
            w0 = 3.0 - 2.0 * j / len(P)
            Q = polyline(rng, P[j], a2, rng.uniform(12, 30), 0.05)
            segs.append((Q, max(0.85 * w0, 1.1), 0.8, 0.75, 0.4))
    for P, w0, w1, b0, b1 in segs:                                          # tapering, dimming
        for p, t in zip(P, np.linspace(0, 1, len(P))):
            w, b = w0 + (w1 - w0) * t, b0 + (b1 - b0) * t
            x0, x1 = int(max(p[0] - 6, 0)), int(min(p[0] + 7, n))
            y0, y1 = int(max(p[1] - 6, 0)), int(min(p[1] + 7, n))
            if x1 <= x0 or y1 <= y0:
                continue
            d2 = (xx[y0:y1, x0:x1] - p[0]) ** 2 + (yy[y0:y1, x0:x1] - p[1]) ** 2
            img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], b * np.exp(-d2 / (2 * (w / 2.2) ** 2)))
    img = ndi.gaussian_filter(img, 0.6)
    noisy = img + np.random.default_rng(100 + seed).normal(0, NOISE, img.shape) + 0.1
    return img, noisy, c


# ------------------------------------------------------------ Chan-Vese level set
def curvature(phi):
    fy, fx = np.gradient(phi)
    fyy, _ = np.gradient(fy)
    fxy, fxx = np.gradient(fx)
    return (fxx * fy ** 2 - 2 * fx * fy * fxy + fyy * fx ** 2) / ((fx ** 2 + fy ** 2) ** 1.5 + 1e-8)


def chan_vese(I, phi, mu, dt=2.0, n_iter=1500, eps=1.0, reinit=10):
    """phi_t = delta(phi) [ mu kappa - (I - c1)^2 + (I - c2)^2 ],  phi > 0 inside."""
    phi = phi.astype(float).copy()
    for it in range(1, n_iter + 1):
        ins = phi > 0
        c1, c2 = I[ins].mean(), I[~ins].mean()
        F = mu * curvature(phi) + (I - c2) ** 2 - (I - c1) ** 2
        phi += dt * (eps / np.pi) / (eps ** 2 + phi ** 2) * F
        if it % reinit == 0:
            phi = np.asarray(skfmm.distance(phi, dx=1.0))     # keeps the zero level, sub-pixel
    return phi


def main():
    fs.setup()
    img, noisy, c = synth_cell()
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    phi0 = np.hypot(xx - c[0], yy - c[1]) - 6.0
    phi = chan_vese(noisy, -phi0, MU)
    phi = np.asarray(skfmm.distance(phi, dx=1.0))

    ys, xs = np.nonzero(img > 0.15)
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    pw = 0.49
    h = 0.5 * (ys.max() - ys.min()) + 12
    w = h * (pw * fs.W_IN) / fs.H_IN

    fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
    axL = fig.add_axes([0.0, 0.0, pw, 1.0])
    axR = fig.add_axes([1 - pw, 0.0, pw, 1.0])
    axL.imshow(noisy, cmap=fs.CMAP_SEQ, vmin=-0.3, vmax=1.4, interpolation="nearest")
    lv = np.arange(-9, -120, -9)[::-1]                                     # outside levels
    cols = [fs.CMAP_SEQ(0.72 - 0.5 * k / len(lv)) for k in range(len(lv))][::-1]
    axR.contour(phi, levels=lv, colors=cols, linewidths=1.6, negative_linestyles="solid")
    axR.contour(phi, [0], colors=fs.CARMINE, linewidths=2.0)
    for a in (axL, axR):
        a.set_xlim(cx - w, cx + w); a.set_ylim(cy + h, cy - h); fs.clean(a)
    axR.text(0.95, 0.10, "φ = 0", transform=axR.transAxes, ha="right", va="bottom",
             fontsize=15, color=fs.CARMINE,
             bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"))
    fs.save(fig, OUT + ".png")
    truth = img > 0.3
    seg = phi > 0
    print("Dice vs bright pixels:", 2 * (seg & truth).sum() / (seg.sum() + truth.sum()))


if __name__ == "__main__":
    main()
