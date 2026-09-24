"""TDA_sound: sliding-window (delay) embedding of two synthetic two-tone sounds.

Every window of a signal (carmine segment, left) becomes one point in R^d.
Left : two tones with pitch ratio 3:2.  The sound is periodic, so the windows
       trace a closed curve: a (2,3) torus knot (a trefoil) on the torus.
Right: two tones with pitch ratio 1:sqrt(2).  The sound never repeats; windows
       taken at random times from a long recording fill the whole torus.
The two phase angles are read off the 4 leading principal components of the
sliding-window point cloud (whitened); points are drawn on a standard torus in
R^3, with the parts behind the surface removed (curve: shown faint).
Persistent homology (ripser) checks the shapes: 3:2 gives one dominant H1 bar and no
long H2 bar; 1:sqrt(2) gives two long H1 bars and one long H2 bar.  All synthetic.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection

sys.path.insert(0, _SRCDIR)
import figstyle as fs
import matplotlib.pyplot as plt

OUT = _FIGDIR + "/TDA_sound"
rng = np.random.default_rng(3)


def sliding_window(sig, d, step=1):
    idx = np.arange(0, len(sig) - d, step)
    return np.stack([sig[i:i + d] for i in idx]), idx


def phases(W):
    """Two phase angles from the 4 leading principal components."""
    pca = PCA(4).fit(W)
    P = pca.transform(W)
    Q = P / np.sqrt(pca.explained_variance_)          # whiten: round circles, even angles
    return np.arctan2(Q[:, 1], Q[:, 0]), np.arctan2(Q[:, 3], Q[:, 2]), P, pca


R, r, TILT = 1.0, 0.42, np.deg2rad(-58)


def torus_xyz(th1, th2, rr=r):
    X = (R + rr * np.cos(th2)) * np.cos(th1)
    Y = (R + rr * np.cos(th2)) * np.sin(th1)
    Z = rr * np.sin(th2)
    Ys = Y * np.cos(TILT) - Z * np.sin(TILT)          # screen vertical
    D = Y * np.sin(TILT) + Z * np.cos(TILT)           # depth, larger = nearer
    return X, Ys, D


# z-buffer of the torus surface, used to hide what lies behind it
XL, YL, NB = (-1.6, 1.6), (-1.3, 1.3), 500
uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 1400), np.linspace(0, 2 * np.pi, 700))
_X, _Y, _D = torus_xyz(uu.ravel(), vv.ravel())
_ix = ((_X - XL[0]) / (XL[1] - XL[0]) * (NB - 1)).astype(int)
_iy = ((_Y - YL[0]) / (YL[1] - YL[0]) * (NB - 1)).astype(int)
ZBUF = np.full((NB, NB), -np.inf)
np.maximum.at(ZBUF, (_iy, _ix), _D)


def visible(X, Y, D, eps=0.03):
    ix = ((X - XL[0]) / (XL[1] - XL[0]) * (NB - 1)).astype(int)
    iy = ((Y - YL[0]) / (YL[1] - YL[0]) * (NB - 1)).astype(int)
    return D >= ZBUF[iy, ix] - eps


def draw_surface(ax, nu=72, nv=36):
    """Pale, softly shaded torus drawn with the painter's algorithm."""
    from matplotlib.collections import PolyCollection
    u = np.linspace(0, 2 * np.pi, nu + 1)
    v = np.linspace(0, 2 * np.pi, nv + 1)
    U, V = np.meshgrid(u, v)
    X, Ys, D = torus_xyz(U, V)
    # normal (unrotated) and its depth component after the tilt
    nx, ny, nz = np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)
    nd = ny * np.sin(TILT) + nz * np.cos(TILT)
    ns = ny * np.cos(TILT) - nz * np.sin(TILT)
    light = np.clip(0.55 * nd + 0.45 * ns + 0.25 * nx, 0, 1)
    polys, depth, shade = [], [], []
    for i in range(nv):
        for j in range(nu):
            q = [(X[i, j], Ys[i, j]), (X[i, j + 1], Ys[i, j + 1]),
                 (X[i + 1, j + 1], Ys[i + 1, j + 1]), (X[i + 1, j], Ys[i + 1, j])]
            polys.append(q)
            depth.append(D[i:i + 2, j:j + 2].mean())
            shade.append(light[i:i + 2, j:j + 2].mean())
    o = np.argsort(depth)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "surf", ["#C9DAE8", fs.PALE, "#F4F8FB"])
    cols = cmap(np.asarray(shade)[o])
    ax.add_collection(PolyCollection([polys[k] for k in o], facecolors=cols,
                                     edgecolors=cols, linewidths=0.4, zorder=0))


# ---- 3:2  (frequencies 2 and 3, period 2*pi) ------------------------------------
dtA = 0.05
dA = int(round(2 * np.pi / dtA))                       # window = one full period
tA = np.arange(0, 40, dtA)
sA = np.cos(2 * tA) + 0.7 * np.cos(3 * tA) + 0.02 * rng.normal(size=tA.size)
WA, idxA = sliding_window(sA, dA, step=1)
a1, a2, PA, pcaA = phases(WA)
print("3:2   explained variance:", np.round(pcaA.explained_variance_ratio_, 3))

# ---- 1:sqrt(2) ---------------------------------------------------------------------
# window of 14*pi ~ 7 periods of tone 1 and ~ 9.9 periods of tone 2
dtB, dB = 0.2, 220
tB = np.arange(0, 20000, dtB)                          # a long recording ...
sB = np.cos(1.0 * tB) + 0.7 * np.cos(np.sqrt(2) * tB) + 0.03 * rng.normal(size=tB.size)
idxB = np.sort(rng.choice(len(tB) - dB, 1700, replace=False))  # ... windows at random times
WB = np.stack([sB[i:i + dB] for i in idxB])
b1, b2, PB, pcaB = phases(WB)
print("1:r2  explained variance:", np.round(pcaB.explained_variance_ratio_, 3))

# ---- persistence check ---------------------------------------------------------------
try:
    from ripser import ripser
    for name, P in (("3:2 ", PA), ("1:r2", PB)):
        sub = rng.choice(len(P), min(400, len(P)), replace=False)
        dg = ripser(P[sub], maxdim=2)["dgms"]
        l1 = np.sort(np.diff(dg[1], axis=1).ravel())[::-1]
        l2 = np.sort(np.diff(dg[2], axis=1).ravel())[::-1]
        print(name, "H1 lifetimes:", np.round(l1[:3], 2), " H2:", np.round(l2[:2], 2))
except ImportError:
    pass

# ---- figure ----------------------------------------------------------------------------
fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)

wL = fig.add_axes([0.04, 0.66, 0.42, 0.15])
wR = fig.add_axes([0.54, 0.66, 0.42, 0.15])
show = 20.0
mA = tA <= show
mB = tB <= show
wL.plot(tA[mA], sA[mA], color=fs.INK, lw=1.6)
k = 150                                               # highlighted window start
wL.plot(tA[k:k + dA], sA[k:k + dA], color=fs.CARMINE, lw=3.2, solid_capstyle="round")
wR.plot(tB[mB], sB[mB], color=fs.INK, lw=1.6)
for a in (wL, wR):
    fs.clean(a)
    a.set_xlim(0, show)
    a.set_ylim(-1.9, 1.9)
fig.text(0.25, 0.875, "pitch ratio 3 : 2", ha="center", va="center", fontsize=13, color=fs.INK)
fig.text(0.75, 0.875, "pitch ratio 1 : √2", ha="center", va="center", fontsize=13, color=fs.INK)

# left: the closed curve drawn on the torus (hidden parts removed)
aL = fig.add_axes([0.01, -0.01, 0.48, 0.64])
draw_surface(aL)
X, Ys, D = torus_xyz(a1, a2, rr=r * 1.02)
vis = visible(X, Ys, D)
pts = np.column_stack([X, Ys])
segs = np.stack([pts[:-1], pts[1:]], axis=1)
near = np.hypot(*(pts[1:] - pts[:-1]).T) < 0.2
front = near & vis[:-1] & vis[1:]
back = near & ~(vis[:-1] & vis[1:])
aL.add_collection(LineCollection(segs[back], colors=fs.SKY, linewidths=1.7,
                                 capstyle="round", zorder=1))       # seen through the surface
aL.add_collection(LineCollection(segs[front], colors=fs.STEEL, linewidths=2.8,
                                 capstyle="round", zorder=2))
j = np.searchsorted(idxA, k)
aL.scatter(X[j], Ys[j], s=140, color=fs.CARMINE, zorder=5, edgecolor="white", linewidth=1.2)
print("carmine point visible:", bool(vis[j]))

# right: filled torus as depth-sorted points
aR = fig.add_axes([0.51, -0.01, 0.48, 0.64])
draw_surface(aR)
X, Ys, D = torus_xyz(b1, b2, rr=r * 1.02)
vis = visible(X, Ys, D)
aR.scatter(X[vis], Ys[vis], s=6, color=fs.STEEL, lw=0, zorder=2)

for a in (aL, aR):
    a.set_aspect("equal"); fs.clean(a)
    a.set_xlim(*XL); a.set_ylim(*YL)

fs.save(fig, OUT + ".png")

from PIL import Image
im = Image.open(OUT + ".png")
print(im.size)
im.resize((320, int(320 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_small.png")
