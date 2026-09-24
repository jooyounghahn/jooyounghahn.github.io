"""TDA_velocity: streaks in a synthetic near-wall velocity slice and their persistence barcode.

Left : a synthetic wall-parallel slice of the streamwise velocity fluctuation u'
       (x = flow direction, vertical; z = spanwise, horizontal; periodic in both).
       Six meandering low-speed streaks (blue) with high-speed fluid between them,
       plus small-scale noise.  The black contour is the sublevel set {u' <= t}.
Right: H0 barcode of the sublevel-set filtration (periodic cubical complex, GUDHI).
       Long bars = low-speed streaks, short bars = noise; the dashed line is the
       level t used for the contour on the left.  All data are synthetic.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import gudhi

sys.path.insert(0, _SRCDIR)
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = _FIGDIR + "/TDA_velocity"

rng = np.random.default_rng(7)

# ---- synthetic streak field (wall units) -----------------------------------
Lz, Lx = 600.0, 960.0            # spanwise, streamwise extent
nz, nx = 150, 240
z = np.arange(nz) * Lz / nz
x = np.arange(nx) * Lx / nx
Z, X = np.meshgrid(z, x)          # rows = x (flow, shown vertically), cols = z

def pdist(a, b, L):
    d = np.abs(a - b) % L
    return np.minimum(d, L - d)

u = np.zeros_like(Z)
nstreak = 6
spacing = Lz / nstreak            # 100 wall units
for k in range(nstreak):
    for sign, off in ((-1.0, 0.0), (+0.8, 0.5)):          # low-speed, high-speed
        z0 = (k + off) * spacing + rng.normal(0, 6)
        amp = 12 + rng.normal(0, 3)
        lam = rng.uniform(500, 900)
        zc = z0 + amp * np.sin(2 * np.pi * X / lam + rng.uniform(0, 2 * np.pi))
        # finite streak length: smooth envelope along x (periodic)
        xc = rng.uniform(0, Lx)
        half = rng.uniform(300, 400)
        env = 1.0 / (1.0 + np.exp((pdist(X, xc, Lx) - half) / 25.0))
        u += sign * env * np.exp(-pdist(Z, zc, Lz) ** 2 / (2 * 18.0 ** 2))

noise = gaussian_filter(rng.normal(size=u.shape), sigma=2.6, mode="wrap")
noise /= noise.std()
u = u + 0.09 * noise
u = u / np.abs(u).max()

# ---- sublevel-set persistence on the torus -----------------------------------
cc = gudhi.PeriodicCubicalComplex(top_dimensional_cells=u, periodic_dimensions=[True, True])
pers = cc.persistence()
h0 = np.array([bd for dim, bd in pers if dim == 0])
fin = np.isfinite(h0[:, 1])
top = u.max()
h0[~fin, 1] = top                      # essential class: draw to the top level
life = h0[:, 1] - h0[:, 0]
order = np.argsort(-life)
h0, life = h0[order], life[order]
keep = life > 0.03
h0, life = h0[keep], life[keep]
nshow = min(len(h0), 22)
h0, life = h0[:nshow], life[:nshow]
nlong = int(np.sum(life > 0.45))
print("H0 bars kept:", len(h0), " long:", nlong)

# level t: all streaks born, noise mostly dead
t = -0.38

# ---- figure -------------------------------------------------------------------
fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
axF = fig.add_axes([0.085, 0.05, 0.40, 0.90])
axB = fig.add_axes([0.54, 0.10, 0.45, 0.80])

# asymmetric norm: low speed (the subject) in full blue, high speed only pale rose
axF.imshow(u, cmap=fs.CMAP_DIV, norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.4),
           origin="lower", extent=[0, Lz, 0, Lx], interpolation="bilinear")
# contour a periodically padded field so islands that wrap around the edge
# are cut cleanly by the frame instead of leaving broken fragments
pad = 3
up = np.pad(u, pad, mode="wrap")
zp = np.arange(-pad, nz + pad) * Lz / nz
xp = np.arange(-pad, nx + pad) * Lx / nx
axF.contour(zp, xp, up, levels=[t], colors=[fs.INK], linewidths=1.8,
           linestyles="solid")
axF.set_xlim(0, Lz); axF.set_ylim(0, Lx)
fs.clean(axF)
for s in axF.spines.values():
    s.set_visible(True); s.set_color(fs.MUTED); s.set_linewidth(0.8)
axF.annotate("", xy=(-0.09, 0.80), xytext=(-0.09, 0.62), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=1.8, mutation_scale=14))
axF.text(-0.09, 0.44, "flow", transform=axF.transAxes, rotation=90, ha="center",
         va="center", fontsize=13, color=fs.INK)

# barcode
yy = np.arange(len(h0))[::-1]
for i, ((b, d), l) in enumerate(zip(h0, life)):
    col = fs.STEEL if l > 0.45 else fs.MUTED
    lw = 4.2 if l > 0.45 else 2.2
    axB.plot([b, d], [yy[i], yy[i]], color=col, lw=lw, solid_capstyle="butt")
axB.axvline(t, color=fs.INK, lw=1.6, ls=(0, (3, 2.2)))
fs.clean(axB)
axB.set_xlim(-1.05, top + 0.02)
axB.set_ylim(-1, len(h0))
axB.text(top, yy[3], "streaks", fontsize=13, color=fs.STEEL, ha="right", va="center")
noise_rows = yy[nlong:]
axB.text(t + 0.12, noise_rows[len(noise_rows) // 3], "noise", fontsize=13,
         color=fs.MUTED, ha="left", va="center")

# consistency check: islands of {u <= t} on the torus == bars crossing t
from scipy.ndimage import label
lab, n = label(u <= t)
parent = list(range(n + 1))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
    if a and b:
        parent[find(a)] = find(b)
print("islands at t:", len({find(i) for i in range(1, n + 1)}),
      " bars crossing t:", int(np.sum((h0[:, 0] <= t) & (h0[:, 1] > t))))

fs.save(fig, OUT + ".png")

from PIL import Image
im = Image.open(OUT + ".png")
print(im.size)
im.resize((320, int(320 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_small.png")
