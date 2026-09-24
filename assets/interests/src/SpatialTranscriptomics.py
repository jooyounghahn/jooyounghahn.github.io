"""Spatial transcriptomics research-interest figure (synthetic toy example).

A synthetic tissue slice (blob with a slit-shaped cavity) holds two cell types:
oligodendrocytes carrying a state score (a planted high-score hotspot plus a
weaker second one) and astrocytes.  Steps:
  1. score-weighted Gaussian density f of the oligodendrocytes inside tissue;
  2. superlevel-set persistence of f (GUDHI cubical complex on -f): the most
     persistent peak is kept, and the high-score region is its superlevel
     component at the level halfway between the peak and the saddle where it
     merges with the next peak;
  3. geodesic distance to that region inside the tissue, i.e. the solution of
     the eikonal equation |grad u| = 1 with u = 0 on the region, computed by
     the fast marching method (scikit-fmm) with the cavity masked out;
  4. for one astrocyte behind the cavity: geodesic path (descent on u) and the
     straight segment for comparison.
Background: geodesic distance (light = far), iso-distance lines.  No real data.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys

import gudhi
import numpy as np
import skfmm
import matplotlib.patheffects as pe
from scipy import ndimage

SP = _SRCDIR
sys.path.insert(0, SP)
import figstyle as fs  # noqa: E402

rng = np.random.default_rng(11)

# ---------------------------------------------------------------- tissue domain
nx_, ny_ = 400, 300                              # grid, 4:3
h = 1.0 / 100.0                                  # grid spacing (domain 4 x 3)
X, Y = np.meshgrid(np.arange(nx_) * h, np.arange(ny_) * h)
cx, cy = 2.0, 1.45
th = np.arctan2(Y - cy, X - cx)
wobble = 1 + 0.05 * np.sin(3 * th + 0.4) + 0.03 * np.sin(5 * th + 1.3)
g_out = np.hypot((X - cx) / 1.8, (Y - cy) / 1.12) - wobble   # < 0 inside slice
# curved slit-shaped cavity (ventricle-like gap): arc around (2.65, 1.5)
ac = np.array([2.65, 1.45])
arc_r = np.hypot(X - ac[0], Y - ac[1])
arc_th = np.abs(np.angle(np.exp(1j * (np.arctan2(Y - ac[1], X - ac[0]) - np.pi))))
half_w, half_ang, R = 0.055, np.deg2rad(62), 0.62
tip = np.hypot(np.maximum(arc_th - half_ang, 0) * R, 0)
g_cav = np.hypot(np.abs(arc_r - R), tip) - half_w           # < 0 inside cavity
tissue = (g_out < 0) & (g_cav > 0)


def inside(p):
    i = np.clip((p[:, 1] / h).astype(int), 0, ny_ - 1)
    j = np.clip((p[:, 0] / h).astype(int), 0, nx_ - 1)
    return tissue[i, j]


def sample(n):
    out = np.empty((0, 2))
    while len(out) < n:
        p = rng.uniform([0, 0], [nx_ * h, ny_ * h], size=(4 * n, 2))
        out = np.vstack([out, p[inside(p)]])
    return out[:n]


# ---------------------------------------------------------------- cells + scores
oligo = sample(170)
astro = sample(34)
hot1, hot2 = np.array([1.0, 1.45]), np.array([2.95, 2.2])
score = (0.9 * np.exp(-np.sum((oligo - hot1) ** 2, 1) / (2 * 0.32 ** 2))
         + 0.5 * np.exp(-np.sum((oligo - hot2) ** 2, 1) / (2 * 0.25 ** 2))
         + 0.12 * rng.random(len(oligo)))
score = np.clip(score, 0, 1)

# score-weighted Gaussian density on the grid, restricted to tissue
img = np.zeros((ny_, nx_))
ii = (oligo[:, 1] / h).astype(int)
jj = (oligo[:, 0] / h).astype(int)
np.add.at(img, (ii, jj), score)
f = ndimage.gaussian_filter(img, sigma=22)
f[~tissue] = 0.0

# ---------------------------------------------------------------- persistence
cc = gudhi.CubicalComplex(top_dimensional_cells=-f)
pers = cc.persistence()
h0 = sorted([(b, d) for dim, (b, d) in pers if dim == 0],
            key=lambda bd: (bd[1] - bd[0]), reverse=True)
fmax = -h0[0][0]                                 # essential class: global max
saddle = -h0[1][1]                               # where the 2nd peak merges
tau = 0.5 * (fmax + saddle)
lab, _ = ndimage.label(f >= tau)
imax = np.unravel_index(np.argmax(f), f.shape)
region = lab == lab[imax]

# ---------------------------------------------------------------- eikonal distance
phi = np.where(region, -1.0, 1.0)
phi = np.ma.MaskedArray(phi, mask=~tissue)
u = skfmm.distance(phi, dx=h)
u = np.ma.filled(u, np.nan)
u[region] = 0.0

# geodesic path from an astrocyte behind the cavity: steepest descent on u
gy, gx = np.gradient(np.nan_to_num(u, nan=np.nanmax(u) + 1), h)
start = np.array([2.72, 1.4])                    # astrocyte inside the arc
# label boxes (x0, x1, y0, y1): keep them free of cell markers
LBL_ASTRO = (start[0] + 0.1, start[0] + 1.0, start[1] - 0.14, start[1] + 0.14)
LBL_PATH = (1.75, 2.95, 0.42, 0.72)
LBL_REGION = (0.28, 1.75, 2.02, 2.32)


def free(pts, boxes):
    keep = np.ones(len(pts), bool)
    for x0, x1, y0, y1 in boxes:
        keep &= ~((pts[:, 0] > x0) & (pts[:, 0] < x1)
                  & (pts[:, 1] > y0) & (pts[:, 1] < y1))
    return keep


astro = astro[free(astro, [LBL_ASTRO, LBL_PATH, LBL_REGION])
              & (np.linalg.norm(astro - start, axis=1) > 0.2)]
keep_o = free(oligo, [LBL_ASTRO, LBL_PATH, LBL_REGION])
oligo, score = oligo[keep_o], score[keep_o]
path = [start.copy()]
p = start.copy()
for _ in range(4000):
    i, j = int(p[1] / h), int(p[0] / h)
    if region[i, j]:
        break
    g = np.array([gx[i, j], gy[i, j]])
    p = p - 0.004 * g / (np.linalg.norm(g) + 1e-12)
    path.append(p.copy())
path = np.array(path)
i0, j0 = int(start[1] / h), int(start[0] / h)
geo_len = u[i0, j0]
# straight segment to the nearest region point
ri, rj = np.where(region)
rp = np.column_stack([rj, ri]) * h
straight_end = rp[np.argmin(np.sum((rp - start) ** 2, 1))]
eu_len = np.linalg.norm(straight_end - start)

# ---------------------------------------------------------------- figure
fs.setup()
fig, ax = fs.figure()
fig.subplots_adjust(0, 0, 1, 1)
ext = (0, nx_ * h, 0, ny_ * h)
cmap = fs.CMAP_SEQ
umax = np.nanmax(u)
shade = np.where(tissue, 0.05 + 0.2 * (1 - np.nan_to_num(u) / umax), np.nan)
ax.imshow(np.ma.masked_invalid(shade), origin="lower", extent=ext, cmap=cmap,
          vmin=0, vmax=1, interpolation="bilinear", zorder=0)
ax.contour(X + h / 2, Y + h / 2, np.ma.masked_invalid(u),
           levels=np.arange(0.25, umax, 0.25), colors=fs.SKY, linewidths=1.0,
           zorder=1)
ax.contour(X, Y, g_out, levels=[0], colors=fs.MUTED, linewidths=1.6, zorder=2)
ax.contour(X, Y, np.where(g_out < 0.02, g_cav, 1), levels=[0], colors=fs.MUTED,
           linewidths=1.6, zorder=2)
ax.contour(X + h / 2, Y + h / 2, np.where(region | ~(f >= tau), f, 0.0),
           levels=[tau], colors=fs.CARMINE, linewidths=2.6, zorder=3)
# cells
ax.scatter(oligo[:, 0], oligo[:, 1], s=14 + 36 * score, c=score,
           cmap=cmap, vmin=-0.55, vmax=1.0, edgecolors="white",
           linewidths=0.5, zorder=4)
ax.scatter(astro[:, 0], astro[:, 1], s=34, marker="D", c=fs.SAND,
           edgecolors="white", linewidths=0.6, zorder=4)
# straight segment vs geodesic path
ax.plot([start[0], straight_end[0]], [start[1], straight_end[1]],
        color=fs.MUTED, lw=1.7, ls=(0, (3, 2.5)), zorder=5)
ax.plot(path[:, 0], path[:, 1], color=fs.INK, lw=2.2, zorder=6)
ax.scatter([start[0]], [start[1]], s=90, marker="D", c=fs.SAND,
           edgecolors=fs.INK, linewidths=1.6, zorder=7)
ax.set_xlim(ext[0], ext[1])
ax.set_ylim(ext[2] - 0.08, ext[3] - 0.08)          # centre the slice
ax.set_aspect("equal")
fs.clean(ax)
# labels
halo = [pe.withStroke(linewidth=3.5, foreground="white")]
ax.text(LBL_REGION[0] + 0.02, LBL_REGION[2] + 0.02, "high-score region",
        fontsize=13, color=fs.CARMINE, ha="left", va="bottom",
        path_effects=halo, zorder=8)
ax.text(LBL_ASTRO[0] + 0.04, start[1], "astrocyte", fontsize=13,
        color=fs.INK, ha="left", va="center", path_effects=halo, zorder=8)
ax.text(0.5 * (LBL_PATH[0] + LBL_PATH[1]), LBL_PATH[3] - 0.02, "geodesic path",
        fontsize=13, color=fs.INK, ha="center", va="top", path_effects=halo,
        zorder=8)

out = _FIGDIR + "/SpatialTranscriptomics.png"
fs.save(fig, out)

from PIL import Image  # noqa: E402

im = Image.open(out)
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    _FIGDIR + "/SpatialTranscriptomics_small.png")
print(im.size, "fmax %.3f saddle %.3f tau %.3f" % (fmax, saddle, tau),
      "region px", int(region.sum()), "geodesic %.2f straight %.2f" % (geo_len, eu_len),
      "start", np.round(start, 2))
