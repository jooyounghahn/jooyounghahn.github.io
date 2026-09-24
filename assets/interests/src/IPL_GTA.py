"""IPL_GTA: the pores of a synthetic 2-D 'sponge', seen two ways.

Left : pore space coloured by distance to the nearest wall (the top and bottom
       edges of the sample count as walls).  The critical
       radius r_c is the largest disk that can still travel from the left face
       to the right face (largest r for which {distance >= r} connects the two
       faces).  White line: a route of the disk centre; carmine circle: the disk
       of radius r_c at the bottleneck of that route.
Right: steady diffusion (Laplace equation, c = 1 on the left face, c = 0 on the
       right face, no flux through walls, top and bottom) solved by finite
       volumes on the pore pixels.  Flux lines start on the left face at equal
       steps of the inflow, so each line carries the same flux: crowded lines
       are highways, empty pores are dead ends.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import heapq
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

SCR = (_SRCDIR)
sys.path.insert(0, SCR)
import figstyle as fs  # noqa: E402
from matplotlib.colors import ListedColormap, PowerNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

SEED = 10                                        # chosen from a scan of seeds
OUT = _FIGDIR + "/IPL_GTA"

NY, NX = 230, 180

# ---------------------------------------------------------------- geometry
rng = np.random.default_rng(SEED)
yy, xx = np.mgrid[0:NY, 0:NX]
solid = np.zeros((NY, NX), bool)
while solid.mean() < 0.47:                       # overlapping disk grains
    cx, cy = rng.uniform(-10, NX + 10), rng.uniform(-10, NY + 10)
    r = rng.uniform(6, 15)
    solid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
pore = ~solid
# top and bottom of the sample are walls (no flux); left and right are open
dist = ndimage.distance_transform_edt(np.pad(pore, ((1, 1), (0, 0))))[1:-1]

# pores connected to the left or right face take part in transport
lab, _ = ndimage.label(pore)
touch = (set(np.unique(lab[:, 0])) | set(np.unique(lab[:, -1]))) - {0}
active = np.isin(lab, list(touch))
isolated = pore & ~active


# ----------------------------------------------------------- critical radius
def crosses(r):
    lab_r, _ = ndimage.label(dist >= r)
    left = set(np.unique(lab_r[:, 0])) - {0}
    right = set(np.unique(lab_r[:, -1])) - {0}
    return len(left & right) > 0


vals = np.unique(dist[pore])
lo, hi = 0, len(vals) - 1
while lo < hi:                                   # bisection on the level
    m = (lo + hi + 1) // 2
    if crosses(vals[m]):
        lo = m
    else:
        hi = m - 1
r_crit = vals[lo]
print(f"seed {SEED}  porosity {pore.mean():.3f}  critical radius {r_crit:.2f} px")

# route of the disk centre inside {dist >= r_c}: Dijkstra with a cost that
# prefers pore centres (cost 1/dist per step), from left face to right face
ok = dist >= r_crit - 1e-9
cost = 1.0 / np.maximum(dist, 1e-3)
D = np.full((NY, NX), np.inf)
prev = -np.ones((NY, NX, 2), int)
heap = []
for i in range(NY):
    if ok[i, 0]:
        D[i, 0] = cost[i, 0]
        heap.append((D[i, 0], i, 0))
heapq.heapify(heap)
end = None
steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
while heap:
    d0, i, j = heapq.heappop(heap)
    if d0 > D[i, j]:
        continue
    if j == NX - 1:
        end = (i, j)
        break
    for di, dj, h in steps:
        a, b = i + di, j + dj
        if 0 <= a < NY and 0 <= b < NX and ok[a, b]:
            nd = d0 + h * 0.5 * (cost[i, j] + cost[a, b])
            if nd < D[a, b]:
                D[a, b] = nd
                prev[a, b] = (i, j)
                heapq.heappush(heap, (nd, a, b))
path = [end]
while prev[path[-1]][0] >= 0:
    path.append(tuple(prev[path[-1]]))
path = np.array(path[::-1], float)
dpath = dist[path[:, 0].astype(int), path[:, 1].astype(int)]
k = int(np.argmin(dpath))
by, bx = path[k]
print(f"bottleneck at x={bx:.0f}, y={by:.0f}")
# smooth the drawn route a little (display only)
sm = ndimage.uniform_filter1d(path, 7, axis=0, mode="nearest")

# -------------------------------------------------- finite-volume diffusion
idx = -np.ones((NY, NX), int)
cells = np.argwhere(active)
n = len(cells)
idx[active] = np.arange(n)
rows, cols, vals_ = [], [], []
rhs = np.zeros(n)
diag = np.zeros(n)
for (i, j) in cells:
    p = idx[i, j]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = i + di, j + dj
        if 0 <= a < NY and 0 <= b < NX:
            if active[a, b]:
                rows.append(p); cols.append(idx[a, b]); vals_.append(-1.0)
                diag[p] += 1.0
        elif b < 0:              # left face, c = 1 (half-cell distance)
            diag[p] += 2.0; rhs[p] += 2.0
        elif b >= NX:            # right face, c = 0
            diag[p] += 2.0
        # top, bottom and walls: no flux
rows += list(range(n)); cols += list(range(n)); vals_ += list(diag)
A = coo_matrix((vals_, (rows, cols)), shape=(n, n)).tocsr()
c = np.zeros((NY, NX))
c[active] = spsolve(A, rhs)

fx = np.zeros((NY, NX + 1))                    # face fluxes (unit spacing)
fy = np.zeros((NY + 1, NX))
fx[:, 1:-1] = np.where(active[:, 1:] & active[:, :-1], c[:, :-1] - c[:, 1:], 0.0)
fx[:, 0] = np.where(active[:, 0], 2 * (1 - c[:, 0]), 0.0)
fx[:, -1] = np.where(active[:, -1], 2 * c[:, -1], 0.0)
fy[1:-1, :] = np.where(active[1:, :] & active[:-1, :], c[:-1, :] - c[1:, :], 0.0)
Jin, Jout = fx[:, 0].sum(), fx[:, -1].sum()
eps = pore.mean()
tau = eps * (1.0 / NX) / (Jin / NY)             # tortuosity factor eps*D/D_eff
print(f"flux in {Jin:.4f}  out {Jout:.4f}  tortuosity factor {tau:.2f}")

# ------------------------------------------------------------------ figure
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
pw = 0.47                                        # panel width (figure frac.)
ph = pw * fs.W_IN / fs.H_IN * NY / NX            # keep pixels square
axL = fig.add_axes([0.02, 0.025, pw, ph])
axR = fig.add_axes([0.02 + pw + 0.02, 0.025, pw, ph])

grain = ListedColormap([fs.MUTED])
iso = ListedColormap([fs.PALE])
bg = ListedColormap([fs.CMAP_SEQ(0.0)])


def draw_solid(ax, z):
    ax.imshow(np.where(solid, 1.0, np.nan), cmap=grain, origin="lower",
              interpolation="nearest", zorder=z)
    ax.imshow(np.where(isolated, 1.0, np.nan), cmap=iso, origin="lower",
              interpolation="nearest", zorder=z)


# left: distance to wall, route of the disk centre, disk of radius r_c
draw_solid(axL, 1)
axL.imshow(np.where(active, dist, np.nan), cmap=fs.CMAP_SEQ,
           norm=PowerNorm(0.6, vmin=0, vmax=dist[active].max()),
           origin="lower", interpolation="nearest", zorder=1)
axL.plot(sm[:, 1], sm[:, 0], color=fs.WHITE, lw=2.2, solid_capstyle="round", zorder=3)
axL.add_patch(Circle((bx, by), r_crit, facecolor=fs.CARMINE, alpha=0.35,
                     edgecolor="none", zorder=4))
axL.add_patch(Circle((bx, by), r_crit, facecolor="none", edgecolor=fs.CARMINE,
                     lw=2.6, zorder=4))

# right: flux lines; start points on the left face at equal steps of the
# cumulative inflow, so every line carries the same share of the flux
qx = np.where(active, 0.5 * (fx[:, 1:] + fx[:, :-1]), 0.0)
qy = np.where(active, 0.5 * (fy[1:, :] + fy[:-1, :]), 0.0)
cum = np.concatenate([[0], np.cumsum(fx[:, 0])])
NL = 26
levels = (np.arange(NL) + 0.5) / NL * cum[-1]
y0 = np.interp(levels, cum, np.arange(NY + 1) - 0.5)
starts = np.column_stack([np.full(NL, 0.5), y0])
axR.imshow(np.where(pore, 1.0, np.nan), cmap=bg, origin="lower",
           interpolation="nearest", zorder=0)
axR.streamplot(np.arange(NX), np.arange(NY), qx, qy, start_points=starts,
               color=fs.STEEL, linewidth=1.7, arrowsize=0, density=40,
               broken_streamlines=False, integration_direction="forward", zorder=1)
draw_solid(axR, 2)                  # grains on top hide corner-cutting lines

for ax in (axL, axR):
    fs.clean(ax)
    ax.set_xlim(-0.5, NX - 0.5)
    ax.set_ylim(-0.5, NY - 0.5)

kw = dict(fontsize=13, ha="center", va="bottom", color=fs.INK)
axL.text(NX / 2, NY + 5, "distance to wall", **kw)
axR.text(NX / 2, NY + 5, "diffusion flux", **kw)
# label the disk: place text above it, inside the panel if possible
tx = min(max(bx - 40, 50), NX - 52)
axL.annotate("bottleneck", xy=(bx, by + r_crit + 1), xytext=(tx, by + r_crit + 22),
             ha="center", va="bottom", fontsize=13, color=fs.INK, zorder=6,
             bbox=dict(boxstyle="round,pad=0.15", fc=fs.WHITE, ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="-", color=fs.INK, lw=1.6, shrinkA=0, shrinkB=0))

fs.save(fig, OUT + ".png")

from PIL import Image  # noqa: E402
im = Image.open(OUT + ".png")
im.resize((320, round(320 * im.height / im.width)), Image.LANCZOS).save(OUT + "_small.png")
print(im.size)
