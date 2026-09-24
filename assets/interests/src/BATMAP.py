"""BATMAP: splitting a synthetic, tightly packed brown-fat tissue image into
cells whose membranes are faint and broken.

Synthetic image only (no real data): a warped, Lloyd-relaxed Voronoi packing
of cells. Membranes are thin stained lines with faint stretches and up to 16
gaps. A gap is placed only on a wall between two cells, at least 20 px from a
triple junction and 45 px from other gaps, where both cells' seeds are at
about the same distance along the whole opening; there the two fronts meet
inside the gap instead of one leaking through and taking a piece of the
neighbouring cell. Most cells are multilocular (many small unstained lipid droplets); two
have one large droplet (white-fat-like), one in each part of the figure.
Blur + noise.

Segmentation: one seed per cell, at the innermost point of the cell (maximum
of its distance transform). Every seed launches a front that solves the
eikonal equation |grad T| = 1 / F by fast marching (the level-set method for
a front that only moves outwards), with speed F = 1 in cytoplasm and
droplets and small on stained membrane pixels. The level sets of the arrival
time T are the moving contours; each pixel goes to the seed whose front
arrives first, so neighbouring fronts meet on the membranes and also inside
the membrane gaps, which closes the broken outlines.

Left part: the raw image. Right part: the same tissue (lightened) with the
front positions at equal time steps (thin lines) and the cell outlines where
fronts meet (thick lines); outline stretches that lie in a membrane gap, i.e.
were closed by the meeting fronts, are drawn in carmine. The script asserts
that at most 3 % of the true-membrane pixels (outside gaps) in the right part
lie more than 4 px from a computed outline.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np

sys.path.insert(0, _SRCDIR)
import figstyle as fs
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import skfmm

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NAME = sys.argv[2] if len(sys.argv) > 2 else "BATMAP"
OUT = (_FIGDIR + "/")
rng = np.random.default_rng(SEED)

H, W = 450, 600
yy, xx = np.mgrid[0:H, 0:W].astype(float)

# ------------------------------------------------------------ cell packing
n_cells = 14
pad = 80
pts = np.c_[rng.uniform(-pad, W + pad, n_cells + 10),
            rng.uniform(-pad, H + pad, n_cells + 10)]
for _ in range(6):                                   # Lloyd relaxation
    gy, gx = np.mgrid[-pad:H + pad:4, -pad:W + pad:4]
    g = np.c_[gx.ravel(), gy.ravel()]
    lab = cKDTree(pts).query(g)[1]
    for k in range(len(pts)):
        m = lab == k
        if m.any():
            pts[k] = g[m].mean(0)

# smooth random warp so cell walls are gently curved
def smooth_field(scale, amp):
    f = ndi.gaussian_filter(rng.standard_normal((H, W)), scale)
    return amp * f / np.abs(f).max()

wx, wy = smooth_field(40, 14), smooth_field(40, 14)
labels = cKDTree(pts).query(np.c_[(xx + wx).ravel(), (yy + wy).ravel()])[1]
labels = labels.reshape(H, W)
present = [k for k in range(len(pts)) if (labels == k).sum() > 120]
big = [k for k in present if (labels == k).sum() > 6000]

# ---------------------------------------------------------------- membranes
edge = np.zeros((H, W), bool)
edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
edge[1:, :] |= labels[1:, :] != labels[:-1, :]
memb = ndi.binary_dilation(edge, iterations=1)
faint = 0.5 + 0.5 * (smooth_field(12, 1.0) > -0.1)     # faint stretches
faint = ndi.gaussian_filter(faint, 3)

# one seed per cell: its innermost point (maximum of the distance transform)
seeds = {}
for k in present:
    dt = ndi.distance_transform_edt(np.pad(labels == k, 1))[1:-1, 1:-1]
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
    seeds[k] = (float(cy), float(cx))


def n_labels(lab, rad):
    """Number of distinct labels in the (2 rad + 1)^2 neighbourhood."""
    P = np.pad(lab, rad, mode="edge")
    n = 2 * rad + 1
    st = np.stack([P[i:i + H, j:j + W] for i in range(n) for j in range(n)], -1)
    st.sort(-1)
    return 1 + (np.diff(st, axis=-1) != 0).sum(-1)


# membrane gaps (broken stretches). A gap is placed only where the two
# neighbouring fronts arrive at about the same time (|d_a - d_b| < 4 px),
# away from triple junctions and from other gaps, so the fronts meet inside
# the gap instead of one front leaking through it into the next cell.
OPEN_TOL = 10.0         # px, max |d_a - d_b| over the opening
junction = n_labels(labels, 2) >= 3
far_junc = ndi.distance_transform_edt(~junction) >= 20
gaps = np.zeros((H, W), bool)
gap_list = []
ey, ex = np.nonzero(edge)
for i in rng.permutation(len(ey)):
    y_, x_ = ey[i], ex[i]
    if not far_junc[y_, x_]:
        continue
    nb = set(labels[max(y_ - 1, 0):y_ + 2, max(x_ - 1, 0):x_ + 2].ravel().tolist())
    if len(nb) != 2 or not nb <= set(seeds):
        continue
    a, b = nb
    da = np.hypot(y_ - seeds[a][0], x_ - seeds[a][1])
    db = np.hypot(y_ - seeds[b][0], x_ - seeds[b][1])
    if abs(da - db) >= 4:
        continue
    # ... and about the same time along the whole opening, not only at its
    # centre (an oblique wall lets one front slip through the gap's side)
    near = (ey - y_) ** 2 + (ex - x_) ** 2 < 17 ** 2
    oy, ox = ey[near], ex[near]
    dd = (np.hypot(oy - seeds[a][0], ox - seeds[a][1])
          - np.hypot(oy - seeds[b][0], ox - seeds[b][1]))
    if np.abs(dd).max() >= OPEN_TOL:
        continue
    if any((y_ - p) ** 2 + (x_ - q) ** 2 < 45 ** 2 for p, q, _ in gap_list):
        continue
    gap_list.append((y_, x_, rng.uniform(9, 15)))
    if len(gap_list) == 16:
        break
for cy, cx, r in gap_list:
    gaps |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2

# ------------------------------------------------------------------ droplets
stain = np.full((H, W), 0.42)                         # cytoplasm
drop = np.zeros((H, W), bool)
SPLIT = int(0.42 * W)          # left part: raw image; right part: fronts
cxs = {k: xx[labels == k].mean() for k in big}
big_l = [k for k in big if cxs[k] < SPLIT]
big_r = [k for k in big if cxs[k] > SPLIT + 40]
uni = {int(rng.choice(big_l)), int(rng.choice(big_r))}   # one on each side
for k in present:
    cell = labels == k
    inner = ndi.distance_transform_edt(cell)
    if k in uni:                                     # one big droplet
        drop |= inner > rng.uniform(4.5, 6.5)
        continue
    cand = np.argwhere(inner > 6)
    rng.shuffle(cand)
    placed = []
    for (cy, cx) in cand[:1500]:
        r = rng.uniform(3.5, 8.0)
        if inner[cy, cx] < r + 2.5:
            continue
        if all((cy - py) ** 2 + (cx - px) ** 2 > (r + pr + 2.2) ** 2
               for py, px, pr in placed):
            placed.append((cy, cx, r))
    for cy, cx, r in placed:
        drop |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
stain[drop] = 0.04
m_on = memb & ~gaps
stain = np.where(m_on, 0.42 + 0.48 * faint, stain)
img = ndi.gaussian_filter(stain, 0.9) + rng.normal(0, 0.035, (H, W))
img = img.clip(0, 1)

# ---------------------------------------------------- competing fronts (FMM)
sm = ndi.gaussian_filter(img, 1.2)
speed = np.exp(-np.clip(sm - 0.45, 0, None) / 0.025)  # slow on membranes
speed = np.clip(speed, 0.02, 1.0)
T = np.full((H, W), np.inf)
L = np.full((H, W), -1)
for k, (cy, cx) in seeds.items():
    phi = np.hypot(yy - cy, xx - cx) - 2.0
    t = np.asarray(skfmm.travel_time(phi, speed, dx=1.0))
    better = t < T
    T[better], L[better] = t[better], k

# ------------------------------------------------------------------ figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path

fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img, cmap=fs.CMAP_SEQ, vmin=0, vmax=1, interpolation="bilinear",
          extent=(0, W, H, 0))
veil = np.zeros((H, W, 4))
veil[..., :3] = 1.0
veil[:, SPLIT:, 3] = 0.55
ax.imshow(veil, extent=(0, W, H, 0), interpolation="nearest")
right = Rectangle((SPLIT, 0), W - SPLIT, H, transform=ax.transData,
                  visible=False)
ax.add_patch(right)

# front positions at equal time steps (level sets of the arrival time)
slow = ndi.binary_dilation(speed < 0.5, iterations=3)
cs = ax.contour(xx, yy, np.ma.masked_where(slow, T),
                levels=np.arange(16, 400, 16), colors=fs.SKY, linewidths=1.6)
cs.set_clip_path(right)

# cell outlines = where fronts from different seeds meet
bd = np.zeros((H, W), bool)
bd[:, 1:] |= L[:, 1:] != L[:, :-1]
bd[1:, :] |= L[1:, :] != L[:-1, :]

# check: the true membranes (outside the gaps) in the right part must lie on
# the computed outlines; count membrane pixels more than 4 px from an outline
true_m = edge & ~gaps
true_m[:, :SPLIT] = False
off = true_m & (ndi.distance_transform_edt(~bd) > 4)
n_true, n_off = int(true_m.sum()), int(off.sum())
print(f"membrane px (right, outside gaps): {n_true}, >4 px from outline: "
      f"{n_off} ({100 * n_off / n_true:.1f} %)")
assert n_off <= 0.03 * n_true, "segmentation misses membranes"

for k in seeds:
    if (L[:, SPLIT:] == k).sum() < 50:
        continue
    ind = ndi.gaussian_filter((L == k).astype(float), 1.5)
    c = ax.contour(xx, yy, ind, levels=[0.5], colors=fs.INK, linewidths=2.4,
                   zorder=3)
    c.set_clip_path(right)

# the same outlines again, in carmine, only where the membrane was missing:
# these stretches were closed by the meeting fronts
gap_paths = []
for cy, cx, r in gap_list:
    if cx > SPLIT + r + 4:          # never cut at the white divider
        gap_paths.append(Path.circle((cx, cy), r))
gap_clip = Path.make_compound_path(*gap_paths)
for k in seeds:
    if (L[:, SPLIT:] == k).sum() < 50:
        continue
    ind = ndi.gaussian_filter((L == k).astype(float), 1.5)
    c = ax.contour(xx, yy, ind, levels=[0.5], colors=fs.CARMINE,
                   linewidths=3.4, zorder=4)
    c.set_clip_path(gap_clip, ax.transData)

# label one drawn gap: the label box (about 132 x 33 px, right below or
# right above the gap) must stay inside the right part and cover as little
# of the outlines as possible
HW, HB = 66, 33
bd_wide = ndi.binary_dilation(bd, iterations=3)
best, best_score = None, np.inf
for cy, cx, r in gap_list:
    if cx <= SPLIT + r + 4:
        continue
    for side in (+1, -1):                        # +1: below, -1: above
        y0 = cy + r + 6 if side > 0 else cy - r - 6 - HB
        x0, x1, y1 = cx - HW, cx + HW, y0 + HB
        if x0 < SPLIT + 8 or x1 > W - 6 or y0 < 6 or y1 > H - 6:
            continue
        score = bd_wide[int(y0):int(y1), int(x0):int(x1)].sum()
        if score < best_score:
            best, best_score = (cy, cx, r, side), score
if best is not None:
    cy, cx, r, side = best
    ax.text(cx, cy + side * (r + 6), "closed gap", ha="center",
            va="top" if side > 0 else "bottom", fontsize=13,
            color=fs.CARMINE, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc=fs.WHITE, ec="none",
                      alpha=0.9))
ax.axvline(SPLIT, color=fs.WHITE, lw=4, zorder=5)
ax.set_xlim(0, W); ax.set_ylim(H, 0)
fs.clean(ax)

fs.save(fig, OUT + NAME + ".png")
from PIL import Image
im = Image.open(OUT + NAME + ".png")
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    OUT + NAME + "_small.png")
assert min(cxs[k] for k in uni) < SPLIT < max(cxs[k] for k in uni)
print("cells", len(present), "uni", sorted(uni),
      "uni centroid x", [round(cxs[k]) for k in sorted(uni)],
      "gaps", len(gap_list), "carmine gaps", len(gap_paths))
