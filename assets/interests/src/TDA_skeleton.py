"""TDA_skeleton: a synthetic microglia-like tree and its Topological Morphology
Descriptor (TMD) barcode.

Left : a random 2-D branching tree grown from a cell body (soma); faint rings
       mark the radial distance from the soma. Every edge is coloured by the
       bar it belongs to (tip -> branch point where it merges into a branch
       that reaches farther).
Right: the TMD barcode (Kanari et al. 2018), computed from scratch. Each bar
       runs from the radial distance of the branch point where the branch dies
       to the radial distance of its tip; the longest branch survives to the
       soma (distance 0). Bars sorted by length; the rings on the left are the
       vertical grid lines on the right.
One branch and its bar are drawn in carmine.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np

sys.path.insert(0, _SRCDIR)
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 13
OUT = (_FIGDIR + "/")
NAME = sys.argv[2] if len(sys.argv) > 2 else "TDA_skeleton"
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- grow tree
pos = [np.zeros(2)]          # node coordinates
parent = [-1]                # parent index


def grow(start, angle, length, depth):
    """Grow a tortuous branch from node `start`, then maybe bifurcate."""
    n_seg = max(3, int(length / 0.16))
    cur = start
    a = angle
    for _ in range(n_seg):
        a += rng.normal(0.0, 0.13)
        p = pos[cur] + (length / n_seg) * np.array([np.cos(a), np.sin(a)])
        pos.append(p)
        parent.append(cur)
        cur = len(pos) - 1
    if depth > 0:
        spread = rng.uniform(0.45, 0.85)
        for s in (-1, 1):
            if rng.random() < 0.9:
                grow(cur, a + s * spread + rng.normal(0, 0.1),
                     length * rng.uniform(0.5, 0.95), depth - 1)


n_primary = 5
base = rng.uniform(0, 2 * np.pi)
for k in range(n_primary):
    ang = base + 2 * np.pi * k / n_primary + rng.normal(0, 0.2)
    grow(0, ang, rng.uniform(0.8, 1.3), depth=int(rng.integers(2, 4)))

pos = np.array(pos)
parent = np.array(parent)
N = len(pos)
children = [[] for _ in range(N)]
for v in range(1, N):
    children[parent[v]].append(v)
f = np.linalg.norm(pos, axis=1)            # radial distance from soma

# ------------------------------------------------------ TMD (from scratch)
order, stack = [], [0]
while stack:                                # DFS, then reverse = post-order
    v = stack.pop()
    order.append(v)
    stack.extend(children[v])
order = order[::-1]

alive = np.zeros(N)          # largest tip value carried through node v
owner = np.zeros(N, int)     # tip whose component passes through v
bars = []                    # (death, birth, tip)
for v in order:
    ch = children[v]
    if not ch:
        alive[v], owner[v] = f[v], v
        continue
    vals = [alive[c] for c in ch]
    j = int(np.argmax(vals))                # elder rule: farthest tip survives
    alive[v], owner[v] = vals[j], owner[ch[j]]
    for i, c in enumerate(ch):
        if i != j:
            bars.append((f[v], alive[c], owner[c]))
bars.append((0.0, alive[0], owner[0]))      # survivor dies at the soma
bars = np.array(bars)

tip_to_bar = {int(t): i for i, t in enumerate(bars[:, 2])}
edge_bar = np.array([tip_to_bar[int(owner[v])] for v in range(1, N)])
length = bars[:, 1] - bars[:, 0]

# highlight: a mid-length branch that does not reach the soma
cand = np.argsort(length)
hl = int(cand[int(0.70 * len(cand))])

cmap = LinearSegmentedColormap.from_list("bars", [fs.SKY, fs.STEEL, fs.INK])
norm = plt.Normalize(length.min(), length.max())

# ------------------------------------------------------------------ figure
fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
TREE_W = 0.55
axT = fig.add_axes([0.0, 0.0, TREE_W, 1.0])
axB = fig.add_axes([0.575, 0.20, 0.36, 0.72])

R = f.max() * 1.02
rings = np.arange(1, int(R) + 1) * 1.0
for r in rings:
    axT.add_patch(plt.Circle((0, 0), r, fill=False, color=fs.PALE, lw=1.8,
                             zorder=0))

segs = np.stack([pos[1:], pos[parent[1:]]], axis=1)
is_hl = edge_bar == hl
axT.add_collection(LineCollection(
    segs[~is_hl], colors=[cmap(norm(length[b])) for b in edge_bar[~is_hl]],
    linewidths=3.0, capstyle="round", zorder=2))
axT.add_collection(LineCollection(segs[is_hl], colors=fs.CARMINE,
                                  linewidths=4.2, capstyle="round", zorder=3))
axT.scatter([0], [0], s=480, color=fs.INK, zorder=4)
# frame: soma at the centre, outermost tip just inside the panel
box_w, box_h = TREE_W * fs.W_IN, 1.0 * fs.H_IN
hx = R * 1.03
axT.set_xlim(-hx, hx)
axT.set_ylim(-hx * box_h / box_w, hx * box_h / box_w)
axT.set_aspect("equal")
fs.clean(axT)

idx = np.lexsort((bars[:, 0], -length))     # longest first
for y, b in enumerate(idx):
    col = fs.CARMINE if b == hl else cmap(norm(length[b]))
    axB.plot([bars[b, 0], bars[b, 1]], [y, y], color=col,
             lw=5.0 if b == hl else 4.6, solid_capstyle="butt", zorder=3)
for r in rings:
    axB.axvline(r, color=fs.PALE, lw=1.8, zorder=0)
axB.set_xlim(0, R)
axB.set_ylim(len(idx) - 0.4, -0.6)
axB.set_yticks([])
axB.set_xticks([])
axB.spines["left"].set_visible(False)
axB.spines["bottom"].set_linewidth(1.8)
axB.spines["bottom"].set_color(fs.INK)
axB.set_xlabel("distance from cell body", fontsize=13, color=fs.INK,
               labelpad=7)

fs.save(fig, OUT + NAME + ".png")

from PIL import Image
im = Image.open(OUT + NAME + ".png")
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    OUT + NAME + "_small.png")
print("seed", SEED, "tips:", len(bars), "nodes:", N)
