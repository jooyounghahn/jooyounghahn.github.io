"""TDA_ORCA: a Ball Mapper graph of synthetic single-cell chromatin traces.

Synthetic data only (no real traces): 900 random 3-D polymer chains of 36
beads (Gaussian network model: a chain of springs plus extra springs that
fold it), from three folding families with a random folding strength s in
[0, 1]:
  - one compact globule (weak springs between all beads),
  - two domains (weak springs within each half of the chain),
  - a loop (one spring joining two beads near the chain ends).
At s = 0 every family is the same random walk. Condition A and condition B
have 450 traces each and differ only in how often each family occurs
(globule / two domains / loop: A 60/10/30 %, B 10/60/30 %). Each bead is
detected with probability 0.6, as in chromatin tracing where many points are
missing.

Each trace -> rotation- and translation-free numbers: the mean log distance
between detected beads for every pair of 12-bead blocks of the chain (a
coarse internal distance map, 6 numbers, from detected beads only).
Ball Mapper (Dlotko 2019) on these vectors, radius 0.7: greedy epsilon-net
of landmark traces, one node per ball, an edge when two balls share traces;
largest connected component shown. Node colour = share of condition B traces
in the ball; node area ~ number of traces. Node positions: Kamada-Kawai
layout, rotated so the three family arms point left / up-right / down-right.
Insets: median internal distance map (dark = close) of the traces in the
ball at the tip of each arm - loop (left: contact between the chain ends),
globule (top right: everything close), two domains (bottom right: two
blocks); each inset is labelled with its fold name.
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
from matplotlib.colors import Normalize
import networkx as nx

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
NAME = sys.argv[3] if len(sys.argv) > 3 else "TDA_ORCA"
OUT = (_FIGDIR + "/")
rng = np.random.default_rng(SEED)

NB, PDET, WM, BS = 36, 0.6, 4.0, 12
N_PER = 450
MIX = {"A": [0.6, 0.1, 0.3], "B": [0.1, 0.6, 0.3]}


# ------------------------------------------------------------ polymer model
def chain_lap():
    L = np.zeros((NB, NB))
    for i in range(NB - 1):
        L[i, i] += 1; L[i + 1, i + 1] += 1
        L[i, i + 1] -= 1; L[i + 1, i] -= 1
    return L


def add_domain(L, idx, w):
    idx = np.asarray(idx)
    L[np.ix_(idx, idx)] -= w
    L[idx, idx] += w * len(idx)


def add_spring(L, a, b, w):
    L[a, a] += w; L[b, b] += w; L[a, b] -= w; L[b, a] -= w


def sample(L):
    lam, U = np.linalg.eigh(L)          # lam[0] = 0: free translation
    Z = rng.standard_normal((NB - 1, 3))
    return U[:, 1:] @ (Z / np.sqrt(lam[1:, None]))


def make(fam, s):
    L = chain_lap()
    if fam == 0:                        # one globule
        add_domain(L, range(NB), s * WM / NB)
    elif fam == 1:                      # two domains
        add_domain(L, range(0, NB // 2), s * WM / (NB // 2))
        add_domain(L, range(NB // 2, NB), s * WM / (NB // 2))
    else:                               # loop between two anchors
        add_spring(L, 3, NB - 4, s * 6.0)
    return sample(L)


X, F, S, C = [], [], [], []
for ci, key in enumerate("AB"):
    for _ in range(N_PER):
        fam = rng.choice(3, p=MIX[key])
        s = rng.uniform(0, 1)
        X.append(make(fam, s)); F.append(fam); S.append(s); C.append(ci)
X, F, S, C = map(np.array, (X, F, S, C))
N = len(X)
det = rng.random((N, NB)) < PDET

# ------------------------------- coarse internal distance map (detected only)
full = np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=-1)
nbk = NB // BS
feat = []
for a in range(nbk):
    for b in range(a, nbk):
        ia, ib = slice(a * BS, (a + 1) * BS), slice(b * BS, (b + 1) * BS)
        m = det[:, ia][:, :, None] & det[:, ib][:, None, :]
        if a == b:
            m &= ~np.eye(BS, dtype=bool)[None]
        v = np.log(np.where(m, full[:, ia, ib], 1.0).clip(1e-6))
        feat.append((v * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1))
V = np.array(feat).T
D = np.linalg.norm(V[:, None] - V[None], axis=-1)

# --------------------------------------------------------------- Ball Mapper
land = []
for i in rng.permutation(N):
    if not land or D[i, land].min() > EPS:
        land.append(i)
land = np.array(land)
memb = D[land] <= EPS                       # ball membership
size = memb.sum(1)
share = (memb * C).sum(1) / size            # share of condition B
ov = memb.astype(int) @ memb.T.astype(int)
G = nx.Graph()
G.add_nodes_from(range(len(land)))
for a in range(len(land)):
    for b in range(a + 1, len(land)):
        if ov[a, b] > 0:
            G.add_edge(a, b)

# keep the largest connected component
keep = np.array(sorted(max(nx.connected_components(G), key=len)))
remap = {int(k): i for i, k in enumerate(keep)}
land, memb, size, share = land[keep], memb[keep], size[keep], share[keep]
G = nx.relabel_nodes(G.subgraph(keep).copy(), remap)

# arm tip of each family: among balls with >= 10 traces, the one with the
# most strongly folded traces of that family (family share x mean strength)
frac = np.array([np.bincount(F[m], minlength=3) / m.sum() for m in memb])
mean_s = np.array([S[m].mean() for m in memb])
score = frac * mean_s[:, None] * (size >= 10)[:, None]
tip = [int(np.argmax(score[:, f])) for f in range(3)]

# node positions: Kamada-Kawai layout of the graph, turned rigidly
# (rotation, reflection allowed) so that the loop tip points left, the
# globule tip up-right and the two-domain tip down-right
kk = nx.kamada_kawai_layout(G)
P = np.array([kk[k] for k in range(len(land))])
P -= P.mean(0)
tdir = np.array([[1.0, 0.55], [1.0, -0.55], [-1.0, 0.0]])
tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
best = None
for refl in (1.0, -1.0):
    for th in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Qt = (P * [1.0, refl]) @ R_.T
        d = Qt[tip] - Qt.mean(0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sc = (d * tdir).sum()
        if best is None or sc > best[0]:
            best = (sc, Qt)
Q = best[1]
Q -= (Q.min(0) + Q.max(0)) / 2
Q /= np.abs(Q).max()

# median internal distance map of the traces in each tip ball
def median_map(ids):
    M = full[ids].copy()
    ok = det[ids][:, :, None] & det[ids][:, None, :]
    M[~ok] = np.nan
    med = np.nanmedian(M, axis=0)
    np.fill_diagonal(med, 0.0)
    return med

maps = [median_map(np.where(memb[t])[0]) for t in tip]
print("nodes", len(land), "edges", G.number_of_edges())

# ------------------------------------------------------------------ figure
fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
# graph box in figure fractions
gx0, gx1, gy0, gy1 = 0.245, 0.755, 0.15, 0.93
ax = fig.add_axes([gx0, gy0, gx1 - gx0, gy1 - gy0])
div = Normalize(-0.25, 1.25)                # share 0 -> steel, 1 -> carmine

segs = [(Q[a], Q[b]) for a, b in G.edges]
ax.add_collection(LineCollection(segs, colors=fs.MUTED, linewidths=1.6,
                                 alpha=0.38, zorder=1))
order_ = np.argsort(-size)                  # small nodes on top
ax.scatter(Q[order_, 0], Q[order_, 1], s=14 + 1.5 * size[order_],
           c=share[order_], cmap=fs.CMAP_DIV, norm=div, edgecolors=fs.INK,
           linewidths=0.8, zorder=3, clip_on=False)
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
fs.clean(ax)
ax.patch.set_visible(False)
ax.apply_aspect()

w = 0.19
h = w * fs.W_IN / fs.H_IN
boxes = [(0.975 - w / 2, 0.955 - h / 2),   # globule: top right
         (0.975 - w / 2, 0.175 + h / 2),   # two domains: bottom right
         (0.02 + w / 2, 0.56)]             # loop: left
vmax = np.nanpercentile(np.concatenate([m.ravel() for m in maps]), 97)
to_fig = fig.transFigure.inverted()
for fam, nd in zip(range(3), tip):
    fx, fy = boxes[fam]
    nxy = to_fig.transform(ax.transData.transform(Q[nd]))
    d = np.array([fx, fy]) - nxy
    edge = np.array([fx, fy]) - d / np.abs(d / np.array([w / 2, h / 2])).max()
    fig.add_artist(plt.Line2D([nxy[0], edge[0]], [nxy[1], edge[1]],
                              color=fs.MUTED, lw=1.6, ls=(0, (1.5, 1.5)),
                              zorder=0))
    ia = fig.add_axes([fx - w / 2, fy - h / 2, w, h])
    ia.imshow(maps[fam], cmap=fs.CMAP_SEQ.reversed(), vmin=0, vmax=vmax,
              interpolation="nearest")
    for sp in ia.spines.values():
        sp.set_visible(True); sp.set_color(fs.INK); sp.set_linewidth(1.0)
    ia.set_xticks([]); ia.set_yticks([])
    # fold names: 'loop' under the left inset; 'globule' / 'two domains'
    # right-aligned just left of the right insets, level with their top /
    # bottom edge (clear of the graph and of the colour key)
    if fam == 2:
        fig.text(fx, fy - h / 2 - 0.02, "loop", ha="center", va="top",
                 fontsize=13, color=fs.INK)
    elif fam == 0:
        fig.text(fx - w / 2 - 0.01, fy + h / 2, "globule", ha="right",
                 va="top", fontsize=13, color=fs.INK)
    else:
        fig.text(fx - w / 2 - 0.01, fy - h / 2, "two domains", ha="right",
                 va="bottom", fontsize=13, color=fs.INK)

# colour key
cax = fig.add_axes([0.33, 0.075, 0.34, 0.03])
grad = np.linspace(0, 1, 256)[None]
cax.imshow(grad, aspect="auto", cmap=fs.CMAP_DIV, norm=div,
           extent=(0, 1, 0, 1))
fs.clean(cax)
fig.text(0.315, 0.09, "condition A", ha="right", va="center", fontsize=13,
         color=fs.INK)
fig.text(0.685, 0.09, "condition B", ha="left", va="center", fontsize=13,
         color=fs.INK)

fs.save(fig, OUT + NAME + ".png")
from PIL import Image
im = Image.open(OUT + NAME + ".png")
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    OUT + NAME + "_small.png")
