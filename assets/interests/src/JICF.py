"""JICF research-interest figure (synthetic toy example).

A 2-D caricature of a premixed jet injected into a hot vitiated crossflow.
The jet centreline bends downstream, y_c(x) = 1.6 (x - x0)^(1/3); across it the
jet mixture fraction Z is a Gaussian that widens with the distance s along the
centreline.  Temperature rises where jet and crossflow mix and burn (more on
the leeward side), fluid age grows with s, vorticity peaks in the two shear
layers.  NO is a thermal-NO caricature: residence time times exp(-Ta/T).
Every cell gets four features (T, Z, age, |omega|), each mapped to [0, 1] by
one quantile transform, as in the real feature matrices, so that bimodal
fields do not collapse into one giant ball.
Ball Mapper on a random sample of cells: greedy landmarks at radius eps, balls
that share a cell are linked; node colour = rank of the mean NO of its cells.
Left: temperature field with the jet centreline and the high-NO cells outlined
(carmine).  Right: Ball Mapper graph; nodes whose mean NO is in the top decile
of nodes are ringed in carmine, the same cells as the outline on the left.
No real data are used.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

SP = _SRCDIR
sys.path.insert(0, SP)
import figstyle as fs  # noqa: E402

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- synthetic field
nx_, ny_ = 360, 180
x = np.linspace(0, 10, nx_)
y = np.linspace(0, 5, ny_)
X, Y = np.meshgrid(x, y)
x0 = 1.2                                   # jet exit

# centreline and local frame: nearest centreline point for every cell
xs = np.linspace(x0, 10.5, 1200)
yc = 1.6 * (xs - x0) ** (1 / 3)
ds = np.hypot(np.diff(xs), np.diff(yc))
sc = np.r_[0, np.cumsum(ds)]               # arc length
tree = cKDTree(np.c_[xs, yc])
dist, k = tree.query(np.c_[X.ravel(), Y.ravel()])
s = sc[k].reshape(X.shape)
tx, ty = np.gradient(xs)[k], np.gradient(yc)[k]
side = np.sign((X.ravel() - xs[k]) * ty - (Y.ravel() - yc[k]) * tx).reshape(X.shape)
n = dist.reshape(X.shape) * side           # + leeward (below/behind), - windward

w = 0.25 + 0.07 * s                        # jet widens downstream
Z = np.exp(-(n / w) ** 2) * np.exp(-0.05 * s)
Z[(X < x0 - 0.3)] *= 0.0                   # nothing upstream of the exit
prog = 1 - np.exp(-s / 1.8)                # reaction progress along the jet
lee = 1 + 0.9 * (n > 0)                    # leeward side burns more
T_cf, T_jet, dT = 0.55, 0.12, 0.9          # normalized temperatures
T0 = T_cf * (1 - Z) + T_jet * Z + dT * 4 * Z * (1 - Z) * prog * lee / 1.9
T = T0 + 0.015 * rng.standard_normal(T0.shape)
age = (s / s.max()) * np.tanh(Z / 0.1) + 0.3 * X / 10
vort = np.exp(-((np.abs(n) - w) / (0.35 * w)) ** 2) * np.exp(-0.15 * s) * (X > x0 - 0.3)
NO = age * np.exp(-6.0 * (1 / np.clip(T0, 0.2, None) - 1 / 1.3))

# ---------------------------------------------------------------- features, quantile map
def quantile01(v):
    r = np.argsort(np.argsort(v))
    return r / (len(v) - 1)


pick = rng.choice(X.size, 3500, replace=False)
F = np.c_[T.ravel()[pick], Z.ravel()[pick], age.ravel()[pick], vort.ravel()[pick]]
F = np.column_stack([quantile01(c) for c in F.T])
no = NO.ravel()[pick]

# ---------------------------------------------------------------- Ball Mapper
eps = 0.16
ft = cKDTree(F)
landmarks, covered = [], np.zeros(len(F), bool)
for p in rng.permutation(len(F)):
    if not covered[p]:
        landmarks.append(p)
        covered[ft.query_ball_point(F[p], eps)] = True
balls = [np.array(ft.query_ball_point(F[l], eps)) for l in landmarks]
G = nx.Graph()
for i, b in enumerate(balls):
    G.add_node(i, size=len(b), no=no[b].mean())
member = [set(b) for b in balls]
for i in range(len(balls)):
    for j in range(i + 1, len(balls)):
        if member[i] & member[j]:
            G.add_edge(i, j)
main = G.subgraph(max(nx.connected_components(G), key=len)).copy()
pos = nx.kamada_kawai_layout(main)
nodes = list(main.nodes)
nno = np.array([main.nodes[q]["no"] for q in nodes])
hi_thr = np.quantile(nno, 0.9)
hot = nno >= hi_thr

# ---------------------------------------------------------------- figure
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)

ax = fig.add_axes([0.03, 0.34, 0.52, 0.52])
ax.imshow(T, origin="lower", extent=(0, 10, 0, 5), cmap=fs.CMAP_SEQ, vmin=0.1, vmax=1.35,
          aspect="auto", interpolation="bilinear")
ax.plot(xs[xs <= 10], yc[xs <= 10], color=fs.WHITE, lw=1.3, ls=(0, (4, 3)))
ax.contour(X, Y, NO, levels=[np.quantile(no, 0.9)], colors=[fs.CARMINE], linewidths=1.8)
ax.plot([0, 10], [0, 0], color=fs.INK, lw=2.0)
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
fs.clean(ax)
ax.annotate("", xy=(1.0, 4.2), xytext=(0.15, 4.2),
            arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=1.4))
ax.text(0.15, 4.45, "hot crossflow", fontsize=11, color=fs.INK, ha="left", va="bottom")
ax.text(x0, -0.25, "jet", fontsize=12, color=fs.INK, ha="center", va="top")
ax.text(5.0, -0.95, "temperature, high-NO cells", fontsize=12, color=fs.INK,
        ha="center", va="top")

axg = fig.add_axes([0.58, 0.22, 0.40, 0.72])
for u, v in main.edges:
    axg.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=fs.MUTED, lw=1.0, zorder=1)
xy = np.array([pos[q] for q in nodes])
sz = np.array([main.nodes[q]["size"] for q in nodes])
axg.scatter(xy[:, 0], xy[:, 1], s=6 + 5 * np.sqrt(sz), c=quantile01(nno), cmap=fs.CMAP_SEQ,
            vmin=0, vmax=1, edgecolors="white", linewidths=0.4, zorder=2)
axg.scatter(xy[hot, 0], xy[hot, 1], s=6 + 5 * np.sqrt(sz[hot]), facecolors="none",
            edgecolors=fs.CARMINE, linewidths=1.4, zorder=3)
axg.set_aspect("equal")
fs.clean(axg)
axg.margins(0.06)
axg.text(0.5, -0.02, "Ball Mapper graph\ncolored by NO", transform=axg.transAxes,
         fontsize=13, color=fs.INK, ha="center", va="top")

out = _FIGDIR + "/JICF.png"
fs.save(fig, out)
print("nodes", main.number_of_nodes(), "of", G.number_of_nodes(), "edges", main.number_of_edges(),
      "hot nodes", int(hot.sum()))
