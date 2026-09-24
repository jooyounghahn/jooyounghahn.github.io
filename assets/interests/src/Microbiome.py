"""Microbiome research-interest figure (synthetic toy example).

Synthetic gut time series (3 hosts x 40 days) with three planted teams of
taxa: team A collapses during an antibiotic course (carmine), team B blooms
during it, team C rises slowly and ignores the drug.  Generalist taxa are
mixtures of two teams.
Counts -> relative abundance -> Hellinger transform (square roots).
NMF (K = 3) of the taxa x samples table assigns each taxon to a team (argmax
of its row of W).
Ball Mapper on the taxa: each taxon's Hellinger profile over hosts and time is
scaled to length 1, greedy landmarks at radius eps cover the cloud, balls that
share a taxon are linked.
Left: raw counts of 12 taxa per team in one host, each scaled to its own
maximum, coloured by NMF team (thin) with team median (thick).  Counts are
shown rather than relative abundances because closure (the total grows as
team C rises) would make team A appear to slide before the antibiotics.
Right: Ball Mapper graph, node size ~ number of taxa, colour = majority
NMF team.  No real data are used.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys

import networkx as nx
import numpy as np
from sklearn.decomposition import NMF

SP = _SRCDIR
sys.path.insert(0, SP)
import figstyle as fs  # noqa: E402

rng = np.random.default_rng(3)

# ---------------------------------------------------------------- synthetic data
n_host, n_day = 3, 40
days = np.arange(n_day)
ab_on, ab_off = 16, 23                      # antibiotic course (days)
host_shift = (0.0, 1.5, -1.0)               # hosts respond slightly out of step


def smooth_step(x, x0, w=1.2):
    return 1.0 / (1.0 + np.exp(-(x - x0) / w))


def taxon_curve(k, d, lag):
    """Response of a taxon of team k at days d, with its own time lag."""
    if k == 0:   # A: collapses during antibiotics, slow partial recovery
        return (1.0 - 0.92 * smooth_step(d, ab_on + 2.4 + lag, 0.8)
                + 0.45 * smooth_step(d, ab_off + 6 + lag, 2.5))
    if k == 1:   # B: opportunists, bloom during the course, fade afterwards
        return 0.08 + 0.92 * smooth_step(d, ab_on + 2.4 + lag, 0.8) * (
            1 - smooth_step(d, ab_off + 3.8 + lag, 1.8))
    # C: slow steady rise with a weekly wobble, independent of the drug
    return (0.16 + 1.0 * np.clip(d / n_day, 0, None) ** 1.5
            + 0.08 * np.sin(2 * np.pi * (d + lag) / 7.0))


sizes = [150, 120, 135]
n_mix = 100
taxa, lt = [], []                           # (list of (team, weight), lag)
for k, m in enumerate(sizes):
    for _ in range(m):
        taxa.append(([(k, 1.0)], rng.uniform(-1.4, 1.4)))
        lt.append(k)
for q in range(n_mix):                      # generalists: A+C or B+C mixtures
    i = (0, 1)[q % 2]
    lam = rng.uniform(0.0, 1.0) ** 2         # weight of team A or B
    taxa.append(([(i, lam), (2, 1 - lam)], rng.uniform(-1.4, 1.4)))
    lt.append(-1)
lt = np.array(lt)                           # -1 = generalist
n_taxa = len(lt)
base = rng.lognormal(0.0, 0.5, n_taxa)      # taxon abundance scale

counts = np.zeros((n_taxa, n_host * n_day))
for h in range(n_host):
    d = days + host_shift[h]
    sig = np.array([sum(w * taxon_curve(k, d, lag) for k, w in comp)
                    for comp, lag in taxa])
    lam_ = base[:, None] * sig * np.exp(0.03 * rng.standard_normal(sig.shape))
    counts[:, h * n_day:(h + 1) * n_day] = rng.poisson(20000 * lam_ + 0.2)

rel = counts / counts.sum(axis=0, keepdims=True)   # relative abundance
hel = np.sqrt(rel)                                 # Hellinger transform

# ---------------------------------------------------------------- NMF teams
W = NMF(n_components=3, init="nndsvda", max_iter=3000,
        random_state=0).fit_transform(hel)
share = W / (W.sum(axis=1, keepdims=True) + 1e-12)
team = np.argmax(share, axis=1)
perm = [np.bincount(team[lt == k], minlength=3).argmax() for k in range(3)]
inv = {p: k for k, p in enumerate(perm)}
team = np.array([inv[c] for c in team])            # 0,1,2 = A,B,C

# ---------------------------------------------------------------- Ball Mapper
prof = hel / np.linalg.norm(hel, axis=1, keepdims=True)
D = np.linalg.norm(prof[:, None, :] - prof[None, :, :], axis=2)
eps = 0.075
landmarks = []
for p in rng.permutation(n_taxa):
    if not landmarks or D[p, landmarks].min() > eps:
        landmarks.append(p)
balls = [np.where(D[l] <= eps)[0] for l in landmarks]
G = nx.Graph()
for i, b in enumerate(balls):
    G.add_node(i, size=len(b), team=np.bincount(team[b], minlength=3).argmax())
for i in range(len(balls)):
    for j in range(i + 1, len(balls)):
        if np.intersect1d(balls[i], balls[j]).size:
            G.add_edge(i, j)

# layout: main component by Kamada-Kawai (unit median edge), rotated so that
# its long axis is vertical with team A on top; smaller components stacked
# beside its lower end
def layout(G, team_of):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = G.subgraph(comps[0])
    p = nx.kamada_kawai_layout(main)
    el = np.median([np.linalg.norm(p[u] - p[v]) for u, v in main.edges])
    nodes = list(p)
    xy = np.array([p[n] for n in nodes]) / el
    xy -= xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    xy = xy @ vt.T[:, ::-1]                      # long axis -> y
    a_nodes = [i for i, n in enumerate(nodes) if team_of[n] == 0]
    if a_nodes and xy[a_nodes, 1].mean() < 0:
        xy[:, 1] *= -1
    pos = {n: xy[i] for i, n in enumerate(nodes)}
    x_right = xy[:, 0].max() + 0.6
    y = xy[:, 1].min()
    for c in comps[1:]:
        H = G.subgraph(c)
        if len(c) >= 3:
            q = nx.kamada_kawai_layout(H)
            e2 = np.median([np.linalg.norm(q[u] - q[v]) for u, v in H.edges])
            q = {n: v / e2 for n, v in q.items()}
        else:
            q = {n: np.array([0.0, float(i)]) for i, n in enumerate(c)}
        qq = np.array(list(q.values()))
        lo, hi = qq.min(axis=0), qq.max(axis=0)
        for n, v in q.items():
            pos[n] = v - lo + np.array([x_right, y])
        y += hi[1] - lo[1] + 1.0
    return pos, comps


pos, comps = layout(G, {n: G.nodes[n]["team"] for n in G.nodes})

# ---------------------------------------------------------------- figure
TEAM_COL = [fs.CARMINE, fs.STEEL, fs.SAND]
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)

ax = fig.add_axes([0.05, 0.16, 0.42, 0.68])
c0 = counts[:, :n_day]                             # raw counts, host 0
r0 = c0 / c0.max(axis=1, keepdims=True)
ax.axvspan(ab_on, ab_off, color=fs.PALE, lw=0, zorder=0)
for k in (2, 1, 0):
    idx = np.where((team == k) & (lt == k))[0]
    for i in rng.choice(idx, 12, replace=False):
        ax.plot(days, r0[i], color=TEAM_COL[k], lw=0.9, alpha=0.4, zorder=1)
    ax.plot(days, np.median(r0[idx], axis=0), color=TEAM_COL[k], lw=3.0, zorder=3)
ax.set_xlim(0, n_day - 1)
ax.set_ylim(0, 1.05)
fs.clean(ax)
for side in ("bottom", "left"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.4)
ax.text(0.5 * (ab_on + ab_off), 1.08, "antibiotics", fontsize=13,
        color=fs.INK, ha="center", va="bottom")
ax.text(0.5, -0.04, "time", transform=ax.transAxes, fontsize=13,
        color=fs.INK, ha="center", va="top")

axg = fig.add_axes([0.53, 0.2, 0.44, 0.74])
for u, v in G.edges:
    axg.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
             color=fs.MUTED, lw=2.0, zorder=1)
nodes = list(G.nodes)
xy = np.array([pos[n] for n in nodes])
sz = np.array([G.nodes[n]["size"] for n in nodes])
col = [TEAM_COL[G.nodes[n]["team"]] for n in nodes]
axg.scatter(xy[:, 0], xy[:, 1], s=16 + 14 * np.sqrt(sz), c=col,
            edgecolors="white", linewidths=0.7, zorder=2)
axg.set_aspect("equal")
fs.clean(axg)
axg.margins(0.1)
axg.text(0.5, -0.02, "Ball Mapper\ngraph of taxa", transform=axg.transAxes,
         fontsize=13, color=fs.INK, ha="center", va="top")

out = _FIGDIR + "/Microbiome.png"
fs.save(fig, out)

from PIL import Image  # noqa: E402

im = Image.open(out)
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    _FIGDIR + "/Microbiome_small.png")
ok = lt >= 0
print(im.size, "nodes", G.number_of_nodes(), "edges", G.number_of_edges(),
      "components", len(comps), [len(c) for c in comps],
      "NMF agreement on planted teams", round(np.mean(team[ok] == lt[ok]), 3),
      "node teams", np.bincount([G.nodes[n]['team'] for n in nodes], minlength=3))
