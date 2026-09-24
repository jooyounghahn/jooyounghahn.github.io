"""MultiChromat: every mass spectrum read as a word, all spectra in one trie.

Synthetic data: 8 made-up compounds in 3 chemical families (a family shares its
most intense fragment mass).  Each compound gives many noisy spectra (random
intensities, 1 ppm mass error, a few small random peaks).  A spectrum is read
as a word: its fragment masses sorted from the most to the least intense; the
first three letters are used.  The tolerant trie groups the k-th letters of the
spectra in one node by a greedy anchored sweep with a 10 ppm window.

Top   : one spectrum; its three most intense peaks are its first three letters.
Bottom: the trie over all spectra, root at the top, level k = k-th letter;
        the carmine sticks at the left (nominal m/z) repeat the example's peaks
        of rank 1, 2, 3 beside the level each one selects: the word 91-105-77.
        Edge width grows with the number of spectra; thin side twigs come from
        spectra whose two peaks of almost equal height swapped rank.  Carmine:
        the path of the spectrum shown on top.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys

import numpy as np

SCR = (_SRCDIR)
sys.path.insert(0, SCR)
import figstyle as fs  # noqa: E402

OUT = _FIGDIR + "/MultiChromat"
rng = np.random.default_rng(11)

# --------------------------------------------------------------- compounds
# {fragment m/z: relative intensity}; masses are illustrative, not assigned
COMPOUNDS = [
    # family A (base peak 43.0178)
    ({43.0178: 100, 61.0284: 45, 70.0419: 28, 88.0524: 14, 45.0335: 6}, 70),
    ({43.0178: 100, 61.0284: 45, 73.0284: 25, 101.0597: 12, 55.0178: 5}, 50),
    ({43.0178: 100, 58.0413: 55, 71.0491: 22, 86.0726: 10}, 60),
    # family B (base peak 57.0699)
    ({57.0699: 100, 41.0386: 60, 43.0542: 50, 71.0855: 25, 85.1012: 12}, 80),
    ({57.0699: 100, 56.0621: 45, 41.0386: 30, 70.0777: 16}, 45),
    # family C (base peak 91.0542)
    ({91.0542: 100, 92.0621: 45, 106.0777: 28, 65.0386: 12}, 55),
    ({91.0542: 100, 105.0699: 70, 120.0934: 32, 77.0386: 15}, 65),
    ({91.0542: 100, 105.0699: 70, 77.0386: 35, 51.0229: 14}, 60),
]
DEPTH, PPM = 3, 10.0


def make_spectrum(peaks):
    mz = np.array(list(peaks.keys()))
    it = np.array(list(peaks.values()), float)
    it = it * np.exp(rng.normal(0, 0.10, it.size))           # intensity noise
    mz = mz * (1 + rng.normal(0, 1e-6, mz.size))              # 1 ppm mass error
    k = rng.integers(4, 9)                                     # small random peaks
    mz = np.concatenate([mz, rng.uniform(40, 125, k)])
    it = np.concatenate([it, rng.uniform(2, 10, k)])
    return mz, it


spectra, owner = [], []
for c, (peaks, n) in enumerate(COMPOUNDS):
    for _ in range(n):
        spectra.append(make_spectrum(peaks)); owner.append(c)
words = np.array([mz[np.argsort(-it)][:DEPTH] for mz, it in spectra])


# ------------------------------------------------------------ tolerant trie
def sweep(masses):
    """Greedy anchored sweep: sort, anchor, take all within PPM of the anchor."""
    order = np.argsort(masses)
    groups, cur, anchor = [], [], None
    for i in order:
        if anchor is None or (masses[i] - anchor) / anchor * 1e6 > PPM:
            if cur:
                groups.append(cur)
            cur, anchor = [i], masses[i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


class Node:
    def __init__(self, members, level, letter=None):
        self.members, self.level, self.letter = members, level, letter
        self.children = []


def build(node):
    if node.level == DEPTH:
        return
    m = words[node.members, node.level]
    for grp in sweep(m):
        ids = node.members[grp]
        child = Node(ids, node.level + 1, float(np.mean(m[grp])))
        node.children.append(child)
    node.children.sort(key=lambda ch: -len(ch.members))
    for ch in node.children:
        build(ch)


root = Node(np.arange(len(words)), 0)
build(root)


def walk(node):
    yield node
    for ch in node.children:
        yield from walk(ch)


nodes = list(walk(root))
leaves = [n for n in nodes if not n.children]
print(f"{len(words)} spectra, {len(nodes) - 1} nodes, {len(leaves)} leaves")
for lev in range(1, DEPTH + 1):
    print("level", lev, "nodes", sum(n.level == lev for n in nodes))

# ------------------------------------------------------------------ layout
# leaves equally spaced in depth-first order (big branches first); parents at
# the mean of their children


def order_children(node):
    # put the heaviest child in the middle so the tree looks balanced
    ch = sorted(node.children, key=lambda c: -len(c.members))
    out = []
    for i, c in enumerate(ch):
        if i % 2:
            out.append(c)
        else:
            out.insert(0, c)
    node.children = out
    for c in node.children:
        order_children(c)


order_children(root)
x = {}
counter = [0.0]


def place(node):
    if not node.children:
        x[id(node)] = counter[0]
        counter[0] += 1.0
    else:
        for c in node.children:
            place(c)
        x[id(node)] = np.mean([x[id(c)] for c in node.children])


place(root)

# the example spectrum: first spectrum of compound 7 (family C)
ex = owner.index(7)
ex_path = [n for n in nodes if ex in n.members]

# ------------------------------------------------------------------ figure
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
axS = fig.add_axes([0.04, 0.68, 0.92, 0.28])
axT = fig.add_axes([0.04, 0.03, 0.92, 0.60])

# spectrum
mz, it = spectra[ex]
it = it / it.max()
top = np.argsort(-it)[:DEPTH]
axS.vlines(mz, 0, it, color=fs.MUTED, lw=2.0)
axS.vlines(mz[top], 0, it[top], color=fs.CARMINE, lw=2.8)
axS.axhline(0, color=fs.MUTED, lw=1.2)
axS.set_xlim(38, 127)
axS.set_ylim(0, 1.25)
fs.clean(axS)

# trie
tot = len(words)
Y = {lev: DEPTH - lev for lev in range(DEPTH + 1)}


def edge(ax, p, c, color, lw, z):
    x0, y0, x1, y1 = x[id(p)], Y[p.level], x[id(c)], Y[c.level]
    t = np.linspace(0, 1, 40)
    s = 3 * t ** 2 - 2 * t ** 3                                   # smooth step
    ax.plot(x0 + (x1 - x0) * s, y0 + (y1 - y0) * t, color=color, lw=lw,
            solid_capstyle="round", zorder=z)


for n in nodes:
    for c in n.children:
        w = 1.8 + 12.0 * (len(c.members) / tot) ** 0.8
        edge(axT, n, c, fs.SKY, w, 1)
for a, b in zip(ex_path[:-1], ex_path[1:]):
    edge(axT, a, b, fs.CARMINE, 2.6, 3)
for n in nodes:
    ms = 3.5 + 10.0 * (len(n.members) / tot) ** 0.5
    on = n in ex_path and n is not root
    axT.plot(x[id(n)], Y[n.level], "o", ms=ms, zorder=4,
             color=fs.CARMINE if on else fs.STEEL, mec=fs.WHITE, mew=0.9)
# left margin: at each level, a stick as tall as the example's peak of that
# rank (the tallest peak picks the level-1 branch, the next one level 2, ...)
GX = -0.75
for k, n in zip(top, ex_path[1:]):
    y0 = Y[n.level] - 0.22
    axT.plot([GX, GX], [y0, y0 + 0.6 * it[k]], color=fs.CARMINE, lw=2.8,
             solid_capstyle="butt", zorder=5)
    axT.plot([GX - 0.16, GX + 0.16], [y0, y0], color=fs.MUTED, lw=1.2, zorder=5)
    axT.text(GX - 0.3, y0, f"{mz[k]:.0f}", ha="right", va="bottom", fontsize=13,
             color=fs.CARMINE)
axT.set_xlim(-2.1, counter[0] - 0.4)
axT.set_ylim(-0.3, DEPTH + 0.25)
fs.clean(axT)

fs.save(fig, OUT + ".png")

from PIL import Image  # noqa: E402
im = Image.open(OUT + ".png")
im.resize((320, round(320 * im.height / im.width)), Image.LANCZOS).save(OUT + "_small.png")
print(im.size)
