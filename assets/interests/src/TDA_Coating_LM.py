"""TDA_Coating_LM: superlevel-set persistence (prominence) of a synthetic coating edge profile.

A synthetic cross-section of a coated film near its edge: flat plateau, two small
bumps, a dip, an edge ridge, then the drop to the foil, plus small measurement noise.
0-dimensional superlevel-set persistence is computed by sorting and union-find
(checked against GUDHI).  Every peak gets a vertical bar from its height down to the
level at which it merges with a higher peak (its prominence); a dashed line goes to
that col.  Long bars = bumps and ridge; short bars = noise.  All data are synthetic.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
import gudhi

sys.path.insert(0, _SRCDIR)
import figstyle as fs
import matplotlib.pyplot as plt

OUT = _FIGDIR + "/TDA_Coating_LM"
rng = np.random.default_rng(11)

# ---- synthetic edge profile (film height / plateau height) --------------------
N = 700
x = np.linspace(0.0, 1.0, N)
G = lambda c, w: np.exp(-0.5 * ((x - c) / w) ** 2)
h = 1.0 + 0.042 * G(0.50, 0.024) + 0.058 * G(0.62, 0.024) - 0.028 * G(0.725, 0.022) \
    + 0.125 * G(0.83, 0.024)
taper = 0.5 * (1.0 - np.tanh((x - 0.885) / 0.012))
noise = gaussian_filter1d(rng.normal(size=N), 2.0)
noise /= noise.std()
h = h * taper + 0.0028 * noise * taper


# ---- 0-dim superlevel persistence by sorting + union-find -----------------------
def superlevel_pairs(f):
    """Return (peak_index, birth, death, col_index) for every finite H0 class,
    plus the index of the global maximum (essential class)."""
    n = len(f)
    order = np.argsort(-f, kind="stable")
    parent = -np.ones(n, dtype=int)          # -1 = not yet added
    peak = np.arange(n)                      # highest vertex of each component

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for v in order:
        parent[v] = v
        for u in (v - 1, v + 1):
            if 0 <= u < n and parent[u] >= 0:
                ru, rv = find(u), find(v)
                if ru == rv:
                    continue
                # elder rule: the component with the lower peak dies at f[v]
                young, old = (ru, rv) if f[peak[ru]] < f[peak[rv]] else (rv, ru)
                if peak[young] != v:
                    pairs.append((peak[young], f[peak[young]], f[v], v))
                parent[young] = old
    return pairs, int(order[0])


pairs, top = superlevel_pairs(h)
pers = np.array([b - d for _, b, d, _ in pairs])

# check against GUDHI (superlevel of h = sublevel of -h)
cc = gudhi.CubicalComplex(top_dimensional_cells=-h)
g = np.array([bd for dim, bd in cc.persistence() if dim == 0 and np.isfinite(bd[1])])
g_pers = np.sort(g[:, 1] - g[:, 0])[::-1]
assert np.allclose(np.sort(pers)[::-1][:len(g_pers)], g_pers[:len(pers)], atol=1e-12)
print("pairs:", len(pairs), " matches GUDHI; largest prominences:", np.round(np.sort(pers)[::-1][:5], 4))

# ---- figure -----------------------------------------------------------------------
fs.setup()
fig = plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
y0, y1 = 0.955, 1.14
ax.fill_between(x, y0 - 1, h, color=fs.PALE, lw=0)
ax.plot(x, h, color=fs.INK, lw=1.8)

plateau = 1.0
for (i, b, d, j), p in zip(pairs, pers):
    if p < 0.003 or i < 8:          # skip the half-peak at the cut-off left boundary
        continue
    col, lw = (fs.STEEL, 3.2) if p > 0.02 else (fs.MUTED, 2.0)
    ax.plot([x[i], x[i]], [b, d], color=col, lw=lw, solid_capstyle="butt")
    if p > 0.02:
        ax.plot([x[i], x[j]], [d, d], color=col, lw=1.4, ls=(0, (2.5, 2)))
# the ridge is the global maximum: its bar goes down to the plateau
ax.plot([x[top], x[top]], [h[top], plateau], color=fs.CARMINE, lw=3.4, solid_capstyle="butt")
ax.plot([0.45, x[top]], [plateau, plateau], color=fs.MUTED, lw=1.0, ls=(0, (1, 2.5)))

ax.set_xlim(0.0, 1.0)
ax.set_ylim(y0, y1)
fs.clean(ax)
ax.text(x[top] + 0.035, h[top] - 0.004, "ridge", color=fs.CARMINE, fontsize=13,
        ha="left", va="center")
ax.text(0.56, 1.075, "bumps", color=fs.STEEL, fontsize=13, ha="center", va="bottom")
ax.text(0.21, 1.016, "noise", color=fs.MUTED, fontsize=13, ha="center", va="bottom")

fs.save(fig, OUT + ".png")

from PIL import Image
im = Image.open(OUT + ".png")
print(im.size)
im.resize((320, int(320 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_small.png")
