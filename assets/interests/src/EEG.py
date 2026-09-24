"""EEG research-interest figure (synthetic toy example).

A synthetic 10 Hz alpha-like rhythm with background noise is turned into a
point cloud by Takens' delay embedding (x(t), x(t + tau)).  The rhythm makes
the cloud a loop.  Persistent homology (Vietoris-Rips, ripser) of the cloud
gives the H1 barcode: one long bar (the loop, carmine) and many short noise bars.
No real EEG data are used.
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch
from ripser import ripser

SP = _SRCDIR
sys.path.insert(0, SP)
import figstyle as fs  # noqa: E402

rng = np.random.default_rng(7)

# ---- synthetic signal: 10 Hz rhythm, slow amplitude drift, pink-ish noise
fs_hz = 250.0
T = 3.0
t = np.arange(0, T, 1 / fs_hz)
amp = 1.0 + 0.18 * np.sin(2 * np.pi * 0.4 * t + 0.5)
phase = 2 * np.pi * 10.0 * t + 0.35 * np.sin(2 * np.pi * 0.7 * t)
white = rng.standard_normal(t.size)
# 1/f-like noise by filtering white noise in the Fourier domain
F = np.fft.rfft(white)
f = np.fft.rfftfreq(t.size, 1 / fs_hz)
F[1:] /= np.sqrt(f[1:])
F[0] = 0
noise = np.fft.irfft(F, n=t.size)
noise *= 0.16 / noise.std()
x = amp * np.sin(phase) + noise

# ---- delay embedding, tau = quarter period (25 ms)
tau = int(round(fs_hz / 10.0 / 4))
X = np.column_stack([x[:-tau], x[tau:]])

# subsample for persistent homology (one point every 5 samples)
Xs = X[::5]
dgm1 = ripser(Xs, maxdim=1)["dgms"][1]
life = dgm1[:, 1] - dgm1[:, 0]
order = np.argsort(life)[::-1]
dgm1 = dgm1[order]

# ---- figure
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)

# (a) signal strip
ax_s = fig.add_axes([0.05, 0.75, 0.53, 0.17])
seg = t < 0.6
ax_s.plot(t[seg], x[seg], color=fs.STEEL, lw=1.7)
fs.clean(ax_s)
ax_s.set_xlim(0, 0.6)
ax_s.text(-0.01, 1.0, "x(t)", transform=ax_s.transAxes, fontsize=13,
          color=fs.INK, ha="left", va="bottom")

# (b) delay embedding
ax_e = fig.add_axes([0.02, 0.1, 0.58, 0.6])
ax_e.plot(X[:, 0], X[:, 1], color=fs.SKY, lw=0.8, alpha=0.5, zorder=1)
ax_e.scatter(Xs[:, 0], Xs[:, 1], s=16, color=fs.STEEL, lw=0, zorder=2)
lim = 1.66
ax_e.set_xlim(-lim, lim)
ax_e.set_ylim(-lim, lim)
ax_e.set_aspect("equal")
fs.clean(ax_e)
ax_e.axhline(0, color=fs.PALE, lw=1.0, zorder=0)
ax_e.axvline(0, color=fs.PALE, lw=1.0, zorder=0)
ax_e.text(0.5, -0.03, "delay embedding", transform=ax_e.transAxes,
          fontsize=13, color=fs.INK, ha="center", va="top")

# arrow from signal to embedding
arr = FancyArrowPatch((0.31, 0.74), (0.31, 0.685), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=12, lw=1.6, color=fs.MUTED)
fig.patches.append(arr)

# (c) H1 barcode
ax_b = fig.add_axes([0.68, 0.16, 0.29, 0.6])
nbar = min(10, len(dgm1))
for i in range(nbar):
    b, d = dgm1[i]
    c = fs.CARMINE if i == 0 else fs.MUTED
    lw = 8.0 if i == 0 else 5.0
    ax_b.plot([b, d], [nbar - i, nbar - i], color=c, lw=lw,
              solid_capstyle="butt")
ax_b.set_ylim(0, nbar + 1.5)
ax_b.set_xlim(0, dgm1[0, 1] * 1.05)
fs.clean(ax_b)
ax_b.spines["bottom"].set_visible(True)
ax_b.spines["bottom"].set_linewidth(1.2)
ax_b.text(0.5 * (dgm1[0, 0] + dgm1[0, 1]), nbar + 1.0, "persistent\nloop",
          fontsize=13, color=fs.CARMINE, ha="center", va="bottom")

# arrow from embedding to barcode
arr2 = FancyArrowPatch((0.585, 0.39), (0.665, 0.39), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=12, lw=1.6, color=fs.MUTED)
fig.patches.append(arr2)

out = _FIGDIR + "/EEG.png"
fs.save(fig, out)

from PIL import Image  # noqa: E402

im = Image.open(out)
im.resize((320, int(320 * im.height / im.width)), Image.LANCZOS).save(
    _FIGDIR + "/EEG_small.png")
print(im.size, "bars:", len(dgm1), "top lifetimes:", np.round(life[order][:4], 3))
