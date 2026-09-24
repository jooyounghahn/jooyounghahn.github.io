"""SERS: is the classifier seeing the disease or the patient?

Synthetic SERS spectra: 16 people (8 ill, 8 healthy), 30 repeated spectra each.
Every person has an own spectral signature (random band intensities and small
band shifts); the disease raises two bands by 35 %; every repeat
gets a hotspot scale factor, a smooth fluorescence background and noise.
Pipeline: asymmetric-least-squares baseline, vector normalisation, PCA (15
components), logistic regression.

Left : PC1-PC2 scores; the repeats of one person form a tight cluster, the
       two diagnoses are mixed.
Right: cross-validated accuracy (5 folds x 10 repetitions) when spectra are
       split at random (repeats of a person in train and test) versus split
       by person (all repeats of a person on the same side).
"""
import os as _os
_SRCDIR = _os.path.dirname(_os.path.abspath(__file__))   # this folder (figstyle.py lives here)
_FIGDIR = _os.path.dirname(_SRCDIR)                          # assets/interests/: the PNG is written here
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # small problem: one thread is fastest

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

SCR = (_SRCDIR)
sys.path.insert(0, SCR)
import figstyle as fs  # noqa: E402

OUT = _FIGDIR + "/SERS"
rng = np.random.default_rng(3)

# ------------------------------------------------------------ synthetic data
wn = np.linspace(400, 1800, 400)                          # Raman shift, cm^-1
centres = np.array([520, 640, 730, 850, 1003, 1090, 1130, 1240, 1330, 1450, 1580, 1655])
widths = np.array([14, 12, 10, 16, 8, 14, 12, 20, 16, 14, 12, 18])
base_amp = np.array([.5, .4, 1.0, .5, .9, .6, .4, .5, .7, .6, .5, .4])
disease_bands = [3, 8]                                     # 850 and 1330 cm^-1
disease_gain = 0.35                                        # +35 % in ill people
N_PER, N_REP = 8, 30


def spectrum(amp, shift):
    return sum(a * np.exp(-0.5 * ((wn - c - s) / w) ** 2)
               for a, c, s, w in zip(amp, centres, shift, widths))


X, y, g = [], [], []
for person in range(2 * N_PER):
    ill = person < N_PER
    amp = base_amp * np.exp(rng.normal(0, 0.25, len(base_amp)))   # own signature
    shift = rng.normal(0, 2.0, len(centres))
    if ill:
        amp[disease_bands] *= 1 + disease_gain
    for _ in range(N_REP):
        a = amp * np.exp(rng.normal(0, 0.03, len(amp)))             # repeat jitter
        hot = np.exp(rng.normal(0, 0.6))                            # hotspot scale
        bg = np.polyval(rng.normal([0, 0.4, 0.6], [0.1, 0.1, 0.2]), (wn - 400) / 1400)
        s = hot * spectrum(a, shift) + 3 * bg + rng.normal(0, 0.02, wn.size)
        X.append(s); y.append(int(ill)); g.append(person)
X, y, g = np.array(X), np.array(y), np.array(g)


# ------------------------------------------------------------ preprocessing
def als_baseline(s, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens)."""
    L = s.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D @ D.T
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w)
        z = spsolve((W + H).tocsc(), w * s)
        w = p * (s > z) + (1 - p) * (s < z)
    return z


Xc = np.array([s - als_baseline(s) for s in X])
Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)

model = make_pipeline(PCA(n_components=15), LogisticRegression(max_iter=2000))


def cv_acc(split, reps=10):
    acc = []
    for r in range(reps):
        if split == "random":
            cv = StratifiedKFold(5, shuffle=True, random_state=r).split(Xn, y)
        else:
            perm = np.random.default_rng(r).permutation(2 * N_PER)
            cv = GroupKFold(5).split(Xn, y, groups=perm[g])
        for tr, te in cv:
            model.fit(Xn[tr], y[tr])
            acc.append((model.predict(Xn[te]) == y[te]).mean())
    return np.array(acc)


acc_rand = cv_acc("random")
acc_grp = cv_acc("person")
print(f"random split    : {acc_rand.mean():.3f} +- {acc_rand.std():.3f}")
print(f"split by person : {acc_grp.mean():.3f} +- {acc_grp.std():.3f}")

Z = PCA(n_components=2).fit_transform(Xn)

# ------------------------------------------------------------------ figure
fs.setup()
fig = fs.plt.figure(figsize=(fs.W_IN, fs.H_IN), dpi=fs.DPI)
axL = fig.add_axes([0.02, 0.06, 0.52, 0.88])
axR = fig.add_axes([0.655, 0.2, 0.335, 0.72])

col = {0: fs.SAND, 1: fs.STEEL}
order = rng.permutation(len(y))
axL.scatter(Z[order, 0], Z[order, 1], s=16, c=[col[v] for v in y[order]],
            edgecolors=fs.WHITE, linewidths=0.3, alpha=0.95)
fs.clean(axL)
axL.set_aspect("equal", adjustable="datalim")
axL.text(0.52, 0.97, "ill", transform=axL.transAxes, ha="left", va="top",
         fontsize=13, color=fs.STEEL)
axL.text(0.52, 0.89, "healthy", transform=axL.transAxes, ha="left", va="top",
         fontsize=13, color="#A8843C")          # darker sand: SAND is too pale for text

rj = np.random.default_rng(0)
for x0, acc, c in ((0, acc_rand, fs.CARMINE), (1, acc_grp, fs.INK)):
    axR.scatter(x0 + rj.uniform(-0.17, 0.17, acc.size), acc, s=14, color=c,
                alpha=0.55, edgecolors="none")
    axR.plot([x0 - 0.3, x0 + 0.3], [acc.mean()] * 2, color=c, lw=3.2,
             solid_capstyle="butt")
axR.axhline(0.5, color=fs.MUTED, lw=1.6, ls=(0, (3, 2)))
axR.set_xlim(-0.6, 1.6)
axR.set_ylim(0.2, 1.04)
axR.set_yticks([0.5, 1.0])
axR.set_yticklabels(["0.5", "1"], fontsize=13)
axR.set_xticks([0, 1])
axR.set_xticklabels(["random\nsplit", "by\nperson"], fontsize=13,
                    linespacing=1.0)
axR.tick_params(axis="x", length=0, pad=6, colors=fs.INK)
axR.tick_params(axis="y", length=3, colors=fs.INK)
axR.spines["bottom"].set_visible(False)
axR.set_ylabel("accuracy", fontsize=13, labelpad=2)

fs.save(fig, OUT + ".png")

from PIL import Image  # noqa: E402
im = Image.open(OUT + ".png")
im.resize((320, round(320 * im.height / im.width)), Image.LANCZOS).save(OUT + "_small.png")
print(im.size)
