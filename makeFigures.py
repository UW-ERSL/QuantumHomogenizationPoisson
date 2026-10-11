"""makeFigures.py -- the two data figures of the paper, written to figs/ as PDF.

    python makeFigures.py              # about 2 minutes (N = 64 is dense)

figs/spectrum.pdf   (fig:spectrum)   eigenvalues of K^chi and of M, square inclusion, rho = 10
figs/qsvt_poly.pdf  (fig:qsvt-poly)  the odd QSVT polynomial q_t, rho = 10, eps_p = 1e-3

Every number the text quotes about a figure is ASSERTED here.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_square import operators, circulant

RHO = 10.0
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42})
os.makedirs("figs", exist_ok=True)

# ---- spectra ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)
prev = None
for N, ls, col in ((16, "-", "0.6"), (32, "--", "0.3"), (64, "-", "k")):
    A, St, K, Kc, chi = operators(N, RHO)
    Kih = circulant(lambda e: e ** -0.5, N)
    lk = np.linalg.eigvalsh(Kc); lk = lk[lk > 1e-9 * lk.max()]
    lm = np.linalg.eigvalsh(Kih @ Kc @ Kih); lm = lm[np.abs(lm) > 1e-9]
    assert abs(lm.min() * RHO - 1) < 1e-9 and abs(lm.max() - 1) < 1e-9      # "in [1/rho, 1]"
    inside = np.sum((lm > 1 / RHO + 1e-9) & (lm < 1 - 1e-9))
    assert inside == 2 * N - 1, (N, inside)                                 # "2N - 1 strictly between"
    r = lk.min() / lk.max()
    if prev is not None:
        assert 3.5 < prev / r < 4.2, (N, prev / r)                          # "about 4 per doubling"
    prev = r
    ax[0].loglog(np.arange(1, len(lk) + 1), lk / lk.max(), ls, color=col, lw=1.2, label=f"$N = {N}$")
    ax[1].loglog(np.arange(1, len(lm) + 1), lm, ls, color=col, lw=1.2)
    print(f"N = {N}: lmin/lmax(K^chi) = {r:.3e}; eigenvalues of M strictly inside: {inside}")
for a in ax:
    a.set_xlabel("eigenvalue index (ascending)"); a.set_xlim(1, 5000); a.set_ylim(1e-4, 2)
ax[0].legend(loc="lower right", frameon=False)
for v, s in ((1 / RHO, r"$1/\rho$"), (1.0, "$1$")):
    ax[1].axhline(v, color="0.5", lw=0.8, ls=":"); ax[1].text(1.3, v * 1.3, s, fontsize=8)
ax[0].set_ylabel("eigenvalue")
ax[0].set_title(r"(a) $\mathbf{K}^{\chi}/\lambda_{\max}(\mathbf{K}^{\chi})$", fontsize=9)
ax[1].set_title(r"(b) $\mathbf{M}$", fontsize=9)
fig.tight_layout(); fig.savefig("figs/spectrum.pdf")

# ---- the inverting polynomial ----------------------------------------------
def T(t, z):
    """Chebyshev polynomial T_t, valid inside and outside [-1, 1]."""
    z = np.asarray(z, float); out = np.empty_like(z); i = np.abs(z) <= 1
    out[i] = np.cos(t * np.arccos(z[i]))
    out[~i] = np.sign(z[~i]) ** t * np.cosh(t * np.arccosh(np.abs(z[~i])))
    return out


eps = 1e-3
a = 1 / RHO ** 2; s = (1 + a) / (1 - a)
t = int(np.ceil(np.arccosh(1 / eps) / np.arccosh(s)))
qt = lambda x: (1 - T(t, (1 + a - 2 * x ** 2) / (1 - a)) / np.cosh(t * np.arccosh(s))) / x
x = np.linspace(-1, 1, 400001); x = x[x != 0]
q = qt(x)
C, xr = np.abs(q).max(), abs(x[np.argmax(np.abs(q))])
xp = np.linspace(1 / RHO, 1, 20001)
res = np.abs(xp * qt(xp) - 1)
assert 2 * t - 1 == 75 and abs(C / RHO - 1.30) < 0.005 and abs(xr * RHO - 0.585) < 0.001
assert res.max() <= eps and abs(res.max() - 9.8e-4) < 0.05e-4                # "9.8e-4 <= eps_p"
print(f"q_t: d = {2 * t - 1}, C = {C / RHO:.3f} rho at |x| = {xr * RHO:.3f}/rho, residual {res.max():.3e}")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
ax[0].plot(x, q, "k", lw=1.0, label=r"$q_t(x)$")
xx = np.linspace(1 / RHO, 1, 200)
ax[0].plot(xx, 1 / xx, color="0.55", lw=2.5, ls=":", label=r"$1/x$")
ax[0].plot(-xx, -1 / xx, color="0.55", lw=2.5, ls=":")
for v in (-1 / RHO, 1 / RHO):
    ax[0].axvline(v, color="0.7", lw=0.6)
ax[0].set_xlim(-1, 1); ax[0].set_ylim(-1.6 * RHO, 1.6 * RHO); ax[0].set_xlabel("$x$")
ax[0].legend(loc="upper left", frameon=False)
ax[0].set_title(f"(a) $q_t$ of degree $d = {2 * t - 1}$", fontsize=9)
ax[1].semilogy(xp, res + 1e-18, "k", lw=0.8)
ax[1].axhline(eps, color="0.5", lw=0.8, ls="--")
ax[1].text(0.115, eps * 1.6, r"$\epsilon_p = 10^{-3}$", fontsize=8)
ax[1].set_xlim(1 / RHO, 1); ax[1].set_ylim(1e-7, 1e-2); ax[1].set_xlabel("$x$")
ax[1].set_title(r"(b) $|x\,q_t(x) - 1|$ on $[1/\rho, 1]$", fontsize=9)
fig.tight_layout(); fig.savefig("figs/qsvt_poly.pdf")
print("figs/spectrum.pdf, figs/qsvt_poly.pdf")
