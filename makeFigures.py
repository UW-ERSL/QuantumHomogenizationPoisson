"""makeFigures.py -- the figures of the paper, written to figs/ as PDF.

    python makeFigures.py              all four figures of the paper (about 10 minutes)
    python makeFigures.py mesh         just that one
    python makeFigures.py --list       names and destinations

figs/spectrum.pdf       (fig:spectrum, Sec. 2.2)   eigenvalues of K^chi and of M, square, rho = 10
figs/cells.pdf          (fig:cells, Sec. 4.1)      the three test cells at N = 16
figs/mesh.pdf           (fig:mesh, Sec. 4.2)       mesh refinement, N = 16 ... 1024, three cells
figs/isometry_cost.pdf  (fig:isometry-cost, App. A) delta 2^F / N for the 42 cases

Not in the paper, drawn only when named: `circuits` (the readout and QSVT
circuit figures of the companion paper, circuit_figures.py; needs Qiskit and
pyqsp), `qsvt_poly` (q_t at eps_p = 1e-3) and `angles` (omega and sin^2 tau
over the wavenumbers, N = 32).

Every number the text quotes about a figure is ASSERTED before the figure is
written. The test cells and the matrix-free operators are those of cells.py.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cells

RHO = 10.0
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42})
os.makedirs("figs", exist_ok=True)
STYLE = {"square": ("-", "k", "o"), "cross": ("--", "0.35", "s"), "laminate": (":", "0.55", "^")}


def draw_spectrum():
    """figs/spectrum.pdf (fig:spectrum): eigenvalues of K^chi and of M, square inclusion, rho = 10."""
    from verify_square import operators, circulant
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
    print("figs/spectrum.pdf")


def draw_cells():
    """figs/cells.pdf (fig:cells): the three test cells at N = 16."""
    fig, ax = plt.subplots(1, 3, figsize=(5.0, 1.9))
    Nc = 16
    for a_, g in zip(ax, cells.GEOMETRIES):
        img = cells.chi(g, Nc).reshape(Nc, Nc).T                   # rows y, columns x
        a_.imshow(img, origin="lower", cmap="Greys", vmin=-0.4, vmax=1.3,
                  extent=(0, 1, 0, 1), interpolation="nearest")
        for k in range(Nc + 1):
            a_.axhline(k / Nc, color="0.85", lw=0.3); a_.axvline(k / Nc, color="0.85", lw=0.3)
        a_.set_xticks([0, 1]); a_.set_yticks([0, 1]); a_.tick_params(labelsize=7)
        a_.set_title(f"{g}, $v_f = {cells.chi(g, Nc).mean():.4g}$".replace("0.2344", "15/64")
                     .replace("0.25", "1/4").replace("0.5", "1/2"), fontsize=8)
    fig.tight_layout(); fig.savefig("figs/cells.pdf"); print("figs/cells.pdf")


def draw_mesh(Ns=(16, 32, 64, 128, 256, 512, 1024), rho=RHO, eps_p=1e-6):
    """figs/mesh.pdf (fig:mesh): kappa of K^chi and of Mtilde, the inverse error at fixed d, and the
    Toffoli count of U_V, for the three cells, N = 16 ... 1024; asserts every number Sec. 4.2 quotes."""
    import time
    rows = []
    for g in cells.GEOMETRIES:
        for N in Ns:
            t0 = time.time()
            c = cells.Cell(g, N, rho)
            lo, hi = cells.extreme_eigs(c)
            lk = cells.lambda_min_Kc(c)
            err, d = cells.inverse_error(c, eps_p)
            delta, F = cells.encoding_bits(N, rho, eps_p)
            lamK = 2 * (1 - np.cos(2 * np.pi / N))
            r = dict(geometry=g, N=N, kappa_M=1 / lo, lam_max=hi, kappa_Kc=cells.ALPHA / lk,
                     kappa_P=cells.ALPHA * rho / lamK, inv_err=err, d=d, F=F,
                     toffoli_uv=cells.uv_counts(N, F)["toffoli"], seconds=time.time() - t0)
            rows.append(r)
            print(f"mesh {g:8s} N={N:5d} kappa(M)={r['kappa_M']:.10f} lmax={hi:.12f} "
                  f"kappa(Kc)={r['kappa_Kc']:.4g} kappa_P={r['kappa_P']:.4g} inverse err={err:.3e} d={d} "
                  f"F={F} Toffoli/U_V={r['toffoli_uv']:,} [{r['seconds']:.0f} s]", flush=True)
    # ---- assertions: every claim of Sec. 4.2
    _, d0, _, _ = cells.inverse_plan(rho, eps_p)
    for r in rows:
        assert abs(r["kappa_M"] - rho) < 1e-6 * rho and abs(r["lam_max"] - 1) < 1e-8, r   # kappa = rho
        assert r["inv_err"] <= eps_p and r["d"] == d0, r                                   # within eps_p, flat d
        assert r["kappa_P"] > r["kappa_Kc"], r
    for g in cells.GEOMETRIES:
        k = [r["kappa_Kc"] for r in rows if r["geometry"] == g]
        ratios = np.array(k[1:]) / np.array(k[:-1])
        assert np.all(ratios > 3.5) and abs(ratios[-1] - 4) < 0.02, (g, ratios)          # Theta(N^2)
    tof = {r["N"]: r["toffoli_uv"] for r in rows if r["geometry"] == "square"}
    Fs = {r["N"]: r["F"] for r in rows if r["geometry"] == "square"}
    errs = [r["inv_err"] for r in rows]
    print(f"mesh: inverse error {min(errs):.2e} to {max(errs):.2e} (eps_p = {eps_p:g}, d = {d0}); "
          f"F = {Fs[Ns[0]]} to {Fs[Ns[-1]]}; Toffolis per U_V {tof[Ns[0]]:,} (N={Ns[0]}) to {tof[Ns[-1]]:,} (N={Ns[-1]})")

    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35))
    ndof = lambda g: [r["N"] ** 2 for r in rows if r["geometry"] == g]
    for g in cells.GEOMETRIES:
        ls, col, mk = STYLE[g]
        ax[0].loglog(ndof(g), [r["kappa_Kc"] for r in rows if r["geometry"] == g], ls, color=col,
                     marker=mk, ms=3, lw=1, label=g)
        ax[1].loglog(ndof(g), [r["inv_err"] for r in rows if r["geometry"] == g], ls, color=col,
                     marker=mk, ms=3, lw=1, label=g)
    ax[0].loglog(ndof("square"), [r["kappa_M"] for r in rows if r["geometry"] == "square"], "-",
                 color="k", lw=2.2, label=r"$\kappa(\tilde{\mathbf{M}})$, all")
    ax[0].set_xlabel(r"$N_{\mathrm{dof}}$"); ax[0].set_ylabel("condition number seen by QSVT")
    ax[0].set_ylim(3, 3e7)
    ax[0].legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.12),
                 handlelength=1.6, labelspacing=0.25)
    ax[0].set_title(r"(a) $\kappa(\mathbf{K}^{\chi})$ and $\kappa(\tilde{\mathbf{M}})$", fontsize=9)
    ax[1].axhline(eps_p, color="0.5", lw=0.8, ls="--")
    ax[1].text(ndof("square")[0], eps_p * 1.25, r"$\epsilon_p$", fontsize=8)
    ax[1].set_ylim(1e-7, 1e-5); ax[1].set_xlabel(r"$N_{\mathrm{dof}}$")
    ax[1].set_ylabel("relative energy-norm error")
    ax[1].set_title(f"(b) inverse, cell source, $d = {d0}$", fontsize=9)
    ax[2].semilogx(ndof("square"), [tof[N] / 1e3 for N in Ns], "-o", color="k", ms=3, lw=1)
    ax[2].set_xlabel(r"$N_{\mathrm{dof}}$"); ax[2].set_ylabel(r"Toffoli gates ($10^3$)")
    ax[2].set_title(r"(c) one application of $\mathcal{U}_V$", fontsize=9)
    ax[2].set_ylim(0, None)
    fig.tight_layout(); fig.savefig("figs/mesh.pdf"); print("figs/mesh.pdf")


def draw_isometry():
    """figs/isometry_cost.pdf: encoding error of U_V (App. A); checks the Toffoli fit of App. A."""
    from verify_full_unitary import C_BOUND, full_error
    from quantuminversionpoisson.isometry_circuit import IsometryCircuit
    Fs = (12, 16, 20, 24, 28, 32)
    fig, ax0 = plt.subplots(figsize=(3.6, 2.5))
    ax = [ax0]
    worst, cases = 0.0, 0
    shades = np.linspace(0.75, 0.0, 7)
    for n, g in zip(range(2, 9), shades):
        N = 2 ** n
        c = []
        for F in Fs:
            blk, full, chk = full_error(n, F)
            assert chk["work_bits_left_set"] == 0 and chk["inputs_restored"]     # "work register returns to zero"
            c.append(full * 2.0 ** F / N)
            cases += 1
        worst = max(worst, max(c))
        ax[0].plot(Fs, c, "-o", color=str(g), ms=3, lw=1.0, label=f"$N = {N}$")
        print(f"isometry: N = {N}: max delta 2^F/N = {max(c):.2f}", flush=True)
    assert cases == 42 and worst <= C_BOUND and round(worst, 2) == 5.73, worst   # "42 cases", "delta <= 6 N 2^-F"
    ax[0].axhline(C_BOUND, color="0.4", lw=0.8, ls="--")
    ax[0].text(12.3, C_BOUND * 1.04, r"$\delta = 6N\,2^{-F}$", fontsize=8)
    ax[0].set_xlabel("fractional bits $F$"); ax[0].set_ylabel(r"$\delta\,2^{F}/N$")
    ax[0].set_xticks(Fs); ax[0].set_ylim(0, 9.5)
    ax[0].legend(loc="upper center", frameon=False, ncol=4, fontsize=6.5,
                 columnspacing=0.8, handlelength=1.5, bbox_to_anchor=(0.5, 1.02))

    Fr = np.arange(12, 37)
    dev, resid = 0.0, None
    for n in range(2, 11):                                                         # N = 4 ... 1024
        T = np.array([IsometryCircuit(n=n, F=F).resources()["toffoli_total"] for F in Fr])
        fit = 20.5 * Fr ** 2 + 64 * Fr + 41 + 4 * n
        dev = max(dev, np.max(np.abs(T - fit) / T))
        if resid is None:
            resid = T - fit
        assert np.allclose(T - fit, resid)                 # N enters only through 4 log2 N
    assert round(100 * dev, 2) == 0.43, dev                                        # "within 0.43 percent"
    fig.tight_layout(); fig.savefig("figs/isometry_cost.pdf")
    print(f"figs/isometry_cost.pdf: worst delta 2^F/N = {worst:.2f}, worst fit deviation {100 * dev:.2f}%")


def T(t, z):
    """Chebyshev polynomial T_t, valid inside and outside [-1, 1]."""
    z = np.asarray(z, float); out = np.empty_like(z); i = np.abs(z) <= 1
    out[i] = np.cos(t * np.arccos(z[i]))
    out[~i] = np.sign(z[~i]) ** t * np.cosh(t * np.arccosh(np.abs(z[~i])))
    return out


def draw_qsvt_poly():
    """figs/qsvt_poly.pdf: the odd QSVT polynomial q_t, rho = 10, eps_p = 1e-3 (not in the paper)."""
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


def draw_angles():
    """figs/angles.pdf: omega(k) and the hourglass fraction sin^2 tau(k), N = 32 (not in the paper)."""
    from quantuminversionpoisson.isometry_circuit import exact_angles
    N = 32
    om = np.full((N, N), np.nan); hg = np.full((N, N), np.nan)
    for k1 in range(N):
        for k2 in range(N):
            if (k1, k2) == (0, 0):
                continue
            o, t_ = exact_angles(N, k1, k2)
            hg[k1, k2] = np.sin(t_) ** 2
            if (k1, k2) != (N // 2, N // 2):
                om[k1, k2] = np.degrees(o)
    assert np.nanmax(np.abs(hg[:, 0])) < 1e-14 and np.nanmax(np.abs(hg[0, :])) < 1e-14   # "k on an axis: no hourglass"
    assert abs(hg[N // 2, N // 2] - 1) < 1e-14                                           # "(N/2, N/2): pure hourglass"
    assert abs(hg[N // 4, N // 4] - 0.25) < 1e-14                                        # "(N/4, N/4): 1/4"
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for a_, Z, ttl, lab in ((ax[0], om, r"(a) $\omega(\mathbf{k})$", "degrees"),
                           (ax[1], hg, r"(b) $\sin^2\tau(\mathbf{k})$", "")):
        im = a_.imshow(Z.T, origin="lower", cmap="viridis", extent=(-0.5, N - 0.5, -0.5, N - 0.5))
        a_.set_xlabel("$k_1$"); a_.set_ylabel("$k_2$"); a_.set_title(ttl, fontsize=9)
        a_.set_xticks([0, 8, 16, 24, 31]); a_.set_yticks([0, 8, 16, 24, 31])
        cb = fig.colorbar(im, ax=a_, fraction=0.046, pad=0.04)
        if lab:
            cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig("figs/angles.pdf")
    print("figs/spectrum.pdf, figs/qsvt_poly.pdf, figs/angles.pdf")

def draw_circuits():
    """figs/readout_circuit.pdf and figs/qsvt_circuit.pdf (companion paper)."""
    import circuit_figures
    circuit_figures.draw_circuits()


FIGURES = {
    "spectrum":  (draw_spectrum,  "eigenvalues of K^chi and of M (fig:spectrum, Sec. 2.2)"),
    "cells":     (draw_cells,     "the three test cells (fig:cells, Sec. 4.1)"),
    "mesh":      (draw_mesh,      "mesh refinement, N = 16 ... 1024 (fig:mesh, Sec. 4.2)"),
    "isometry":  (draw_isometry,  "encoding error of U_V (fig:isometry-cost, App. A)"),
}
EXTRA = {
    "circuits":  (draw_circuits,  "readout and QSVT circuits (companion paper; Qiskit, pyqsp)"),
    "qsvt_poly": (draw_qsvt_poly, "q_t at eps_p = 1e-3 (not in the paper)"),
    "angles":    (draw_angles,    "omega and sin^2 tau, N = 32 (not in the paper)"),
}


def main(argv):
    every = {**FIGURES, **EXTRA}
    if "--list" in argv:
        for k, (_, d) in every.items():
            print(f"  {k:<10} {d}")
        return
    wanted = [a for a in argv if not a.startswith("-")] or list(FIGURES)
    bad = [w for w in wanted if w not in every]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; choose from {list(every)}")
    for name in wanted:
        fn, desc = every[name]
        print(f"{name}  ({desc})", flush=True)
        fn()
    print("\ndone")


if __name__ == "__main__":
    main(sys.argv[1:])
