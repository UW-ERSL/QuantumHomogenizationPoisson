"""makeFigures.py -- the figures of the paper, written to figs/ as PDF.

    python makeFigures.py              # all figures, about 6 minutes
    python makeFigures.py circuits     # only the two circuit figures, seconds
    python makeFigures.py isometry     # only the isometry-cost figure (Sec. 3.4), about 4 minutes

figs/readout_circuit.pdf (fig:readout)  the readout circuit of verify_qiskit_qsvt.py, N = 4
figs/qsvt_circuit.pdf    (fig:qsvt)     the QSVT sequence U_q of verify_qiskit_qsvt.py, d = 3
figs/isometry_cost.pdf   (fig:isometry-cost) delta 2^F / N for the 42 cases and the Toffoli count against Eq. (toffoli)
figs/spectrum.pdf   (fig:spectrum)   eigenvalues of K^chi and of M, square inclusion, rho = 10
figs/qsvt_poly.pdf  (fig:qsvt-poly)  the odd QSVT polynomial q_t, rho = 10, eps_p = 1e-3
figs/angles.pdf     (fig:angles)     omega(k) and the hourglass fraction sin^2 tau(k), N = 32

The circuit figures are drawn from the circuits that readout() and u_q() of
verify_qiskit_qsvt.py build, with U_V, U_M, U_q and the material reflection
as labeled blocks; they are written as quantikz (figs/*_circuit.tex) and
compiled with pdflatex. Every number the text quotes about a figure is
ASSERTED here.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO = 10.0
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42})
os.makedirs("figs", exist_ok=True)


# ---- circuit figures ---------------------------------------------------------
from qiskit.circuit import ControlledGate, Gate, Parameter
import verify_qiskit_qsvt as vq


class Box(Gate):
    """Labeled block standing for a subcircuit; its inverse carries the dagger label."""
    def __init__(self, n, label, inv_label=None):
        super().__init__("box", n, [], label=label)
        self.inv_label = label if inv_label is None else inv_label

    def inverse(self, annotated=False):
        return Box(self.num_qubits, self.inv_label, self.label)


class CBox(ControlledGate):
    """Singly controlled Box; self-inverse as used here (-R_chi)."""
    def __init__(self, n, label):
        super().__init__("cbox", n + 1, [], num_ctrl_qubits=1,
                         base_gate=Box(n, label), label=label)

    def inverse(self, annotated=False):
        return CBox(self.num_qubits - 1, self.base_gate.label)


def tex_param(p):
    s = str(p).replace("*", "")
    s = re.sub(r"phi_(\d+)", r"\\phi_{\1}", s)
    return s.replace("theta", r"\theta")


def quantikz(circ, rows, labels, bundles, meters):
    """quantikz source of circ. rows[q] is the row of qubit q, labels[r] the
    label of row r, bundles[r] the number of qubits merged into row r; a
    single-qubit gate applied to every qubit of a bundled row in succession is
    drawn once, with superscript tensor power."""
    ops, data, i = [], circ.data, 0
    while i < len(data):
        ins = data[i]
        q = [circ.find_bit(b).index for b in ins.qubits]
        name, op = ins.operation.name, ins.operation
        if len(q) == 1 and bundles[rows[q[0]]] > 1:              # merge over the bundle
            k = bundles[rows[q[0]]]
            grp = data[i:i + k]
            assert all(g.operation.name == name and len(g.qubits) == 1
                       and rows[circ.find_bit(g.qubits[0]).index] == rows[q[0]] for g in grp), name
            base = {"h": "H", "x": "X"}[name]
            ops.append(("g", [rows[q[0]]], rf"{base}^{{\otimes m}}"))
            i += k
            continue
        r = [rows[x] for x in q]
        if name in ("h", "x"):
            ops.append(("g", r, name.upper()))
        elif name == "ry":
            ops.append(("g", r, rf"R_y({tex_param(op.params[0])})"))
        elif name == "rzz":
            assert abs(r[0] - r[1]) == 1
            ops.append(("rzz", sorted(r), rf"R_{{ZZ}}({tex_param(op.params[0])})"))
        elif name == "box":
            span = sorted(set(r))
            assert span == list(range(span[0], span[-1] + 1)), op.label
            ops.append(("box", span, op.label))
        elif name == "cbox":
            span = sorted(set(r[1:]))
            assert span == list(range(span[0], span[-1] + 1))
            ops.append(("cbox", [r[0]] + span, op.base_gate.label))
        elif name.startswith("mcx"):
            st = op.ctrl_state
            ops.append(("mcx", r, [(st >> j) & 1 for j in range(len(r) - 1)]))
        else:
            raise ValueError(name)
        i += 1

    nrow = len(labels)
    last, cols = [0] * nrow, []
    for kind, r, _ in reversed(ops):                             # as late as possible
        span = range(min(r), max(r) + 1)
        c = max(last[x] for x in span)
        for x in span:
            last[x] = c + 1
        cols.append(c)
    ncol = max(last)
    cols = [ncol - 1 - c for c in reversed(cols)]
    grid = [[""] * ncol for _ in range(nrow)]
    for (kind, r, lab), c in zip(ops, cols):
        if kind == "g":
            grid[r[0]][c] = rf"\gate{{{lab}}}"
        elif kind in ("box", "rzz"):
            grid[r[0]][c] = rf"\gate[{len(r)}]{{{lab}}}"
        elif kind == "cbox":
            grid[r[0]][c] = rf"\ctrl{{{r[-1] - r[0]}}}"
            grid[r[1]][c] = rf"\gate[{len(r) - 1}]{{{lab}}}"
        elif kind == "mcx":
            *ctl, tgt = r
            nodes = sorted(r)
            for x, closed in zip(ctl, lab):
                j = nodes.index(x)
                nxt = nodes[j + 1] if x < tgt else nodes[j - 1]
                grid[x][c] = (r"\ctrl" if closed else r"\octrl") + f"{{{nxt - x}}}"
            grid[tgt][c] = r"\targ{}"
    lines = []
    for x in range(nrow):
        head = rf"\lstick{{{labels[x]}}} & " + (rf"\qwbundle{{m}}" if bundles[x] > 1 else "")
        tail = r" & \meter{}" if meters else r" & "
        lines.append(head + " & " + " & ".join(grid[x]) + tail)
    return ("\\begin{quantikz}[column sep=0.18cm, row sep={0.7cm,between origins}]\n"
            + " \\\\\n".join(lines) + "\n\\end{quantikz}\n")


def compile_circuit(body, name):
    tex = ("\\documentclass[border=2pt]{standalone}\n\\usepackage{amsmath,amssymb}\n"
           "\\usepackage{quantikz}\n\\begin{document}\n" + body + "\\end{document}\n")
    open(f"figs/{name}.tex", "w").write(tex)
    if shutil.which("pdflatex") is None:
        print(f"figs/{name}.tex written; pdflatex not found, PDF not built")
        return
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"],
                         cwd="figs", capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:]
    for ext in ("aux", "log"):
        os.remove(f"figs/{name}.{ext}")
    print(f"figs/{name}.pdf")


def draw_circuits():
    """figs/readout_circuit.pdf and figs/qsvt_circuit.pdf from verify_qiskit_qsvt.py."""
    from qiskit import QuantumCircuit
    m = vq.m
    UV = Box(m + 4, r"\mathcal{U}_V", r"\mathcal{U}_V^{\dagger}")
    UM = Box(m + 5, r"\mathcal{U}_M", r"\mathcal{U}_M^{\dagger}")
    Uq = Box(m + 7, r"\mathcal{U}_q")
    th = Parameter("theta")

    # U_Sigma: Ry(theta) on its ancilla, -R_chi controlled by it, Ry(-theta)
    ref = vq.u_sigma(np.ones(vq.N * vq.N), vq.RHO)
    assert [d.operation.name for d in ref.data] == ["ry", "diagonal", "ry"]
    assert all(ref.find_bit(ref.data[j].qubits[0]).index == m for j in (0, 2))
    US = QuantumCircuit(m + 1, name="U_Sigma")
    US.ry(th, m)
    US.append(CBox(m, r"-R_{\chi}"), [m] + list(range(m)))
    US.ry(-th, m)

    # readout: pos (m), loc (4), qa, z, c, s1, s2, f1, f2 of verify_qiskit_qsvt.py,
    # labeled s_0, z_1, z_2, s_1, s_2, f_1, f_2 in the paper
    R = vq.readout(UV, US, Uq).decompose(gates_to_decompose=["U_Sigma", "U_Sigma_dg"])
    assert R.num_qubits == 15                                                  # "15 qubits"
    rows = [0] * m + list(range(1, 12))
    labels = [r"position", r"local $\ell_0$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$",
              r"$s_0$", r"$z_1$", r"$z_2$", r"$s_1$", r"$s_2$", r"$f_1$", r"$f_2$"]
    compile_circuit(quantikz(R, rows, labels, [m] + [1] * 11, meters=True), "readout_circuit")

    # U_q: d queries to U_M and U_M^dag between d + 1 projector phases
    for d in (3, 15):
        Q = vq.u_q(UM, [Parameter(f"phi_{j}") for j in range(d + 1)])
        assert sum(x.operation.name == "box" for x in Q.data) == d             # "d queries to U_M"
    Q = vq.u_q(UM, [Parameter(f"phi_{j}") for j in range(4)])
    rows = [0] * m + list(range(1, 8))
    labels = [r"position", r"local $\ell_0$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$",
              r"$s_0$", r"$z_1$", r"$z_2$"]
    compile_circuit(quantikz(Q, rows, labels, [m] + [1] * 7, meters=False), "qsvt_circuit")


def draw_isometry():
    """figs/isometry_cost.pdf: encoding error of U_V (Sec. 3.4); checks the Toffoli fit of Sec. 5.5."""
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
        fit = 41 * Fr ** 2 + 168 * Fr + 117 + 4 * n
        dev = max(dev, np.max(np.abs(T - fit) / T))
        if resid is None:
            resid = T - fit
        assert np.array_equal(T - fit, resid)              # N enters only through 4 log2 N
    assert round(100 * dev, 2) == 0.54, dev                                        # "within 0.54 percent"
    fig.tight_layout(); fig.savefig("figs/isometry_cost.pdf")
    print(f"figs/isometry_cost.pdf: worst delta 2^F/N = {worst:.2f}, worst fit deviation {100 * dev:.2f}%")


NEW_FIGURES = {"circuits": draw_circuits, "isometry": draw_isometry}
ONLY = sys.argv[1:]
assert all(x in NEW_FIGURES for x in ONLY), f"choose from {list(NEW_FIGURES)}"
for name, draw in NEW_FIGURES.items():
    if not ONLY or name in ONLY:
        draw()
if ONLY:
    sys.exit()

from verify_square import operators, circulant
from quantuminversionpoisson.isometry_circuit import exact_angles

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

# ---- the two angles over the wavenumbers -------------------------------------
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