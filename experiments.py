"""experiments.py -- the numerical experiments of Sec. 5 (block encoding and its inverse).

    python experiments.py              # all experiments, tables/ and figs/
    python experiments.py mesh         # one experiment: mesh, geometry, resources

Three dyadic cells, each with a material oracle R_chi of O(1) gates:

    square     centered square inclusion of side N/2, v_f = 1/4 (the worked example)
    cross      two orthogonal bands of width N/8 through the center, v_f = 15/64
               (the conducting phase is connected)
    laminate   chi = 1 for x < 1/2, v_f = 1/2 (layers normal to x; exact sigma^H)

Every operator is applied matrix-free (FFT for functions of K, sparse
A = W^{1/2} G), so the experiments reach N = 1024 (N_dof ~ 10^6). Every
number the paper quotes is ASSERTED here; results are written to
experiments.json, tables/*.tex and figs/*.pdf.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantuminversionpoisson.fourier_symbol import symbol
from quantuminversionpoisson.isometry_circuit import IsometryCircuit
from numpy.polynomial import chebyshev as Cheb

from verify_inverse import k_function, q_poly
from verify_square import _B, _CORNERS, assemble

ALPHA = 16 / 3            # subnormalization of the encoding of K^chi / sigma_max (Table 2 baseline)
DELTA_CONST = 6           # delta <= 6 N 2^-F (Sec. 4.4)
GEOMETRIES = ("square", "cross", "laminate")
os.makedirs("tables", exist_ok=True)
os.makedirs("figs", exist_ok=True)


# ==========================================================================
# 1. geometries and their material oracles
# ==========================================================================
def chi(name, N):
    """Phase indicator per element e = i N + j (i along x): 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if name == "square":
        c = (i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)
    elif name == "cross":
        band = lambda k: (k >= 7 * N // 16) & (k < 9 * N // 16)
        c = band(i) | band(j)
    elif name == "laminate":
        c = i < N // 2
    else:
        raise ValueError(name)
    return c.ravel().astype(float)


def oracle_network(name):
    """Gate list of controlled(-R_chi) on (s, i bits, j bits, work bits).

    -R_chi is +1 in the inclusion and -1 in the matrix, so its controlled form
    is the phase (-1)^{s (1 - chi)} = (-1)^s (-1)^{s chi}. Only the leading
    bits of each coordinate enter: b0 = most significant, b1, b2, b3. Gates:
    ("x", t), ("z", t), ("cx", c, t), ("cz", a, b), ("ccx", a, b, t, pa, pb)
    with control polarities pa, pb (1 = filled, 0 = open), ("ccz", a, b, c).
    """
    if name == "laminate":                      # chi = not i0: (-1)^{s (1 - chi)} = CZ(s, i0)
        return [("cz", "s", "i0")]
    g = [("z", "s")]
    if name == "square":                        # chi = (i0 xor i1) and (j0 xor j1)
        comp = [("cx", "i0", "i1"), ("cx", "j0", "j1"), ("ccx", "i1", "j1", "t", 1, 1)]
        return g + comp + [("cz", "s", "t")] + comp[::-1]
    if name == "cross":                         # band: top four bits in {0111, 1000}
        comp = []
        for c in "ij":
            comp += [("cx", f"{c}2", f"{c}3"), ("cx", f"{c}1", f"{c}2"), ("cx", f"{c}0", f"{c}1")]
            # now c1 = b0^b1 (x), c2 = b1^b2 (y), c3 = b2^b3 (z): band = x & !y & !z
            comp += [("ccx", f"{c}1", f"{c}2", f"u{c}", 1, 0),
                     ("ccx", f"u{c}", f"{c}3", f"a{c}", 1, 0)]
        # (-1)^{s (A or B)} = CZ(s,A) CZ(s,B) CCZ(s,A,B)
        mid = [("cz", "s", "ai"), ("cz", "s", "aj"), ("ccz", "s", "ai", "aj")]
        return g + comp + mid + comp[::-1]
    raise ValueError(name)


def oracle_phase(name, n):
    """Simulate oracle_network on every (s, position): returns the phase table and counts."""
    N = 2 ** n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i, j = i.ravel(), j.ravel()
    out = {}
    for s in (0, 1):
        bits = {"s": np.full(N * N, s)}
        for k in range(4):                       # leading bits, b0 most significant
            bits[f"i{k}"] = (i >> (n - 1 - k)) & 1 if k < n else np.zeros_like(i)
            bits[f"j{k}"] = (j >> (n - 1 - k)) & 1 if k < n else np.zeros_like(j)
        phase = np.ones(N * N)
        for gate in oracle_network(name):
            for w in gate[1:4]:
                if isinstance(w, str):
                    bits.setdefault(w, np.zeros(N * N, int))
            op = gate[0]
            if op == "z":
                phase *= (-1.0) ** bits[gate[1]]
            elif op == "cz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]])
            elif op == "ccz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]] & bits[gate[3]])
            elif op == "cx":
                bits[gate[2]] = bits[gate[2]] ^ bits[gate[1]]
            elif op == "ccx":
                a, b, t, pa, pb = gate[1:]
                bits[t] = bits[t] ^ ((bits[a] == pa) & (bits[b] == pb)).astype(int)
        work = [w for w in bits if w[0] in "uat" and w != "s"]
        assert all(not bits[w].any() for w in work), name                  # work bits clean
        for k in range(min(4, n)):                                           # inputs restored
            assert np.array_equal(bits[f"i{k}"], (i >> (n - 1 - k)) & 1)
        out[s] = phase
    net = oracle_network(name)
    counts = dict(toffoli=sum(g[0] in ("ccx", "ccz") for g in net),
                  cnot=sum(g[0] in ("cx", "cz") for g in net),
                  single=sum(g[0] in ("x", "z") for g in net))
    return out, counts


_ORC = {}


def check_oracle(name):
    """controlled(-R_chi) acts as I for s = 0 and 2 chi - 1 for s = 1, at N = 16 ... 256."""
    if name in _ORC:
        return _ORC[name]
    for n in range(4, 9):
        ph, counts = oracle_phase(name, n)
        assert np.all(ph[0] == 1), name
        assert np.array_equal(ph[1], 2 * chi(name, 2 ** n) - 1), (name, n)
    _ORC[name] = counts
    return counts


# ==========================================================================
# 2. operators, matrix-free
# ==========================================================================
_A = {}


def A_matrix(N):
    """A = W^{1/2} G (8 N^2 x N^2), vectorized; equals verify_square.assemble."""
    if N in _A:
        return _A[N]
    h = 1 / N
    i, j = np.divmod(np.arange(N * N), N)
    rows, cols, vals = [], [], []
    for g in range(4):
        Bg = _B(g, h)
        for a, (cx, cy) in enumerate(_CORNERS):
            node = ((i + cx) % N) * N + (j + cy) % N
            for c in range(2):
                rows.append(8 * np.arange(N * N) + 2 * g + c)
                cols.append(node)
                vals.append(np.full(N * N, Bg[c, a]))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(8 * N * N, N * N)) * (h / 2)
    _A[N] = A
    return A


class Cell:
    """Two-phase cell: N, rho, geometry; Mtilde, K^chi and the cell-problem source, matrix-free."""

    def __init__(self, name, N, rho):
        self.name, self.N, self.rho = name, N, rho
        self.A = A_matrix(N)
        self.chi = chi(name, N)
        self.vf = self.chi.mean()
        self.St = np.repeat(np.where(self.chi == 1, 1.0, 1 / rho), 8)       # Sigma / sigma_max
        self.sbar = self.vf + (1 - self.vf) / rho
        self.Kih = k_function(lambda e: e ** -0.5, N)
        self.Kinv = k_function(lambda e: 1 / e, N)
        self.n = N * N

    # Mtilde = M + sbar Pi_0, M = V^T Sigmatilde V, V = A (K^+)^{1/2}
    def Mt(self, v):
        y = self.Kih(v)
        out = self.Kih(self.A.T @ (self.St * (self.A @ y)))
        return out + self.sbar * v.mean() * np.ones_like(v)

    def Mt_op(self):
        return spla.LinearOperator((self.n, self.n), matvec=lambda v: self.Mt(np.ravel(v)), dtype=float)

    def Kc(self, u):                                   # K^chi / sigma_max
        return self.A.T @ (self.St * (self.A @ u))

    def ehat(self, a):
        v = np.zeros(8 * self.n)
        v[(a - 1)::2] = 1 / (2 * self.N)
        return v


def pcg(Aop, Pop, b, tol, x_ref=None, energy_tol=None, maxiter=5000):
    """Preconditioned CG on the complement of the constants. Returns (x, iterations).
    With x_ref and energy_tol, stops when the relative A-norm error reaches energy_tol."""
    b = b - b.mean()
    x = np.zeros_like(b)
    r = b.copy()
    z = Pop(r); z -= z.mean()
    p = z.copy()
    rz = r @ z
    nb = np.sqrt(b @ Pop(b))
    ref_norm = None if x_ref is None else np.sqrt(x_ref @ Aop(x_ref))
    for it in range(1, maxiter + 1):
        Ap = Aop(p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        if x_ref is not None:
            e = x - x_ref
            if np.sqrt(e @ Aop(e)) <= energy_tol * ref_norm:
                return x, it
        z = Pop(r); z -= z.mean()
        rz_new = r @ z
        if x_ref is None and np.sqrt(abs(rz_new)) <= tol * nb:
            return x, it
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError("pcg did not converge")


def extreme_eigs(cell, tol=1e-9):
    """Smallest and largest eigenvalues of Mtilde (Lanczos)."""
    op = cell.Mt_op()
    lo = spla.eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False)[0]
    hi = spla.eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
    return lo, hi


def lambda_min_Kc(cell, tol=1e-9, seed=0):
    """Smallest nonzero eigenvalue of K^chi / sigma_max by LOBPCG, preconditioned by K^+,
    with the constant field as constraint."""
    n = cell.n
    Aop = spla.LinearOperator((n, n), matvec=lambda v: cell.Kc(np.ravel(v)), dtype=float)
    P = spla.LinearOperator((n, n), matvec=lambda v: cell.Kinv(np.ravel(v)), dtype=float)
    X = np.random.default_rng(seed).standard_normal((n, 1))
    X -= X.mean()
    lam, _ = spla.lobpcg(Aop, X, M=P, Y=np.ones((n, 1)), tol=tol, maxiter=500, largest=False)
    return float(lam[0])


# ==========================================================================
# 3. the QSVT inverse polynomial applied matrix-free (Sec. 3.7)
# ==========================================================================
# q_t is the odd polynomial of Sec. 2.4 at kappa = rho with t = ceil(rho/2 ln(2/eps_p)) (Prop. qsvt-poly),
# degree d = 2t - 1. It is applied to Mtilde by the Clenshaw recurrence on its Chebyshev interpolant of
# degree d, which equals q_t up to rounding, with d applications of Mtilde.
_PLAN = {}


def inverse_plan(rho, eps_p):
    """q_t, its degree d, its Chebyshev coefficients, and C = max_[-1,1] |q_t| (Eq. qsvt-norm)."""
    if (rho, eps_p) not in _PLAN:
        q, d = q_poly(rho, eps_p)
        x = np.linspace(1e-9, 1.0, 2_000_001)
        C = float(np.abs(q(x)).max())
        _PLAN[rho, eps_p] = (q, d, Cheb.chebinterpolate(q, d), C)
    return _PLAN[rho, eps_p]


def encoding_bits(N, rho, eps_p):
    """Sec. 4.4: delta = eps_p / (2 d C), so that the encoding error adds at most eps_p to the energy-norm
    error, and F = ceil(log2(6 N / delta)) by Eq. (delta-bound)."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta = eps_p / (2 * d * C)
    return delta, int(np.ceil(np.log2(DELTA_CONST * N / delta)))


def apply_q(cell, coef, b):
    """q_t(Mtilde) b by the Clenshaw recurrence."""
    b1 = np.zeros_like(b); b2 = np.zeros_like(b)
    for c in coef[:0:-1]:
        b1, b2 = 2 * cell.Mt(b1) - b2 + c * b, b1
    return cell.Mt(b1) - b2 + coef[0] * b


def inverse_error(cell, eps_p, a=1):
    """Relative energy-norm error of u_Q = (K^+)^{1/2} q_t(Mtilde) (K^+)^{1/2} f against the cell solution,
    for the cell-problem source f of the macroscopic gradient e_a (Eq. energy-error), sigma_max = 1."""
    _, d, coef, _ = inverse_plan(cell.rho, eps_p)
    f = -(cell.A.T @ (cell.St * cell.ehat(a)))
    u, _ = pcg(cell.Kc, cell.Kinv, f, 1e-13)
    uq = cell.Kih(apply_q(cell, coef, cell.Kih(f)))
    e = uq - u
    return float(np.sqrt((e @ cell.Kc(e)) / (u @ cell.Kc(u)))), d


# ==========================================================================
# 4. resource counts
# ==========================================================================
_UV = {}


def uv_counts(N, F):
    """Toffoli count, controlled rotations and qubits of one application of U_V (gate level, cached)."""
    if (N, F) not in _UV:
        n = int(np.log2(N))
        r = IsometryCircuit(n=n, F=F).resources()
        _UV[N, F] = dict(toffoli=r["toffoli_total"], cordic_rot=2 * F + 1 + 3,
                         qft_rot=2 * 2 * n * (n - 1) // 2, phase_rot=r["phase_gates"],
                         qubits=r["wires"])
    return _UV[N, F]


def inverse_resources(name, N, rho, eps_p):
    """Sec. 3.7: one query to U_M is U_V, U_Sigma (one controlled R_chi), U_V^dag; d queries."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta, F = encoding_bits(N, rho, eps_p)
    uv = uv_counts(N, F)
    t_rchi = check_oracle(name)["toffoli"]
    per_query = 2 * uv["toffoli"] + t_rchi
    # two U_V (CORDIC rotations, QFT controlled phases, the 2n phase gates of
    # e^{i(phi1+phi2)}), and the two Ry of U_Sigma
    rot = 2 * (uv["cordic_rot"] + uv["qft_rot"] + uv["phase_rot"]) + 2
    return dict(name=name, N=N, rho=rho, eps_p=eps_p, d=d, C=C, delta=delta, F=F,
                toffoli_uv=uv["toffoli"], toffoli_per_query=per_query, rot_per_query=rot,
                toffoli_total=float(d) * per_query,
                qubits=uv["qubits"] + 3)    # + the U_Sigma ancilla, the flag qubit, the QSVT ancilla


# ==========================================================================
# 5. output helpers
# ==========================================================================
def sci(x, digits=2):
    """LaTeX number: 1.23 x 10^{k} for large or small x, plain otherwise."""
    if x == 0:
        return "$0$"
    k = int(np.floor(np.log10(abs(x))))
    if -2 <= k <= 3:
        s = f"{x:.{max(digits - 1 - k, 0)}f}" if k < digits else f"{x:,.0f}".replace(",", "{,}")
        return f"${s}$"
    m = x / 10 ** k
    if round(m, digits - 1) >= 10:
        m, k = m / 10, k + 1
    return f"${m:.{digits - 1}f} \\times 10^{{{k}}}$"


def p10(x):
    """$10^{k}$ for an exact power of ten."""
    k = int(round(np.log10(x)))
    assert abs(x - 10.0 ** k) < 1e-9 * x
    return "$10$" if k == 1 else f"$10^{{{k}}}$"


def intx(x):
    return f"${int(x):,}$".replace(",", "{,}")


def write_table(name, header, rows, align):
    out = ["% generated by experiments.py -- do not edit",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + "\\\\", "\\midrule"]
    out += [" & ".join(r) + "\\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    open(f"tables/{name}.tex", "w").write("\n".join(out) + "\n")
    print(f"tables/{name}.tex")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    return plt


STYLE = {"square": ("-", "k", "o"), "cross": ("--", "0.35", "s"), "laminate": (":", "0.55", "^")}
RESULTS = {}


# ==========================================================================
# 6. Experiment 1: mesh independence
# ==========================================================================
def exp_mesh(Ns=(16, 32, 64, 128, 256, 512, 1024), rho=10.0, eps_p=1e-6):
    rows = []
    for g in GEOMETRIES:
        for N in Ns:
            t0 = time.time()
            c = Cell(g, N, rho)
            lo, hi = extreme_eigs(c)
            lk = lambda_min_Kc(c)
            err, d = inverse_error(c, eps_p)
            delta, F = encoding_bits(N, rho, eps_p)
            lamK = 2 * (1 - np.cos(2 * np.pi / N))
            r = dict(geometry=g, N=N, kappa_M=1 / lo, lam_max=hi, kappa_Kc=ALPHA / lk,
                     kappa_P=ALPHA * rho / lamK, inv_err=err, d=d, F=F,
                     toffoli_uv=uv_counts(N, F)["toffoli"], seconds=time.time() - t0)
            rows.append(r)
            print(f"mesh {g:8s} N={N:5d} kappa(M)={r['kappa_M']:.10f} lmax={hi:.12f} "
                  f"kappa(Kc)={r['kappa_Kc']:.4g} kappa_P={r['kappa_P']:.4g} inverse err={err:.3e} d={d} "
                  f"F={F} Toffoli/U_V={r['toffoli_uv']:,} [{r['seconds']:.0f} s]", flush=True)
    RESULTS["mesh"] = rows
    # ---- assertions: every claim of Sec. 5.2
    _, d0, _, _ = inverse_plan(rho, eps_p)
    for r in rows:
        assert abs(r["kappa_M"] - rho) < 1e-6 * rho and abs(r["lam_max"] - 1) < 1e-8, r   # kappa = rho
        assert r["inv_err"] <= eps_p and r["d"] == d0, r                                   # within eps_p, flat d
        assert r["kappa_P"] > r["kappa_Kc"], r
    for g in GEOMETRIES:
        k = [r["kappa_Kc"] for r in rows if r["geometry"] == g]
        ratios = np.array(k[1:]) / np.array(k[:-1])
        assert np.all(ratios > 3.5) and abs(ratios[-1] - 4) < 0.02, (g, ratios)          # Theta(N^2)
    tof = {r["N"]: r["toffoli_uv"] for r in rows if r["geometry"] == "square"}
    Fs = {r["N"]: r["F"] for r in rows if r["geometry"] == "square"}
    errs = [r["inv_err"] for r in rows]
    print(f"mesh: inverse error {min(errs):.2e} to {max(errs):.2e} (eps_p = {eps_p:g}, d = {d0}); "
          f"F = {Fs[Ns[0]]} to {Fs[Ns[-1]]}; Toffolis per U_V {tof[Ns[0]]:,} (N={Ns[0]}) to {tof[Ns[-1]]:,} (N={Ns[-1]})")

    plt = plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35))
    ndof = lambda g: [r["N"] ** 2 for r in rows if r["geometry"] == g]
    for g in GEOMETRIES:
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


# ==========================================================================
# 7. Experiment 2: geometries (tensor, kappa, cancellation)
# ==========================================================================
def exp_geometry(N=256, rhos=(10.0, 1e4)):
    rows = []
    for g in GEOMETRIES:
        orc = check_oracle(g)
        for rho in rhos:
            c = Cell(g, N, rho)
            lo, hi = extreme_eigs(c)
            r = dict(geometry=g, rho=rho, vf=c.vf, oracle_toffoli=orc["toffoli"], kappa_M=1 / lo,
                     lam_min=lo, lam_max=hi)
            rows.append(r)
            print(f"geometry {g:8s} rho={rho:g} vf={c.vf:.4f} R_chi Toffolis={orc['toffoli']} "
                  f"lambda_min*rho={lo * rho:.10f} lambda_max={hi:.12f} kappa(M)/rho={r['kappa_M'] / rho:.8f}",
                  flush=True)
    RESULTS["geometry"] = rows
    for r in rows:
        assert abs(r["kappa_M"] - r["rho"]) < 1e-5 * r["rho"], r          # kappa = rho, every geometry
        assert abs(r["lam_max"] - 1) < 1e-8, r
    plt = plt_setup()                                        # the three cells at N = 16
    fig, ax = plt.subplots(1, 3, figsize=(5.0, 1.9))
    Nc = 16
    for a_, g in zip(ax, GEOMETRIES):
        img = chi(g, Nc).reshape(Nc, Nc).T                   # rows y, columns x
        a_.imshow(img, origin="lower", cmap="Greys", vmin=-0.4, vmax=1.3,
                  extent=(0, 1, 0, 1), interpolation="nearest")
        for k in range(Nc + 1):
            a_.axhline(k / Nc, color="0.85", lw=0.3); a_.axvline(k / Nc, color="0.85", lw=0.3)
        a_.set_xticks([0, 1]); a_.set_yticks([0, 1]); a_.tick_params(labelsize=7)
        a_.set_title(f"{g}, $v_f = {chi(g, Nc).mean():.4g}$".replace("0.2344", "15/64")
                     .replace("0.25", "1/4").replace("0.5", "1/2"), fontsize=8)
    fig.tight_layout(); fig.savefig("figs/cells.pdf"); print("figs/cells.pdf")
    trows = []
    for r in rows:
        trows.append([r["geometry"] if r["rho"] == rhos[0] else "",
                      f"${r['vf']:.3g}$" if r["rho"] == rhos[0] else "",
                      f"${r['oracle_toffoli']}$" if r["rho"] == rhos[0] else "",
                      p10(r["rho"]), f"${r['lam_min'] * r['rho']:.6f}$", f"${r['lam_max']:.6f}$",
                      f"${r['kappa_M'] / r['rho']:.6f}$"])
    write_table("geometry", ["geometry", "$v_f$", "$R_\\chi$ Toffoli", "$\\rho$",
                             "$\\rho\\,\\lambda_{\\min}$", "$\\lambda_{\\max}$",
                             "$\\kappa(\\tilde{\\mat{M}})/\\rho$"], trows, "lrrrrrr")


# ==========================================================================
# 8. Experiment 3: resources of the inverse at N = 1024 (Table resources)
# ==========================================================================
def exp_resources(N=1024, rho=10.0, eps_ps=(1e-3, 1e-6)):
    rows = [inverse_resources("square", N, rho, e) for e in eps_ps]
    for v in rows:
        print(f"resources eps_p={v['eps_p']:g}: d={v['d']} C={v['C']:.3f} (C/rho={v['C'] / rho:.3f}) "
              f"delta={v['delta']:.2e} F={v['F']} U_V Toffoli={v['toffoli_uv']:,} per query={v['toffoli_per_query']:,} "
              f"rotations/query={v['rot_per_query']} total={v['toffoli_total']:.3e} qubits={v['qubits']:,}")
    RESULTS["resources"] = rows
    trows = [[p10(v["eps_p"]), f"${v['d']}$", f"${v['C']:.1f}$", f"${v['F']}$",
              intx(v["toffoli_uv"]), intx(v["toffoli_per_query"]), f"${v['rot_per_query']}$",
              sci(v["toffoli_total"], 2), intx(v["qubits"])] for v in rows]
    write_table("resources", ["$\\epsilon_p$", "$d$", "$C$", "$F$", "Toffoli/$\\mathcal{U}_V$",
                              "Toffoli/query", "rotations/query", "Toffoli", "qubits"], trows, "rrrrrrrrr")

    # tab:uv-cost (Sec. 4.5): one application of U_V at F = 45, by stage of the angle oracle
    v = rows[-1]
    F, n = v["F"], int(np.log2(N))
    from quantuminversionpoisson.isometry_circuit import IsometryCircuit
    circ = IsometryCircuit(n=n, F=F)
    st = {x["stage"]: x for x in circ.stage_counts()}
    tot = sum(x["toffoli"] for x in st.values())
    work = sum(x["qubits"] for x in st.values())
    cordic = st["cos_sin"]["toffoli"] + st["omega"]["toffoli"] + st["tau"]["toffoli"]
    assert tot == v["toffoli_uv"] == 44510 and F == 45
    assert cordic == 20 * F ** 2 + 48 * F == 42660 and tot - cordic == 1850
    assert round(100 * cordic / tot) == 96
    assert work == 5674 and work + 2 * n + 4 == circ.resources()["wires"] == 5698
    assert v["qubits"] == 5701 and round(100 * work / v["qubits"], 1) == 99.5
    assert v["rot_per_query"] == 2 * ((2 * F + 4) + 2 * n * (n - 1) + 2 * n) + 2 == 590
    labels = [("1", "load $\\phi_1 \\pm \\phi_2$: adder and subtractor", "load"),
              ("2", "cosines and sines: two rotation-mode runs", "cos_sin"),
              ("3", "sum to product: $a$, $b$, $\\zeta$", "sum_to_product"),
              ("4", "$\\omega$: vectoring run", "omega"),
              ("4", "$\\sqrt{2/3}\\,K\\zeta$: constant product", "const_product"),
              ("4", "$\\tau$: vectoring run", "tau"),
              ("5", "zero-mode flag", "zero_flag")]
    lines = ["% generated by experiments.py -- do not edit", "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
             "stage & operation & Toffoli & qubits\\\\", "\\midrule"]
    lines += [f"{a} & {b} & {intx(st[k]['toffoli'])} & {intx(st[k]['qubits'])}\\\\" for a, b, k in labels]
    lines += ["\\midrule", f" & total & {intx(tot)} & {intx(work)}\\\\", "\\bottomrule", "\\end{tabular}"]
    open("tables/uv_cost.tex", "w").write("\n".join(lines) + "\n")
    print("tables/uv_cost.tex")
    RESULTS["uv_cost"] = dict(F=F, N=N, stages=list(st.values()), cordic=cordic, total=tot, work=work)


# ==========================================================================
EXPERIMENTS = {"mesh": exp_mesh, "geometry": exp_geometry, "resources": exp_resources}

if __name__ == "__main__":
    only = sys.argv[1:] or list(EXPERIMENTS)
    assert all(x in EXPERIMENTS for x in only), f"choose from {list(EXPERIMENTS)}"
    print("----- output --------------------------------------------------")
    if os.path.exists("experiments.json"):
        RESULTS.update(json.load(open("experiments.json")))
    for name in only:
        t0 = time.time()
        EXPERIMENTS[name]()
        json.dump(RESULTS, open("experiments.json", "w"), indent=1, default=float)
        print(f"[{name}: {time.time() - t0:.0f} s]", flush=True)
    print("---------------------------------------------------------------")