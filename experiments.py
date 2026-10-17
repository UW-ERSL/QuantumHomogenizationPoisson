"""experiments.py -- the numerical experiments of Sec. 5.

    python experiments.py              # all experiments, tables/ and figs/
    python experiments.py mesh         # one experiment: mesh, geometry, contrast, accuracy, crossover

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
from verify_inverse import k_function
from verify_readout import ae_samples
from verify_square import _B, _CORNERS, assemble

ALPHA = 16 / 3            # subnormalization of the encoding of K^chi / sigma_max (Table 2 baseline)
DELTA_CONST = 6           # delta <= 6 N 2^-F (Sec. 3.4)
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
    """Two-phase cell: N, rho, geometry; Mtilde, K^chi and the cell problem, matrix-free."""

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

    def psi(self, a):                                  # V^T Sigmatilde ehat_a
        return self.Kih(self.A.T @ (self.St * self.ehat(a)))

    def solve_Mt(self, b, tol=1e-13):
        x, info = spla.cg(self.Mt_op(), b, rtol=tol, maxiter=2000)
        assert info == 0
        return x

    def fe_sigma(self, tol=1e-12):
        """sigma^H / sigma_max from the cell problem K^chi v = -f by FFT-PCG (Moulinec-Suquet
        preconditioner K^+), independently of the quadratic form."""
        S = np.zeros((2, 2))
        grads = []
        for a in (1, 2):
            e = self.ehat(a)
            f = -(self.A.T @ (self.St * e))
            v, it = pcg(self.Kc, self.Kinv, f, tol)
            grads.append(e + self.A @ v)
        for a in range(2):
            for b in range(2):
                S[a, b] = grads[a] @ (self.St * grads[b])
        return S

    def qf_sigma(self):
        """sigma^H / sigma_max = sbar I - [psi_a^T Mtilde^{-1} psi_b] (Eq. sigmaH-pre)."""
        ps = [self.psi(a) for a in (1, 2)]
        sol = [self.solve_Mt(p) if np.linalg.norm(p) > 0 else 0 * p for p in ps]
        c = np.array([[ps[a] @ sol[b] for b in range(2)] for a in range(2)])
        return self.sbar * np.eye(2) - c, ps, sol


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
# 3. the QSVT polynomial applied matrix-free (least-squares readout)
# ==========================================================================
# sigma^H_11 / sigma_max = g^T (I - Pi) g, g = Sigma~^{1/2} ehat_1, Pi the projector onto the range of
# Vt = Sigma~^{1/2} V.  The even polynomial f(x) = (r_t(x^2) + e_t) / (1 + e_t), r_t the degree-t
# Chebyshev polynomial on [1/rho, 1] with r_t(0) = 1 and e_t = 1 / T_t(z0) = max |r_t|, satisfies
# f(0) = 1 and 0 <= f <= 1 on [-1, 1]; each of its d = 2 t queries is one U_V or U_V^dagger.
def qsvt_degree(rho, eps_p):
    """Smallest t with e_t <= eps_p; the QSVT degree is d = 2 t."""
    a = 1 / rho
    z0 = (1 + a) / (1 - a)
    return int(np.ceil(np.arccosh(1 / eps_p) / np.arccosh(z0)))


def apply_f(cell, eps_p, a=1):
    """Returns g, f(Vt) g, t and e_t, with B = Vt Vt^T applied by the Chebyshev recurrence
    (t applications of B, that is d = 2 t applications of Vt or Vt^T)."""
    rho = cell.rho
    al = 1 / rho
    z0 = (1 + al) / (1 - al)
    t = qsvt_degree(rho, eps_p)
    sq = np.sqrt(cell.St)
    g = sq * cell.ehat(a)
    Vt = lambda w: sq * (cell.A @ cell.Kih(w))
    VtT = lambda u: cell.Kih(cell.A.T @ (sq * u))
    Z = lambda u: ((1 + al) * u - 2 * Vt(VtT(u))) / (1 - al)
    T0, T1 = g, Z(g)
    for _ in range(t - 1):
        T0, T1 = T1, 2 * Z(T1) - T0
    e = 1 / np.cosh(t * np.arccosh(z0))
    r = T1 * e                                                         # r_t(B) g
    return g, (r + e * g) / (1 + e), t, e


def poly_readout(cell, eps_p, a=1):
    """sigma^H_aa / sigma_max by the QSVT polynomial: s = (1 + e_t) A - e_t sbar with A = g^T f(Vt) g >= 0.
    Returns the estimate, A, the degree d and e_t."""
    g, fg, t, e = apply_f(cell, eps_p, a)
    A = float(g @ fg)
    return (1 + e) * A - e * float(g @ g), A, 2 * t, e


def pcg_sigma_iterations(cell, tol, a=1):
    """FFT-PCG iterations until sigma_k - sigma^H <= tol (the error of the energy functional)."""
    e = cell.ehat(a)
    f = -(cell.A.T @ (cell.St * e))
    vref, _ = pcg(cell.Kc, cell.Kinv, f, 1e-13)
    gr = e + cell.A @ vref
    sH = gr @ (cell.St * gr)
    b = f - f.mean()
    x = np.zeros_like(b); r = b.copy(); z = cell.Kinv(r); z -= z.mean(); p = z.copy(); rz = r @ z
    for it in range(1, 5001):
        Ap = cell.Kc(p); al = rz / (p @ Ap); x += al * p; r -= al * Ap
        gk = e + cell.A @ x
        if abs(gk @ (cell.St * gk) - sH) <= tol:
            return it
        z = cell.Kinv(r); z -= z.mean(); rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
    raise RuntimeError("pcg did not converge")


# ==========================================================================
# 4. resource counts
# ==========================================================================
def uv_counts(N, F):
    """Toffoli count, controlled rotations and qubits of one application of U_V."""
    n = int(np.log2(N))
    r = IsometryCircuit(n=n, F=F).resources()
    return dict(toffoli=r["toffoli_total"], cordic_rot=2 * F + 1 + 3,
                qft_rot=2 * 2 * n * (n - 1) // 2, qubits=r["wires"])


def readout_resources(name, N, rho, eps, sigma_ratio=1.0):
    """Sec. 4.3 at error eps relative to the Voigt bound (divided by sigma_ratio for an error
    relative to sigma^H): eps_p = eps_eff / 3, F from 6 N 2^-F, M = ceil(3 pi alpha / (eps_eff sbar)),
    one U_V and one controlled R_chi per query."""
    eps_eff = eps / sigma_ratio
    eps_p = eps_eff / 3
    d = 2 * qsvt_degree(rho, eps_p)
    aq = 1 + eps_p
    vf = chi(name, 16).mean()
    sbar = vf + (1 - vf) / rho
    F = int(np.ceil(np.log2(DELTA_CONST * N * 3 * d * aq / eps_eff)))
    M = int(np.ceil(3 * np.pi * aq / (eps_eff * sbar)))
    queries = (2 * M - 1) * d
    uv = uv_counts(N, F)
    orc = check_oracle(name)
    per_q_tof = uv["toffoli"] + orc["toffoli"]
    per_q_rot = (uv["cordic_rot"] + uv["qft_rot"]) + 2          # + the two Ry of the Sigma^{1/2} LCU
    return dict(name=name, N=N, rho=rho, eps=eps, sigma_ratio=sigma_ratio, F=F, d=d,
                alpha_q=aq, sbar=sbar, M=M, queries=queries, toffoli_per_query=per_q_tof,
                rot_per_query=per_q_rot, oracle_toffoli=orc["toffoli"],
                toffoli_total=float(queries) * per_q_tof, qubits=uv["qubits"] + 5)


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
def exp_mesh(Ns=(16, 32, 64, 128, 256, 512, 1024), rho=10.0, eps_p=1e-6, eps=1e-2):
    rows = []
    for g in GEOMETRIES:
        for N in Ns:
            t0 = time.time()
            c = Cell(g, N, rho)
            lo, hi = extreme_eigs(c)
            lk = lambda_min_Kc(c)
            sFE = c.fe_sigma()[0, 0]
            s_est, A, d, e_t = poly_readout(c, eps_p)
            err = abs(s_est - sFE) / c.sbar                      # error of g^T r_t(B) g, in units of sbar
            assert A >= 0
            # FFT-PCG from v = 0 to the same error in sigma^H (the energy error of the cell solution)
            it = pcg_sigma_iterations(c, max(err, 1e-12) * c.sbar)
            res = readout_resources(g, N, rho, eps)
            lamK = 2 * (1 - np.cos(2 * np.pi / N))
            r = dict(geometry=g, N=N, kappa_M=1 / lo, lam_max=hi, kappa_Kc=ALPHA / lk,
                     kappa_P=ALPHA * rho / lamK, poly_err=err, d=d, pcg_it=it,
                     toffoli_per_query=res["toffoli_per_query"], F=res["F"],
                     queries=res["queries"], seconds=time.time() - t0)
            rows.append(r)
            print(f"mesh {g:8s} N={N:5d} kappa(M)={r['kappa_M']:.10f} lmax={hi:.12f} "
                  f"kappa(Kc)={r['kappa_Kc']:.4g} kappa_P={r['kappa_P']:.4g} err={err:.3e} d={d} "
                  f"pcg={it} F={res['F']} tof/q={r['toffoli_per_query']:,} [{r['seconds']:.0f} s]",
                  flush=True)
    RESULTS["mesh"] = rows
    # ---- assertions: every claim of Sec. 5.2
    for r in rows:
        assert abs(r["kappa_M"] - rho) < 1e-6 * rho and abs(r["lam_max"] - 1) < 1e-8, r   # kappa = rho
        assert r["poly_err"] <= eps_p and r["d"] == 46, r                                  # flat at d = 46
        assert r["kappa_P"] > r["kappa_Kc"], r
    for g in GEOMETRIES:
        k = [r["kappa_Kc"] for r in rows if r["geometry"] == g]
        ratios = np.array(k[1:]) / np.array(k[:-1])
        assert np.all(ratios > 3.5) and abs(ratios[-1] - 4) < 0.02, (g, ratios)          # Theta(N^2)
    its = [r["pcg_it"] for r in rows]
    tof = {r["N"]: r["toffoli_per_query"] for r in rows if r["geometry"] == "square"}
    print(f"mesh: PCG iterations {min(its)}-{max(its)}; Toffolis per query "
          f"{tof[Ns[0]]:,} (N={Ns[0]}) to {tof[Ns[-1]]:,} (N={Ns[-1]})")

    plt = plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35))
    ndof = lambda g: [r["N"] ** 2 for r in rows if r["geometry"] == g]
    for g in GEOMETRIES:
        ls, col, mk = STYLE[g]
        ax[0].loglog(ndof(g), [r["kappa_Kc"] for r in rows if r["geometry"] == g], ls, color=col,
                     marker=mk, ms=3, lw=1, label=g)
        ax[1].loglog(ndof(g), [r["poly_err"] for r in rows if r["geometry"] == g], ls, color=col,
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
    ax[1].set_ylim(1e-8, 1e-5); ax[1].set_xlabel(r"$N_{\mathrm{dof}}$")
    ax[1].set_ylabel(r"$|\hat\sigma - \sigma^H/\sigma_{\max}|/\bar\sigma$")
    ax[1].set_title(f"(b) error of $f$, $d = {rows[0]['d']}$", fontsize=9)
    ax[2].semilogx(ndof("square"), [tof[N] / 1e3 for N in Ns], "-o", color="k", ms=3, lw=1)
    ax[2].set_xlabel(r"$N_{\mathrm{dof}}$"); ax[2].set_ylabel(r"Toffoli gates per query ($10^3$)")
    ax[2].set_title(r"(c) cost of one query to $\mathcal{U}_V$", fontsize=9)
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
            S_fe = c.fe_sigma()
            S_qf, _, _ = c.qf_sigma()
            vf = c.vf
            r = dict(geometry=g, rho=rho, vf=vf, oracle_toffoli=orc["toffoli"], kappa_M=1 / lo,
                     s11=rho * S_fe[0, 0], s22=rho * S_fe[1, 1], s12=rho * S_fe[0, 1],
                     qf_vs_fe=float(np.abs(S_qf - S_fe).max() / np.abs(S_fe).max()),
                     ratio11=c.sbar / S_fe[0, 0], ratio22=c.sbar / S_fe[1, 1])
            if g == "laminate":                       # exact: sigma_11 = Reuss, sigma_22 = Voigt
                reuss = 1 / (vf / rho + 1 - vf)
                voigt = vf * rho + 1 - vf
                r["exact_err"] = max(abs(r["s11"] - reuss) / reuss, abs(r["s22"] - voigt) / voigt)
            rows.append(r)
            print(f"geometry {g:8s} rho={rho:g} vf={vf:.4f} R_chi Toffolis={orc['toffoli']} "
                  f"kappa(M)={r['kappa_M']:.8f} s11={r['s11']:.6f} s22={r['s22']:.6f} "
                  f"s12={r['s12']:.1e} QF-FE={r['qf_vs_fe']:.1e} V/H11={r['ratio11']:.4g}"
                  + (f" exact err={r['exact_err']:.1e}" if g == "laminate" else ""), flush=True)
    RESULTS["geometry"] = rows
    for r in rows:
        assert abs(r["kappa_M"] - r["rho"]) < 1e-5 * r["rho"], r          # kappa = rho, every geometry
        assert r["qf_vs_fe"] < 1e-8, r                                     # quadratic form = FE
        assert abs(r["s12"]) < 1e-9 * r["s11"], r                          # no off-diagonal entry
        if r["geometry"] != "laminate":
            assert abs(r["s11"] - r["s22"]) < 1e-9 * r["s11"], r           # isotropic
        else:
            assert r["exact_err"] < 1e-10, r                               # exact Reuss / Voigt
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
    names = {"square": "square", "cross": "cross", "laminate": "laminate"}
    trows = []
    for r in rows:
        trows.append([names[r["geometry"]] if r["rho"] == rhos[0] else "",
                      f"${r['vf']:.3g}$" if r["rho"] == rhos[0] else "",
                      f"${r['oracle_toffoli']}$" if r["rho"] == rhos[0] else "",
                      p10(r["rho"]), f"${r['kappa_M'] / r['rho']:.6f}$",
                      sci(r["s11"], 4), sci(r["s22"], 4), sci(r["qf_vs_fe"], 1),
                      sci(r["ratio11"], 3)])
    write_table("geometry", ["geometry", "$v_f$", "$R_\\chi$ Toffoli", "$\\rho$",
                             "$\\kappa(\\tilde{\\mat{M}})/\\rho$", "$\\sigma^H_{11}/\\sigma_{\\min}$",
                             "$\\sigma^H_{22}/\\sigma_{\\min}$", "QF vs FE",
                             "$\\sigma^{\\mathrm{V}}/\\sigma^H_{11}$"], trows, "lrrrrrrrr")


# ==========================================================================
# 8. Experiment 3: contrast
# ==========================================================================
def exp_contrast(N_fe=256, N_q=1024, eps=1e-2):
    rhos = np.unique(np.concatenate([np.logspace(0.25, 4, 16), [10.0, 1e2, 1e3, 1e4]]))
    ratio = {g: [] for g in GEOMETRIES}
    for g in GEOMETRIES:
        for rho in rhos:
            c = Cell(g, N_fe, rho)
            ratio[g].append(c.sbar / c.fe_sigma()[0, 0])
        print(f"contrast {g}: sigma^V/sigma^H_11 = " +
              ", ".join(f"{r:.3g}" for r in ratio[g]), flush=True)
    res = {}
    for g in GEOMETRIES:
        for rho, rt in zip(rhos, ratio[g]):
            if rho in (10.0, 1e2, 1e3, 1e4):
                res[g, rho, "V"] = readout_resources(g, N_q, rho, eps)
                res[g, rho, "H"] = readout_resources(g, N_q, rho, eps, sigma_ratio=rt)
    RESULTS["contrast"] = dict(rhos=rhos.tolist(), ratio=ratio,
                               resources={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()})
    for g in GEOMETRIES:                                    # Sec. 4.3 values; the cross stays bounded
        r = dict(zip(rhos, ratio[g]))
        if g == "square":
            assert round(r[10.0], 2) == 2.10 and round(r[1e4], -1) == 1440, r
    assert max(ratio["cross"]) < 2.0, ratio["cross"]
    for k, v in res.items():
        print(f"contrast resources {k}: d={v['d']} alpha_q={v['alpha_q']:.4g} M={v['M']:.3e} "
              f"queries={v['queries']:.3e} Toffolis={v['toffoli_total']:.3e} F={v['F']}")
    trows = []
    for rho in (10.0, 1e2, 1e3, 1e4):
        v = res["square", rho, "V"]
        trows.append([p10(rho), intx(v["d"]), sci(v["M"], 2),
                      sci(v["queries"], 2), sci(v["toffoli_total"], 2),
                      sci(res["square", rho, "H"]["toffoli_total"], 2),
                      sci(res["cross", rho, "H"]["toffoli_total"], 2)])
    write_table("contrast", ["$\\rho$", "$d$", "$M$", "queries", "Toffoli",
                             "square, $\\sigma^H$", "cross, $\\sigma^H$"], trows, "rrrrrrr")

    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for g in GEOMETRIES:
        ls, col, mk = STYLE[g]
        lab = g + (r" ($\sigma^H_{11}$)" if g == "laminate" else "")
        ax[0].loglog(rhos, ratio[g], ls, color=col, lw=1.2, label=lab)
        rr = [r for r in (10.0, 1e2, 1e3, 1e4)]
        ax[1].loglog(rr, [res[g, r, "H"]["toffoli_total"] for r in rr], ls, color=col, marker=mk,
                     ms=3, lw=1.2, label=lab)
    ax[1].loglog(rr, [res["square", r, "V"]["toffoli_total"] for r in rr], "-", color="0.8", lw=2.5,
                 zorder=0, label="relative to Voigt")
    ax[0].set_xlabel(r"contrast $\rho$"); ax[0].set_ylabel(r"$\sigma^{\mathrm{V}}/\sigma^H$")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title(r"(a) cancellation factor $\sigma^{\mathrm{V}}/\sigma^H$", fontsize=9)
    ax[1].set_xlabel(r"contrast $\rho$"); ax[1].set_ylabel("Toffoli gates per readout")
    ax[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    ax[1].set_title(r"(b) readout at $1\%$ relative error, $N = 1024$", fontsize=9)
    fig.tight_layout(); fig.savefig("figs/contrast.pdf"); print("figs/contrast.pdf")


# ==========================================================================
# 9. Experiment 4: accuracy and resources
# ==========================================================================
def exp_accuracy(N=256, rho=10.0, runs=2000, seed=1, N_res=1024):
    c = Cell("square", N, rho)
    target = c.fe_sigma()[0, 0]                               # sigma^H / sigma_max
    epss = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    rows = []
    rng = np.random.default_rng(seed)
    for eps in epss:
        eps_p = eps / 3
        s_poly, A, d, e_t = poly_readout(c, eps_p)
        aq = 1 + eps_p
        M = int(np.ceil(3 * np.pi * aq / (eps * c.sbar)))
        Ahat = np.sqrt(ae_samples(A ** 2, M, runs, rng))
        est = lambda Ah: (1 + e_t) * Ah - e_t * c.sbar
        err = np.abs(est(Ahat) - target) / c.sbar
        med = np.median(Ahat.reshape(-1, 5), axis=1)
        r = dict(eps=eps, d=d, alpha_q=aq, M=M, queries=(2 * M - 1) * d,
                 err_median=float(np.median(err)), err_q=float(np.quantile(err, 8 / np.pi ** 2)),
                 success=float(np.mean(err <= eps)),
                 success_median5=float(np.mean(np.abs(est(med) - target) / c.sbar <= eps)),
                 poly_part=float(abs(s_poly - target) / c.sbar / eps))
        rows.append(r)
        print(f"accuracy eps={eps:g}: d={d} alpha={aq:.4g} M={M} queries={r['queries']:.3e} "
              f"median err/eps={r['err_median'] / eps:.3f} q(8/pi^2)/eps={r['err_q'] / eps:.3f} "
              f"success={r['success']:.4f} median5={r['success_median5']:.3f} "
              f"poly/eps={r['poly_part']:.2e}", flush=True)
    q8 = 1 - 8 / np.pi ** 2                                    # per-run failure bound (Brassard et al.)
    med5 = 1 - sum(__import__("math").comb(5, k) * q8 ** k * (1 - q8) ** (5 - k) for k in (3, 4, 5))
    RESULTS["median5_bound"] = med5
    for r in rows:
        assert r["success"] >= 8 / np.pi ** 2 and r["success_median5"] >= med5, r
        assert r["poly_part"] <= 1 / 3 and r["err_q"] <= r["eps"], r     # guaranteed quantile
    q = np.array([r["queries"] for r in rows]); e = np.array(epss)
    slope = np.polyfit(np.log(e), np.log(q), 1)[0]
    print(f"accuracy: queries ~ eps^{slope:.3f}")
    assert -1.25 < slope < -1.0, slope                         # 1/eps up to logarithms
    res = [readout_resources("square", N_res, rho, eps) for eps in (1e-2, 1e-3, 1e-4)]
    RESULTS["accuracy"] = dict(samples=rows, slope=slope, resources=res)
    trows = []
    for v in res:
        trows.append([p10(v["eps"]), f"${v['F']}$", f"${v['d']}$",
                      sci(v["M"], 2), sci(v["queries"], 2),
                      f"${v['toffoli_per_query']:,}$".replace(",", "{,}"),
                      f"${v['rot_per_query']}$", sci(v["toffoli_total"], 2),
                      f"${v['qubits']:,}$".replace(",", "{,}")])
    write_table("resources", ["$\\epsilon$", "$F$", "$d$", "$M$", "queries",
                              "Toffoli/query", "rotations/query", "Toffoli", "qubits"],
                trows, "rrrrrrrrr")
    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
    ax[0].loglog(e, [r["err_median"] for r in rows], "-o", color="k", ms=3, lw=1, label="median")
    ax[0].loglog(e, [r["err_q"] for r in rows], "--s", color="0.4", ms=3, lw=1,
                 label=r"$8/\pi^2$ quantile")
    ax[0].loglog(e, e, ":", color="0.6", lw=1.2, label=r"bound $\epsilon$")
    ax[0].set_xlabel(r"target $\epsilon$"); ax[0].set_ylabel(r"$|\hat\sigma - \sigma^H/\sigma_{\max}|/\bar\sigma$")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title(f"(a) readout error, {runs} runs", fontsize=9)
    ax[1].loglog(e, q, "-o", color="k", ms=3, lw=1)
    ax[1].set_xlabel(r"target $\epsilon$"); ax[1].set_ylabel(r"queries to $\mathcal{U}_V$")
    ax[1].set_title(f"(b) queries $\\propto \\epsilon^{{{slope:.2f}}}$", fontsize=9)
    fig.tight_layout(); fig.savefig("figs/accuracy.pdf"); print("figs/accuracy.pdf")


# ==========================================================================
# 10. Experiment 5: comparison with FFT-PCG
# ==========================================================================
def toffoli_fit(N, F):
    return 20.5 * F ** 2 + 64 * F + 41 + 4 * np.log2(N)


def quantum_toffolis(N, rho=10.0, eps=1e-2, name="square"):
    """Total Toffolis of one readout at mesh N, with Eq. (toffoli) for U_V beyond N = 1024."""
    d = 2 * qsvt_degree(rho, eps / 3)
    aq = 1 + eps / 3
    vf = chi(name, 16).mean()
    sbar = vf + (1 - vf) / rho
    F = int(np.ceil(np.log2(DELTA_CONST * N * 3 * d * aq / eps)))
    M = int(np.ceil(3 * np.pi * aq / (eps * sbar)))
    return (2 * M - 1) * d * (toffoli_fit(N, F) + check_oracle(name)["toffoli"])


def exp_crossover(Ns=(64, 128, 256, 512, 1024), rho=10.0, eps=1e-2, t_T=(170e-6, 1e-6)):
    rows = []
    for N in Ns:
        c = Cell("square", N, rho)
        e = c.ehat(1)
        f = -(c.A.T @ (c.St * e))
        vref, _ = pcg(c.Kc, c.Kinv, f, 1e-13)
        sH = (e + c.A @ vref) @ (c.St * (e + c.A @ vref))
        best = []
        for rep in range(3):                                  # wall time to |sigma - sigma^H| <= eps sbar
            t0 = time.time()
            x = np.zeros(c.n); r = f - f.mean(); z = c.Kinv(r); p = z.copy(); rz = r @ z; it = 0
            while True:
                it += 1
                Ap = c.Kc(p); al = rz / (p @ Ap); x += al * p; r -= al * Ap
                g = e + c.A @ x
                if abs(g @ (c.St * g) - sH) <= eps * c.sbar:
                    break
                z = c.Kinv(r); rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
            best.append(time.time() - t0)
        rows.append(dict(N=N, iterations=it, seconds=min(best)))
        print(f"crossover classical N={N}: {it} iterations, {min(best):.4f} s", flush=True)
    Nn = np.array([r["N"] for r in rows], float); ts = np.array([r["seconds"] for r in rows])
    work = Nn ** 2 * np.log2(Nn ** 2)
    cfit = float(np.exp(np.mean(np.log(ts[-3:] / work[-3:]))))   # t = c N_dof log2 N_dof, last three
    Nx = 2.0 ** np.arange(6, 26.001, 0.02)                    # fine grid in log2 N
    tq = np.array([quantum_toffolis(N, rho, eps) for N in Nx])
    tc = cfit * Nx ** 2 * np.log2(Nx ** 2)
    cross = {}
    for tt in t_T:
        idx = np.where(tc >= tq * tt)[0]
        cross[tt] = float(Nx[idx[0]] ** 2) if len(idx) else None
        print(f"crossover at t_T = {tt * 1e6:g} us: N_dof = {cross[tt]:.3g}" if cross[tt]
              else f"no crossover below N_dof = {Nx[-1] ** 2:.3g} at t_T = {tt * 1e6:g} us")
    assert 1e10 < cross[170e-6] < 1e12 and 1e8 < cross[1e-6] < 1e10, cross       # Sec. 5.5
    tq1024 = quantum_toffolis(1024, rho, eps)
    RESULTS["crossover"] = dict(classical=rows, c_fit=cfit, crossover_ndof=cross,
                                toffoli_1024=tq1024)
    print(f"crossover: quantum readout at N = 1024: {tq1024:.3e} Toffolis = "
          f"{tq1024 * 170e-6 / 86400:.3g} days at 170 us; classical {ts[-1]:.3g} s")
    plt = plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.8))
    ax.loglog(Nn ** 2, ts, "o", color="k", ms=4, label="FFT-PCG, measured")
    ax.loglog(Nx ** 2, tc, "-", color="k", lw=1, label=r"FFT-PCG, $\propto N_{\mathrm{dof}}\log N_{\mathrm{dof}}$")
    for tt, ls in zip(t_T, ("--", ":")):
        ax.loglog(Nx ** 2, tq * tt, ls, color="0.35", lw=1.2,
                  label=f"quantum readout, {tt * 1e6:g} $\\mu$s per Toffoli")
        if cross[tt]:
            ax.plot(cross[tt], cfit * cross[tt] * np.log2(cross[tt]), "s", color="0.35", ms=4)
    ax.set_xlabel(r"$N_{\mathrm{dof}}$"); ax.set_ylabel("wall time (s)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig("figs/crossover.pdf"); print("figs/crossover.pdf")


# ==========================================================================
EXPERIMENTS = {"mesh": exp_mesh, "geometry": exp_geometry, "contrast": exp_contrast,
               "accuracy": exp_accuracy, "crossover": exp_crossover}

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