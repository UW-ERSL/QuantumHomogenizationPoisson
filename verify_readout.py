"""verify_readout -- readout of the effective conductivity (Sec. 4).

    python verify_readout.py           # about 3 minutes

Three checks, one per proposition of Sec. 4. Square inclusion, v_f = 1/4,
sigma_max = 1.

1. Quadratic form (Prop. qf). sigma^H / sigma_max = sbar delta_ab
   - psi_a^T Mtilde^{-1} psi_b, psi_a = V^T Sigmatilde ehat_a, against the
   direct finite element cell solution (ebar + G v)^T W Sigma (ebar + G v),
   N = 8 ... 64, rho = 1.5 ... 1e4; with the bounds of Sec. 4.1 and the
   polynomial error on the correction term.
2. Readout circuit (Prop. readout-circuit). Statevector emulation of
   Z_ab = (U_psi,a^(2))^dag U_q U_psi,b^(1) with U_V from the gate-level
   circuit, U_Sigma as its two-term LCU, flag qubits marking the local
   states outside the input subspace, and U_q a unitary whose block on the
   input subspace is q_t(Mtilde_c)/C and whose action elsewhere is a random
   unitary. The QSVT phase sequence of U_q is not constructed; its block is
   Prop. qsvt-inverse. Also run with one shared ancilla and no flags.
3. Readout cost (Sec. 4.3, eq:readout-error, eq:readout-queries). Amplitude
   estimation on the all-zero probability Z_aa^2 of the readout circuit
   (Z_aa >= 0, so no controlled copy), sampled from its exact output
   distribution (Brassard et al., Thm. 12), with eps_p = eps/3, F chosen so
   that 6 N 2^-F meets the encoding-error condition, and
   M = ceil(3 pi C / (eps sbar)): with probability >= 8/pi^2 the angle error
   is at most pi/M, hence |A_hat - A| <= pi/M and C |A_hat - A| <= eps sbar/3.
"""
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from verify_square import assemble, square_chi
from verify_inverse import k_function, q_poly
from quantuminversionpoisson.isometry_circuit import IsometryCircuit

RHOS = (1.5, 3.75, 10.0, 100.0, 1e4)
DELTA_CONST = 6          # delta <= 6 N 2^-F (verify_full_unitary.py)


# ---- polynomial constants ----------------------------------------------------
def poly_constants(rho, eps_p):
    """q_t, its degree d, and C = max_[-1,1] |q_t| (grid of 2e6 points, q_t odd)."""
    q, d = q_poly(rho, eps_p)
    x = np.linspace(1e-9, 1.0, 2_000_001)
    return q, d, float(np.abs(q(x)).max())


# ---- 1. quadratic form ----------------------------------------------------------
def e_hat(N, a):
    """W^{1/2} ebar_a: sqrt(w) = 1/(2N) at rows 8 e + 2 g + (a - 1)."""
    v = np.zeros(8 * N * N)
    v[(a - 1)::2] = 1 / (2 * N)
    return v


def fe_sigmaH(N, A, St):
    """Effective conductivity from the cell problem, independently of Prop. qf."""
    Kc = (A.T @ sp.diags(St) @ A).tocsr()
    keep = np.arange(1, N * N)
    S = np.zeros((2, 2))
    grads = []
    for a in (1, 2):
        e = e_hat(N, a)
        f = -(A.T @ (St * e))
        v = np.zeros(N * N)
        v[keep] = spla.spsolve(Kc[keep][:, keep].tocsc(), f[keep])
        grads.append(e + A @ v)
    for a in range(2):
        for b in range(2):
            S[a, b] = grads[a] @ (St * grads[b])
    return S


def qf_case(N, rho, eps_p):
    A = assemble(N).tocsr()
    chi = square_chi(N)
    vf = chi.mean()
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    sbar = vf + (1 - vf) / rho
    Kih = k_function(lambda e: e ** -0.5, N)
    VT = lambda y: Kih(A.T @ y)
    Kc = (A.T @ sp.diags(St) @ A).tocsr()
    M = Kih(Kc @ Kih(np.eye(N * N)))
    lam, U = np.linalg.eigh((M + M.T) / 2)
    lt = np.where(np.abs(lam) > 1e-9, lam, sbar)                # spectrum of Mtilde
    q, d, C = poly_constants(rho, eps_p)
    psi = [VT(St * e_hat(N, a)) for a in (1, 2)]
    w = [U.T @ p for p in psi]
    c = np.array([[w[a] @ (w[b] / lt) for b in range(2)] for a in range(2)])
    cq = np.array([[w[a] @ (q(lt) * w[b]) for b in range(2)] for a in range(2)])
    S_qf = sbar * np.eye(2) - c
    S_fe = fe_sigmaH(N, A, St)
    return dict(
        N=N, rho=rho, sbar=sbar, sigmaH=float(S_fe[0, 0]),
        qf_vs_fe=float(np.abs(S_qf - S_fe).max() / np.abs(S_fe).max()),
        VT_ehat=float(max(np.linalg.norm(VT(e_hat(N, a))) for a in (1, 2))),
        psi_mean=float(max(abs(p.sum()) for p in psi)),
        psi_norm2_over_bound=float(max(p @ p for p in psi) / ((1 - 1 / rho) ** 2 * vf)),
        c_over_bound=float(c[0, 0] / (sbar - 1 / rho)),
        offdiag=float(abs(S_fe[0, 1])), aniso=float(abs(S_fe[1, 1] - S_fe[0, 0])),
        poly_err_over_eps=float(np.abs(cq - c)[0, 0] / c[0, 0] / eps_p),
        voigt_over_sigmaH=float(sbar / S_fe[0, 0]), d=d, C=C)


# ---- 2. readout circuit (statevector) -------------------------------------------
class Readout:
    """State axes: (position N^2, local 16, qa, s1, s2, f1, f2)."""

    def __init__(self, N, rho, F, eps_p, seed=0):
        self.N, self.rho = N, rho
        chi = square_chi(N)
        self.D = 2 * chi - 1                                     # -R_chi on the element index
        vf = chi.mean()
        b0, b1 = (1 + 1 / rho) / 2, (1 - 1 / rho) / 2
        self.R = np.array([[np.sqrt(b0), -np.sqrt(b1)], [np.sqrt(b1), np.sqrt(b0)]])
        self.sbar = vf + (1 - vf) / rho
        circ = IsometryCircuit(int(np.log2(N)), F=F)
        self.Q, self.chk = circ.simulate()                       # (N^2, 16, 16)
        # input-subspace block X of U_V, Mtilde_c and q_t(Mtilde_c) / C from the gates
        Fm = np.kron(*(np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
                       / np.sqrt(N),) * 2)
        BD = np.zeros((16 * N * N, N * N), complex)
        for k in range(N * N):
            BD[16 * k:16 * k + 16, k] = self.Q[k, :, 0]
        self.X = np.kron(Fm.conj().T, np.eye(16)) @ BD @ Fm
        Se = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 16)
        self.Se = Se
        Mc = self.X.conj().T @ (Se[:, None] * self.X)
        lc, Uc = np.linalg.eigh((Mc + Mc.conj().T) / 2)
        q, self.d, self.C = poly_constants(rho, eps_p)
        P = (Uc * (q(lc) / self.C)) @ Uc.conj().T
        self.P = P
        self.Uq = self._unitary_q(P, np.random.default_rng(seed))

    # -- U_q: block P on (local = 0, qa = 0), random unitary elsewhere ----------
    def _unitary_q(self, P, rng):
        n = self.N ** 2
        dim = n * 16 * 2                                         # (position, local, qa)
        idx = np.arange(dim).reshape(n, 16, 2)
        i0, i1 = idx[:, 0, 0], idx[:, 0, 1]
        lp, Up = np.linalg.eigh((P + P.conj().T) / 2)
        Sq = (Up * np.sqrt(np.clip(1 - lp ** 2, 0, None))) @ Up.conj().T
        Dm = np.eye(dim, dtype=complex)
        Dm[np.ix_(i0, i0)], Dm[np.ix_(i0, i1)] = P, Sq
        Dm[np.ix_(i1, i0)], Dm[np.ix_(i1, i1)] = Sq, -P
        comp = np.setdiff1d(np.arange(dim), i0)

        def rand_on_comp():
            G = rng.standard_normal((comp.size,) * 2) + 1j * rng.standard_normal((comp.size,) * 2)
            Qr, Rr = np.linalg.qr(G)
            Qr *= np.diag(Rr) / np.abs(np.diag(Rr))
            Rf = np.eye(dim, dtype=complex)
            Rf[np.ix_(comp, comp)] = Qr
            return Rf
        return rand_on_comp() @ Dm @ rand_on_comp()

    # -- gates on the state ---------------------------------------------------------
    def UV(self, s, dag=False):
        N = self.N
        x = s.reshape(N, N, 16, -1)
        x = np.fft.fft2(x, axes=(0, 1), norm="ortho").reshape(N * N, 16, -1)
        Qk = self.Q.conj().transpose(0, 2, 1) if dag else self.Q
        x = np.einsum("kij,kjr->kir", Qk, x).reshape(N, N, 16, -1)
        return np.fft.ifft2(x, axes=(0, 1), norm="ortho").reshape(s.shape)

    def USigma(self, s, anc, dag=False):
        """PREP^T . (|0><0| x I + |1><1| x (-R_chi)) . PREP on ancilla axis anc (2 = qa ... )."""
        s = np.moveaxis(s, anc, 2)
        R = self.R
        s = np.einsum("ij,pljr->plir", R, s.reshape(s.shape[0], 16, 2, -1))
        s[:, :, 1] *= self.D[:, None, None]
        s = np.einsum("ji,pljr->plir", R, s)                     # R^T
        return np.moveaxis(s.reshape(-1, 16, *([2] * 5)), 2, anc)

    def flag(self, s, f):
        s = s.copy()
        sl = [slice(None)] * 7
        sl[1] = slice(1, 16)
        a, b = sl.copy(), sl.copy()
        a[f], b[f] = 0, 1
        s[tuple(a)], s[tuple(b)] = s[tuple(b)].copy(), s[tuple(a)].copy()
        return s

    def apply_Uq(self, s):
        n = self.N ** 2
        x = np.moveaxis(s, 2, 2).reshape(n * 16 * 2, -1)        # (pos, local, qa) first
        return (self.Uq @ x).reshape(s.shape)

    def prep(self, a):
        """ehat_a on (position, local) for a in (1, 2); 'p' gives (ehat_1 + ehat_2)/sqrt 2."""
        loc = np.zeros(16)
        if a == "p":
            loc[:8] = 1 / np.sqrt(8)
        else:
            loc[a - 1:8:2] = 0.5
        return np.outer(np.full(self.N ** 2, 1 / self.N), loc)

    def amplitude(self, a, b, distinct=True):
        n = self.N ** 2
        s = np.zeros((n, 16, 2, 2, 2, 2, 2), complex)
        s[:, :, 0, 0, 0, 0, 0] = self.prep(b)
        s1, s2 = 3, (4 if distinct else 3)
        s = self.UV(self.USigma(s, s1), dag=True)                # U_psi,b^(1)
        if distinct:
            s = self.flag(s, 5)
        s = self.apply_Uq(s)
        if distinct:
            s = self.flag(s, 6)
        s = self.USigma(self.UV(s), s2, dag=True)               # (U_psi,a^(2))^dag
        return complex(np.vdot(self.prep(a), s[:, :, 0, 0, 0, 0, 0]))

    def block_reference(self, a, b):
        """psi_a^T q_t(Mtilde_c) psi_b / C from the blocks of the same gates."""
        psi = lambda a: self.X.conj().T @ (self.Se * self.prep(a).ravel())
        return complex(psi(a).conj() @ self.P @ psi(b))


def circuit_case(N, rho, F, eps_p):
    t = time.time()
    r = Readout(N, rho, F, eps_p)
    # exact amplitude from the assembled operators
    A = assemble(N).tocsr()
    chi = square_chi(N)
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    Kih = k_function(lambda e: e ** -0.5, N)
    M = Kih((A.T @ sp.diags(St) @ A) @ Kih(np.eye(N * N)))
    lam, U = np.linalg.eigh((M + M.T) / 2)
    lt = np.where(np.abs(lam) > 1e-9, lam, r.sbar)
    q, _, _ = poly_constants(rho, eps_p)
    psi = {a: Kih(A.T @ (St * e_hat(N, a))) for a in (1, 2)}
    psi["p"] = (psi[1] + psi[2]) / np.sqrt(2)
    ex = lambda a, b: (U.T @ psi[a]) @ (q(lt) * (U.T @ psi[b])) / r.C
    out = dict(N=N, rho=rho, F=F, d=r.d, C=r.C,
               delta_bound=DELTA_CONST * N * 2.0 ** -F,
               clean=r.chk["work_bits_left_set"] == 0 and r.chk["inputs_restored"])
    for a, b in ((1, 1), (2, 2), (1, 2), ("p", "p")):
        z = r.amplitude(a, b)
        out[f"Z{a}{b}"] = z.real
        out[f"Z{a}{b}_imag"] = abs(z.imag)
        out[f"Z{a}{b}_vs_blocks"] = abs(z - r.block_reference(a, b))
        out[f"Z{a}{b}_vs_exact"] = abs(z.real - ex(a, b))
    out["Z11_shared"] = r.amplitude(1, 1, distinct=False).real
    out["Z11_exact"] = ex(1, 1)
    out["polarization"] = abs(out["Zpp"] - (out["Z11"] + out["Z22"]) / 2 - out["Z12"])
    out["seconds"] = time.time() - t
    return out


# ---- 3. amplitude estimation ------------------------------------------------------
def ae_samples(p, M, runs, rng):
    """Brassard et al., Thm. 12: estimates of p from QPE with M points, sampled exactly."""
    om = np.arcsin(np.sqrt(p)) / np.pi
    y = np.arange(M)

    def qpe(w):
        dl = w - y / M
        s = np.sin(np.pi * dl)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(np.abs(s) < 1e-15, 1.0, np.sin(M * np.pi * dl) ** 2 / (M ** 2 * s ** 2))
        return P
    P = 0.5 * (qpe(om) + qpe(1 - om))
    P /= P.sum()
    return np.sin(np.pi * rng.choice(M, size=runs, p=P) / M) ** 2


def cost_case(N, rho, eps, runs=2000, seed=1):
    eps_p = eps / 3
    q, d, C = poly_constants(rho, eps_p)
    chi = square_chi(N)
    sbar = chi.mean() + (1 - chi.mean()) / rho
    F = int(np.ceil(np.log2(DELTA_CONST * N * 6 * C * (d + 1) / (eps * sbar))))
    r = circuit_case(N, rho, F, eps_p)
    A = r["Z11"]
    M = int(np.ceil(3 * np.pi * C / (eps * sbar)))
    rng = np.random.default_rng(seed)
    Ahat = np.sqrt(ae_samples(A ** 2, M, runs, rng))           # A >= 0
    qf = qf_case(N, rho, eps_p)
    target = qf["sigmaH"]
    err = np.abs(sbar - C * Ahat - target)
    med = np.median(Ahat.reshape(-1, 5), axis=1)                # median of 5 runs
    err_med = np.abs(sbar - C * med - target)
    return dict(N=N, rho=rho, eps=eps, F=F, d=d, C=C, M=M, queries=(2 * M - 1) * d,
                ae_bound_frac=float(np.mean(np.abs(Ahat - A) <= np.pi / M)),
                success_frac=float(np.mean(err <= eps * sbar)),
                success_frac_median5=float(np.mean(err_med <= eps * sbar)),
                max_err_over_bound=float(err.max() / (eps * sbar)),
                deterministic_err_over_bound=float(abs(sbar - C * A - target) / (eps * sbar)),
                runs=runs, seed=seed)


def assert_claims(out):
    """Every number and inequality that Sec. 4 of the paper states."""
    for r in out["qf"]:
        assert r["qf_vs_fe"] < 1e-9, r                           # Prop. qf
        assert r["VT_ehat"] < 1e-15 and r["psi_mean"] < 1e-13, r # V^T ehat = 0, psi orth. constants
        assert r["psi_norm2_over_bound"] <= 1, r                 # ||psi||^2 <= (1-1/rho)^2 v_f
        assert 0 <= r["c_over_bound"] <= 1, r                    # 0 <= c <= sbar - 1/rho
        assert r["poly_err_over_eps"] <= 1, r                    # |psi^T q psi - c| <= eps_p c
        assert max(r["offdiag"], r["aniso"]) < 1e-15, r          # sigma^H isotropic
    v = {(r["N"], r["rho"]): r["voigt_over_sigmaH"] for r in out["qf"]}
    assert round(v[64, 10.0], 2) == 2.10 and round(v[64, 1e4], -1) == 1440, v
    for r in out["circuit"]:
        assert r["clean"]
        assert r["Z11"] > 0                                      # nonnegative for a = b
        for k in ("11", "22", "12", "pp"):
            assert r[f"Z{k}_vs_blocks"] < 1e-15, (k, r)          # Prop. readout-circuit
            assert r[f"Z{k}_vs_exact"] < 2 * (r["d"] + 1) * r["delta_bound"], (k, r)
        assert r["polarization"] < 1e-15 and abs(r["Z12"]) < 1e-9
        assert abs(r["Z11_shared"] - r["Z11"]) > 1e-3 * r["Z11"] # shared ancilla: wrong
    c = {(r["N"], r["eps"]): r for r in out["cost"]}
    r = c[8, 1e-2]
    assert (r["d"], round(r["C"], 2)) == (63, 12.08)            # worked numbers of Sec. 4.3
    assert r["M"] == 35038 and round(r["queries"], -5) == 4_400_000, r
    for r in out["cost"]:
        assert r["ae_bound_frac"] >= 8 / np.pi ** 2, r           # Brassard et al., Thm. 12
        assert r["success_frac"] >= 8 / np.pi ** 2, r            # Prop. readout-cost
        assert r["success_frac_median5"] == 1.0, r
        assert r["deterministic_err_over_bound"] <= 2 / 3, r     # polynomial + encoding parts
    print("all Sec. 4 claims asserted")


if __name__ == "__main__":
    out = {"qf": [], "circuit": [], "cost": []}
    print("----- output --------------------------------------------------")
    print("1. quadratic form, eps_p = 1e-6")
    print(f"{'N':>3} {'rho':>7} {'sigmaH':>11} {'QF vs FE':>9} {'V^T e':>8} {'|psi|^2/b':>9} "
          f"{'c/b':>6} {'poly/eps':>8} {'V/H':>9} {'off':>8}")
    for N in (8, 16, 32, 64):
        for rho in RHOS:
            r = qf_case(N, rho, 1e-6)
            out["qf"].append(r)
            print(f"{N:3d} {rho:7g} {r['sigmaH']:11.6e} {r['qf_vs_fe']:9.1e} {r['VT_ehat']:8.1e} "
                  f"{r['psi_norm2_over_bound']:9.4f} {r['c_over_bound']:6.4f} "
                  f"{r['poly_err_over_eps']:8.3f} {r['voigt_over_sigmaH']:9.3f} "
                  f"{max(r['offdiag'], r['aniso']):8.1e}", flush=True)
    print("2. readout circuit, rho = 10, eps_p = 1e-6")
    for N, F in ((4, 24), (4, 32), (8, 24), (8, 32)):
        r = circuit_case(N, 10.0, F, 1e-6)
        out["circuit"].append(r)
        print(f"N={N} F={F} d={r['d']} C={r['C']:.3f} clean={r['clean']}  "
              f"Z11={r['Z11']:.6e}  vs blocks {r['Z11_vs_blocks']:.1e}  vs exact "
              f"{r['Z11_vs_exact']:.1e} (6N2^-F = {r['delta_bound']:.1e})  "
              f"Z22-Z11 {abs(r['Z22'] - r['Z11']):.1e}  Z12 {r['Z12']:.1e}  "
              f"polar {r['polarization']:.1e}  shared {r['Z11_shared']:.4e}  "
              f"[{r['seconds']:.0f} s]", flush=True)
    print("3. readout cost, rho = 10")
    for N, eps in ((4, 1e-2), (8, 1e-2), (8, 1e-3)):
        r = cost_case(N, 10.0, eps)
        out["cost"].append(r)
        print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(out, open("readout_numerics.json", "w"), indent=1, default=float)
    assert_claims(out)
    print("---------------------------------------------------------------")
