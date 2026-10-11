"""verify_square -- numerical verification (Sec. 4): square inclusion (v_f = 1/4), scalar conduction.

    python verify_square.py            # a few minutes (N = 64 is dense)

Condition numbers of K^chi / sigma_max (as seen by QSVT, with the LCU
subnormalization alpha = 16/3 of the two-phase scalar encoding of
pyblockencode) and of the preconditioned operator M, and the block of
U_V^dag U_Sigma U_V from the gate-level circuit against Mtilde = M + sbar Pi_0.
"""
import json
import time

import numpy as np
import scipy.sparse as sp

from quantuminversionpoisson.isometry_circuit import _B, _CORNERS, IsometryCircuit
from quantuminversionpoisson.fourier_symbol import symbol

ALPHA = 16 / 3       # LCU subnormalization of K^chi / sigma_max (pyblockencode, POISSON2D_2PHASE)


def square_chi(N):
    """Centered square of side N/2: v_f = 1/4. chi[e] = 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return ((i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)).ravel().astype(float)


def assemble(N):
    """Sparse A = W^{1/2} G (8 N^2 x N^2), row 8 e + 2 g + c, element e = i N + j."""
    h = 1 / N
    Bs = [_B(g, h) for g in range(4)]
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            e = i * N + j
            for g in range(4):
                for a, (cx, cy) in enumerate(_CORNERS):
                    n = ((i + cx) % N) * N + (j + cy) % N
                    for c in range(2):
                        rows.append(8 * e + 2 * g + c); cols.append(n); vals.append(Bs[g][c, a])
    G = sp.csr_matrix((vals, (rows, cols)), shape=(8 * N * N, N * N))
    return (h / 2) * G                                         # sqrt(w) = h / 2


def circulant(f, N):
    """Dense circulant matrix f(K), zero on the constant field; node n = i N + j."""
    S = symbol(N)
    T = np.where(S > 1e-12, f(np.clip(S, 1e-300, None)), 0.0)
    ker = np.fft.ifft2(T).real
    i, j = np.divmod(np.arange(N * N), N)
    return ker[(i[:, None] - i[None, :]) % N, (j[:, None] - j[None, :]) % N]


def operators(N, rho):
    A = assemble(N).tocsr()
    chi = square_chi(N)
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)       # Sigma / sigma_max
    K = (A.T @ A).toarray()
    Kc = (A.T @ sp.diags(St) @ A).toarray()                    # K^chi / sigma_max
    return A, St, K, Kc, chi


_KIH = {}


def spectra(N, rho):
    A, St, K, Kc, chi = operators(N, rho)
    if N not in _KIH:
        _KIH[N] = circulant(lambda e: e ** -0.5, N)
        assert np.abs(circulant(lambda e: e, N) - K).max() < 1e-10 * np.abs(K).max()
    Kih = _KIH[N]
    lm = np.linalg.eigvalsh(Kih @ Kc @ Kih)
    lm = lm[np.abs(lm) > 1e-9]
    lk = np.linalg.eigvalsh(Kc)
    lk = lk[lk > 1e-9 * lk.max()]
    return dict(N=N, rho=rho, Mmin=float(lm.min()), Mmax=float(lm.max()),
                kappaM=float(lm.max() / lm.min()),
                Kmax=float(lk.max()), Kmin=float(lk.min()), kappaK=float(lk.max() / lk.min()))


def qsvt_degree(kappa, eps):
    """Degree 2t - 1 of the odd QSVT polynomial with relative residual eps on [1/kappa, 1]."""
    a = 1 / kappa ** 2
    t = int(np.ceil(np.arccosh(1 / eps) / np.arccosh((1 + a) / (1 - a))))
    return 2 * t - 1


def circuit_block(N, rho, F):
    """Block of U_V^dag U_Sigma U_V on the input subspace, from the gate-level circuit."""
    chi = square_chi(N)
    circ = IsometryCircuit(int(np.log2(N)), F=F)
    Q, chk = circ.simulate()
    F1 = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    Fm = np.kron(F1, F1)
    BD = np.zeros((16 * N * N, N * N), complex)
    for k in range(N * N):
        BD[16 * k:16 * k + 16, k] = Q[k, :, 0]                 # all 16 rows
    X = np.kron(Fm.conj().T, np.eye(16)) @ BD @ Fm              # U_V on the input subspace
    Se = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 16)        # U_Sigma block: Sigma_e x I_16
    return X.conj().T @ (Se[:, None] * X), chk


def prop4_check(N, rho, F):
    """Block from the gate-level circuit against Mtilde = M + sbar Pi_0."""
    A, St, K, Kc, chi = operators(N, rho)
    Kih = circulant(lambda e: e ** -0.5, N)
    M = Kih @ Kc @ Kih
    sbar = chi.mean() + (1 - chi.mean()) / rho
    Mt = M + sbar * np.ones((N * N, N * N)) / N ** 2
    Mc, chk = circuit_block(N, rho, F)
    lm = np.linalg.eigvalsh((Mc + Mc.conj().T) / 2)
    return dict(N=N, rho=rho, F=F, err=float(np.linalg.norm(Mc - Mt, 2)),
                spec_min=float(lm.min()), spec_max=float(lm.max()), sbar=float(sbar),
                clean=chk["work_bits_left_set"] == 0)


def alpha_check():
    """Cross-check ALPHA against pyblockencode, when it is installed."""
    try:
        from pyblockencode import POISSON2D_2PHASE
    except ImportError:
        return None
    return POISSON2D_2PHASE(vf=0.25, E1=1.0, E2=0.1).alpha()


if __name__ == "__main__":
    out = {"spectra": [], "prop4": [], "degree": []}
    print("----- output --------------------------------------------------")
    a = alpha_check()
    print(f"alpha = 16/3 = {ALPHA:.4f}; pyblockencode: {a if a is None else f'{a:.4f}'}")
    print(f"{'N':>3} {'rho':>7} {'kappa(K^chi)':>13} {'kappa(M)':>10} {'M min*rho':>10} {'M max':>8} {'s':>5}")
    prev = {}
    for N in (8, 16, 32, 64):
        for rho in (1.5, 3.75, 10, 100, 1e4):
            t = time.time()
            r = spectra(N, rho)
            out["spectra"].append(r)
            kK = ALPHA / r["Kmin"]
            print(f"{N:3d} {rho:7g} {kK:13.1f} {r['kappaM']:10.6f} {r['Mmin'] * rho:10.6f} "
                  f"{r['Mmax']:8.6f} {time.time() - t:5.1f}", flush=True)
            out["degree"].append(dict(N=N, rho=rho, alpha=ALPHA, kappa_unpre=kK,
                                      d_unpre=qsvt_degree(kK, 1e-6),
                                      d_pre=qsvt_degree(rho, 1e-6)))
    for N in (8, 16):
        for F in (24, 32):
            r = prop4_check(N, 10.0, F)
            out["prop4"].append(r)
            print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(out, open("square_numerics.json", "w"), indent=1)
    print("---------------------------------------------------------------")
