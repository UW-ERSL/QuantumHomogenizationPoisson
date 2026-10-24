"""verify_full_unitary -- encoding error of the wavenumber-controlled stage (Sec. 3.8).

    python verify_full_unitary.py      # a few minutes

delta = max_{k != 0} ||Q~(k)|0> - Vhat(k)||_2, the error of the first column,
which is the only column the block encoding uses. Proposition prop:gate (iii):
with P = e^{i(phi1+phi2)} T and the rotation R_k in the plane of the two real
vectors P^dag Q~(k)|0> and P^dag Vhat(k), Q(k) = P R_k P^dag Q~(k) is unitary,
has first column Vhat(k), and ||Q~(k) - Q(k)||_2 = delta(k); this script
checks both. Q~(0)|0> must lie in the unused states 8..15.
"""
import numpy as np

from quantuminversionpoisson.isometry_circuit import IsometryCircuit, T_matrix, vhat_exact


def _plane_rotation_minus_identity(x, y):
    """R - I for the real rotation R with R x = y in span{x, y}, identity elsewhere
    (x, y real unit vectors); cos - 1 = -||x - y||^2 / 2 keeps it accurate."""
    w = y - (y @ x) * x
    s = np.linalg.norm(w)
    if s < 1e-300:
        return np.zeros((len(x), len(x)))
    B = np.stack([x, w / s], 1)
    cm1 = -0.5 * np.sum((x - y) ** 2)
    return B @ np.array([[cm1, -s], [s, cm1]]) @ B.T


def exact_unitaries(N, Q):
    """Q(k) = Q~(k) + P R_k P^dag Q~(k) - Q~(k) of Prop. prop:gate (iii), for every k
    (Q(0) = Q~(0)): exact first columns, ||Q(k) - Q~(k)||_2 = delta(k)."""
    T, Qe = T_matrix(), Q.copy()
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        v = np.concatenate([vhat_exact(N, k1, k2), np.zeros(8)])
        P = np.exp(1j * np.pi * (k1 + k2) / N) * T
        x, y = (P.conj().T @ Q[kk][:, 0]).real, (P.conj().T @ v).real
        Qe[kk] = Q[kk] + P @ _plane_rotation_minus_identity(x, y) @ P.conj().T @ Q[kk]
    return Qe


def full_error(n, F, check_unitary=False):
    """(delta, checks) of U_V at N = 2^n and F bits; checks the exact unitary if asked."""
    c = IsometryCircuit(n=n, F=F); N = c.N
    Q, chk = c.simulate()
    T = T_matrix()
    delta, gap = 0.0, 0.0
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        v = np.concatenate([vhat_exact(N, k1, k2), np.zeros(8)])
        q = Q[kk][:, 0]
        d = np.linalg.norm(q - v)
        delta = max(delta, d)
        if check_unitary:
            P = np.exp(1j * np.pi * (k1 + k2) / N) * T
            x, y = (P.conj().T @ q).real, (P.conj().T @ v).real
            D = P @ _plane_rotation_minus_identity(x, y) @ P.conj().T @ Q[kk]   # Q(k) - Q~(k)
            Qx = Q[kk] + D
            gap = max(gap, abs(np.linalg.norm(D, 2) - d) / max(d, 1e-300),
                      np.linalg.norm(Qx[:, 0] - v), np.abs(Qx.conj().T @ Qx - np.eye(16)).max())
    chk["zero_mode_outside"] = bool(np.linalg.norm(Q[0][:8, 0]) < 1e-14)
    chk["exact_unitary_gap"] = gap
    return delta, chk


C_BOUND = 6          # delta <= C_BOUND * N * 2^-F in every case below (largest measured: 5.73)

if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"{'N':>4} {'F':>3} {'delta':>10} {'delta 2^F/N':>11} {'gap':>8} {'ok':>3}")
    worst = 0.0
    for n in range(2, 9):
        for F in (12, 16, 20, 24, 28, 32):
            delta, chk = full_error(n, F, check_unitary=n <= 4)
            N = 2 ** n
            c = delta * 2.0 ** F / N
            worst = max(worst, c)
            ok = (c <= C_BOUND and chk["work_bits_left_set"] == 0 and chk["inputs_restored"]
                  and chk["zero_mode_outside"] and chk["exact_unitary_gap"] < 1e-6)
            print(f"{N:4d} {F:3d} {delta:10.2e} {c:11.2f} {chk['exact_unitary_gap']:8.1e} "
                  f"{'yes' if ok else 'NO':>3}", flush=True)
            assert ok
    print(f"largest delta 2^F / N over all cases: {worst:.2f}")
