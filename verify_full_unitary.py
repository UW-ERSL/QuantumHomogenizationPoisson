"""verify_full_unitary -- full 16 x 16 error of the wavenumber-controlled stage (App. A.3).

    python verify_full_unitary.py      # a few minutes

delta = max_{k != 0} ||Q(k) - Q_ex(k)||_2, with Q_ex(k) the unitary of the
two-angle factorization at exact angles. At k = (N/2, N/2), where r = 0,
omega is arbitrary and Q_ex uses the circuit's value. Because the work
register returns to zero and the Fourier stages are exact, delta is the
distance between U_V and an exact unitary whose block is V.
"""
import numpy as np

from quantuminversionpoisson.isometry_circuit import IsometryCircuit, Q_exact, vhat_exact


def full_error(n, F):
    """(block error, full 16 x 16 error, checks) of U_V at N = 2^n and F bits."""
    c = IsometryCircuit(n=n, F=F); N = c.N
    Q, chk = c.simulate()
    blk = full = 0.0
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        om = c.decoded["omega"][kk] if k1 == k2 == N // 2 else None
        full = max(full, np.linalg.norm(Q[kk] - Q_exact(N, k1, k2, om), 2))
        blk = max(blk, np.linalg.norm(Q[kk][:8, 0] - vhat_exact(N, k1, k2)))
    full = max(full, np.linalg.norm(Q[0][:, 0] - np.eye(16)[8]))     # k = 0 column
    return blk, full, chk


C_BOUND = 6          # delta <= C_BOUND * N * 2^-F in every case below (largest measured: 5.73)

if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"{'N':>4} {'F':>3} {'block err':>10} {'full err':>10} {'ratio':>6} "
          f"{'delta 2^F/N':>11} {'ok':>3}")
    worst = 0.0
    for n in range(2, 9):
        for F in (12, 16, 20, 24, 28, 32):
            blk, full, chk = full_error(n, F)
            N = 2 ** n
            c = full * 2.0 ** F / N
            worst = max(worst, c)
            ok = c <= C_BOUND and chk["work_bits_left_set"] == 0 and chk["inputs_restored"]
            print(f"{N:4d} {F:3d} {blk:10.2e} {full:10.2e} {full / blk:6.2f} {c:11.2f} "
                  f"{'yes' if ok else 'NO':>3}", flush=True)
    print(f"largest delta 2^F / N over all cases: {worst:.2f}")
    print("---------------------------------------------------------------")
