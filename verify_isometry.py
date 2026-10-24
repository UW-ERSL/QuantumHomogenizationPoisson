"""verify_isometry -- checks for the gate-level circuit U_V (Secs. 3.5 and 3.8).

    python verify_isometry.py          # about 1 minute

1. Closed-form factorization Vhat(k) = e^{i(phi1+phi2)} T c(k)/||c(k)||,
   c = (a, b, zeta/sqrt3, zeta/sqrt3), every k, N = 8 and 64. No fitting.
2. The circuit, simulated gate by gate: work register returns to zero, Q(0)
   maps |0> into the unused states 8..15, and the dense U_V (with the QFT built
   from gates) against V = W^{1/2} G (K^+)^{1/2} assembled element by element.
3. Error and Toffoli count against N and bit width F.
The arithmetic of the wavenumber-controlled stage is that of the companion
paper (package multiplexed_state_preparation), whose repository checks it in detail.
"""
import numpy as np

from quantuminversionpoisson.isometry_circuit import (IsometryCircuit, T_matrix, vhat_exact,
                                                      end_to_end_error)


def factorization():
    T, worst = T_matrix(), 0.0
    for N in (8, 64):
        for k1 in range(N):
            for k2 in range(N):
                if k1 == k2 == 0:
                    continue
                f1, f2 = np.pi * k1 / N, np.pi * k2 / N
                a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
                c = np.zeros(16); c[:4] = a, b, z / np.sqrt(3), z / np.sqrt(3)
                v = np.exp(1j * (f1 + f2)) * (T @ (c / np.linalg.norm(c)))
                worst = max(worst, np.abs(v[:8] - vhat_exact(N, k1, k2)).max(), np.abs(v[8:]).max())
    return worst


def circuit_checks():
    out = []
    for n in (3, 4):
        c = IsometryCircuit(n, F=24)
        Q, chk = c.simulate()
        q, e, _ = end_to_end_error(c, Q)
        zero = np.linalg.norm(Q[0][:8, 0])                 # Q(0)|0> in the states 8..15
        out.append((c.N, chk["work_bits_left_set"], chk["inputs_restored"], zero, q, e))
    return out


def scaling():
    rows = []
    for n in (4, 5, 6, 7):
        N = 2 ** n
        vh = np.array([vhat_exact(N, k1, k2) for k1 in range(N) for k2 in range(N)])
        for F in (16, 24, 32):
            c = IsometryCircuit(n, F=F)
            Q, chk = c.simulate()
            d = np.linalg.norm(Q[:, :8, 0] - vh, axis=1).max()
            r = c.resources()
            rows.append((N, F, d, chk["work_bits_left_set"] == 0, r["toffoli_total"], r["wires"]))
    return rows


if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"closed-form factorization, N=8 and 64: {factorization():.1e}")
    for N, dirty, rest, zero, q, e in circuit_checks():
        print(f"N={N:<3} F=24  work bits left set {dirty}  inputs restored {rest}  "
              f"|Q(0)|0> on 0..7| {zero:.0e}  QFT circuit {q:.1e}  ||Pi U_V Pi - V||_2 {e:.2e}")
    print(f"{'N':>5} {'F':>4} {'block err':>10} {'clean':>6} {'Toffoli':>9} {'wires':>7}")
    for N, F, d, cl, t, w in scaling():
        print(f"{N:>5} {F:>4} {d:>10.2e} {str(cl):>6} {t:>9,} {w:>7,}")
    print("---------------------------------------------------------------")
