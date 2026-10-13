"""verify_qiskit_qsvt -- the readout of Sec. 4 as one Qiskit circuit, QSVT phases included.

    python verify_qiskit_qsvt.py        # about 30 seconds

Square inclusion, v_f = 1/4, sigma_max = 1, N = 4, rho = 3, eps_p = 1e-2
(degree d = 15). Every stage is a Qiskit gate:

    U_V      inverse QFT on each position coordinate, the wavenumber-controlled
             16 x 16 local unitary Q(k) at exact angles (one multiplexed
             UnitaryGate on position and local qubits; Q(0) maps |0> to |8>,
             the zero-mode branch of the gate-level circuit), QFT
    U_Sigma  Ry on one ancilla, a +-1 diagonal on position controlled by it, Ry^T
    U_M      U_V^dag U_Sigma U_V, block on (local = 0, ancilla = 0)
    U_q      QSVT: d alternating calls to U_M and U_M^dag between projector
             phases exp(i phi (2 Pi - I)); the real part is taken by an LCU over
             +phi and -phi on one extra qubit (one RZZ per phase, no extra query)
    readout  prep, U_Sigma, U_V^dag, flag, U_q, flag, U_V, U_Sigma^dag, prep^dag

The fixed-point arithmetic that computes the angles of Q(k) is checked
separately in verify_isometry_aer.py; here Q(k) enters at exact angles.

Checks:
1. block of U_M equals Mtilde = M + sbar Pi_0, assembled independently;
2. block of U_q equals q_t(Mtilde) / alpha_q;
3. all-zero amplitude of the readout circuit equals psi^T q_t(Mtilde) psi / alpha_q,
   and sbar - alpha_q A reproduces the finite element sigma^H within eps_p;
4. the all-zero probability A^2, sampled with Aer (shots below), gives sigma^H
   within the shot-noise interval.

QSP phases: symmetric-QSP Newton solver of pyqsp (Dong et al.), following
qsp_phases() of UW-ERSL/SpectralCorrectionQSVT (QSVTSolvers.py), then mapped
from the Wx convention to the reflection convention of a projected block
encoding.
"""
import contextlib
import io
import time

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import Chebyshev
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, DiagonalGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

from verify_square import assemble, square_chi
from verify_inverse import k_function, q_poly
from verify_readout import e_hat, fe_sigmaH
from quantuminversionpoisson.isometry_circuit import Q_exact

N, RHO, EPS_P, SHOTS, SEED = 4, 3.0, 1e-2, 200_000, 7
n = int(np.log2(N))
m = 2 * n                                                    # position qubits
Q_ZERO = np.kron(np.array([[0, 1], [1, 0]]), np.eye(8))     # k = 0: |0> -> |8>, as in the gates


# ---- polynomial and phases ---------------------------------------------------------
def scaled_poly(rho, eps_p):
    """q_t / alpha_q as a Chebyshev series of odd parity; alpha_q = max_[-1,1] |q_t|."""
    q, d = q_poly(rho, eps_p)
    p = Chebyshev(np.polynomial.chebyshev.chebinterpolate(q, d))
    p.coef[0::2] = 0.0
    x = np.linspace(1e-9, 1.0, 2_000_001)
    alpha = float(np.abs(q(x)).max())
    return q, d, alpha, Chebyshev(p.coef / alpha)


def wx_phases(p):
    """Wx phases with Re <0|U|0> = p (SpectralCorrectionQSVT, QSVTSolvers.qsp_phases)."""
    from pyqsp.sym_qsp_opt import newton_solver               # only the phases need pyqsp
    c = np.asarray(p.coef, float)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, seq = newton_solver(c[1::2], 1, crit=1e-14)
    ph = np.array(seq.full_phases, float)
    ph[0] -= np.pi / 2
    return ph


def reflection_phases(ph):
    """R(x) = -i e^{i pi Z/4} W(x) e^{i pi Z/4}: shift interior phases by -pi/2, end
    phases by -pi/4, and absorb (-i)^d into the first phase so that Re is unchanged."""
    d = len(ph) - 1
    r = ph - np.pi / 2
    r[0] += np.pi / 4 + d * np.pi / 2
    r[-1] += np.pi / 4
    return r


def scalar_response(phi, x):
    """<0| e^{i phi_0 Z} R(x) e^{i phi_1 Z} ... R(x) e^{i phi_d Z} |0>."""
    s = np.sqrt(1 - x * x)
    U = np.diag(np.exp(1j * phi[0] * np.array([1, -1])))
    for f in phi[1:]:
        U = U @ np.array([[x, s], [s, -x]]) @ np.diag(np.exp(1j * f * np.array([1, -1])))
    return U[0, 0]


# ---- circuits ----------------------------------------------------------------------
def u_v(chi):
    """On [position (m), local (4)]: fft2, wavenumber-controlled Q(k), ifft2."""
    pos, loc = QuantumRegister(m, "pos"), QuantumRegister(4, "loc")
    qc = QuantumCircuit(pos, loc, name="U_V")
    j, i = pos[:n], pos[n:]                                  # position p = i N + j
    for reg in (j, i):
        qc.append(QFTGate(n).inverse(), reg)                     # numpy fft = inverse QFT
    mux = np.zeros((16 * N * N,) * 2, complex)               # index p + N^2 l
    for k1 in range(N):
        for k2 in range(N):
            k = k1 * N + k2
            mux[k::N * N, k::N * N] = Q_exact(N, k1, k2) if k else Q_ZERO
    qc.append(UnitaryGate(mux, label="Q(k)"), list(pos) + list(loc))
    for reg in (j, i):
        qc.append(QFTGate(n), reg)
    return qc


def u_sigma(chi, rho):
    """On [position (m), ancilla]: block b0 I + b1 (2 chi - 1) = Sigmatilde."""
    b0, b1 = (1 + 1 / rho) / 2, (1 - 1 / rho) / 2
    th = 2 * np.arctan2(np.sqrt(b1), np.sqrt(b0))
    qc = QuantumCircuit(m + 1, name="U_Sigma")
    qc.ry(th, m)
    qc.append(DiagonalGate([1.0] * N * N + list(2.0 * chi - 1)), range(m + 1))
    qc.ry(-th, m)
    return qc


def u_m(UV, US):
    """On [position (m), local (4), ancilla]: U_V^dag U_Sigma U_V."""
    qc = QuantumCircuit(m + 5, name="U_M")
    qc.append(UV, range(m + 4))
    qc.append(US, list(range(m)) + [m + 4])
    qc.append(UV.inverse(), range(m + 4))
    return qc


def projector_phase(qc, phi, anc, z, c):
    """exp(i phi Z_c (2 Pi - I)), Pi = |0><0| on anc: mark Pi on z, RZZ(2 phi), unmark."""
    qc.mcx(anc, z, ctrl_state=0)
    qc.rzz(2 * phi, c, z)
    qc.mcx(anc, z, ctrl_state=0)


def u_q(UM, phi):
    """On [position (m), local (4), ancilla, z, c]: block Re P(Mtilde) on c = z = anc = 0."""
    qc = QuantumCircuit(m + 7, name="U_q")
    data, anc, z, c = list(range(m + 5)), list(range(m, m + 5)), m + 5, m + 6
    UMd = UM.inverse()
    qc.h(c)
    d = len(phi) - 1
    for j in range(d + 1):                                   # rightmost factor first
        projector_phase(qc, phi[d - j], anc, z, c)
        if j < d:
            qc.append(UM if j % 2 == 0 else UMd, data)
    qc.h(c)
    return qc


def flag(qc, loc, f):
    """f ^= (local != 0)."""
    qc.x(f)
    qc.mcx(loc, f, ctrl_state=0)


def readout(UV, US, Uq):
    """Qubits: pos (m), loc (4), qa, z, c, s1, s2, f1, f2."""
    qc = QuantumCircuit(m + 11, name="readout")
    pos, loc = list(range(m)), list(range(m, m + 4))
    qa, z, c, s1, s2, f1, f2 = range(m + 4, m + 11)
    prep = pos + [loc[1], loc[2]]                            # ehat: local states 2g, g = 0..3
    qc.h(prep)
    qc.append(US, pos + [s1])
    qc.append(UV.inverse(), pos + loc)
    flag(qc, loc, f1)
    qc.append(Uq, pos + loc + [qa, z, c])
    flag(qc, loc, f2)
    qc.append(UV, pos + loc)
    qc.append(US.inverse(), pos + [s2])
    qc.h(prep)
    return qc


# ---- independent references ----------------------------------------------------------
def references(chi, rho):
    A = assemble(N).tocsr()
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    sbar = chi.mean() + (1 - chi.mean()) / rho
    Kih = k_function(lambda e: e ** -0.5, N)
    M = Kih((A.T @ sp.diags(St) @ A) @ Kih(np.eye(N * N)))
    M = (M + M.T) / 2
    Mt = M + sbar * np.full((N * N, N * N), 1 / (N * N))       # + sbar Pi_0
    psi = Kih(A.T @ (St * e_hat(N, 1)))
    return Mt, psi, sbar, fe_sigmaH(N, A, St)[0, 0]


def matfun(Mt, f):
    lam, U = np.linalg.eigh(Mt)
    return (U * f(lam)) @ U.T, lam


if __name__ == "__main__":
    t0 = time.time()
    chi = square_chi(N).astype(float)
    Mt, psi, sbar, sH = references(chi, RHO)
    q, d, alpha, p = scaled_poly(RHO, EPS_P)
    phi = reflection_phases(wx_phases(p))
    xs = np.linspace(-1, 1, 2001)
    phase_err = max(abs(scalar_response(phi, x).real - p(x)) for x in xs)

    UV, US = u_v(chi), u_sigma(chi, RHO)
    UM = u_m(UV, US)
    Uq = u_q(UM, phi)
    nb = N * N                                               # block: first N^2 indices
    BM = Operator(UM).data[:nb, :nb]
    Bq = np.array([Statevector.from_int(k, 2 ** Uq.num_qubits).evolve(Uq).data[:nb]
                   for k in range(nb)]).T
    Pq, lam = matfun(Mt, lambda x: q(x) / alpha)

    R = readout(UV, US, Uq)
    A = Statevector(R).data[0]
    A_ref = psi @ Pq @ psi
    c_exact = psi @ matfun(Mt, lambda x: 1 / x)[0] @ psi

    Rm = R.copy()
    Rm.measure_all()
    sim = AerSimulator(method="statevector", seed_simulator=SEED)
    counts = sim.run(transpile(Rm, sim, optimization_level=0), shots=SHOTS).result().get_counts()
    p0 = counts.get("0" * R.num_qubits, 0) / SHOTS
    A_hat = np.sqrt(p0)
    dA = 1.96 * np.sqrt(p0 * (1 - p0) / SHOTS) / (2 * A_hat)  # 95% interval on A

    print("----- output --------------------------------------------------")
    print(f"N = {N}, rho = {RHO:g}, eps_p = {EPS_P:g}: d = {d}, alpha_q = {alpha:.4f}, "
          f"spectrum of Mtilde in [{lam.min():.4f}, {lam.max():.4f}], sbar = {sbar:.4f}")
    print(f"qubits: U_M {UM.num_qubits}, U_q {Uq.num_qubits}, readout {R.num_qubits}; "
          f"U_M queries per readout {d}")
    print(f"1. |block(U_M) - Mtilde|_max                 = {np.abs(BM - Mt).max():.2e}")
    print(f"   QSP phases: max_[-1,1] |Re S(x) - q_t/alpha| = {phase_err:.2e}")
    print(f"2. |block(U_q) - q_t(Mtilde)/alpha_q|_max    = {np.abs(Bq - Pq).max():.2e}")
    print(f"3. A = {A.real:.10f} (imag {abs(A.imag):.1e}),  psi^T q_t psi / alpha_q = {A_ref:.10f}")
    print(f"   sbar - alpha_q A = {sbar - alpha * A.real:.8f},  FE sigma^H = {sH:.8f},  "
          f"rel. error in correction {abs(alpha * A.real - c_exact) / c_exact:.2e} (eps_p = {EPS_P:g})")
    print(f"4. Aer, {SHOTS} shots: P(all zero) = {p0:.6f}, A_hat = {A_hat:.5f} +- {dA:.5f}, "
          f"sigma^H estimate = {sbar - alpha * A_hat:.5f} +- {alpha * dA:.5f}")
    print(f"[{time.time() - t0:.0f} s]")
    print("---------------------------------------------------------------")
    assert phase_err < 1e-10
    assert np.abs(BM - Mt).max() < 1e-12 and np.abs(Bq - Pq).max() < 1e-10
    assert abs(A - A_ref) < 1e-10 and abs(alpha * A.real - c_exact) <= EPS_P * c_exact
    assert abs(sbar - alpha * A_hat - sH) <= alpha * dA
