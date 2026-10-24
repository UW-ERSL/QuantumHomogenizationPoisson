"""isometry_circuit -- a gate-level circuit U_V for the partial isometry V (scalar case).

V = W^{1/2} G (K^+)^{1/2} = (F^dag (x) I_8) blockdiag_k Vhat(k) F
(Proposition 2). Vhat(k) is a unit vector of length 8 for k != 0. With
phi_j = pi k_j / N (Sec. 3.8),

    a = sin phi1 cos phi2,   b = cos phi1 sin phi2,   zeta = sin phi1 sin phi2,

    Vhat(k) = e^{i (phi1 + phi2)} T c(k) / ||c(k)||,   c = (a, b, zeta/sqrt3, zeta/sqrt3),

with T the fixed unitary of T_matrix. IsometryCircuit prepares c(k)/||c(k)||
for all k at once with the multiplexed CORDIC circuit of the companion paper
(multiplexed_state_preparation.poisson_isometry), without tabulating anything, and
checks it. Q_exact, exact_angles, and canonical are an exact two-angle
unitary with the same first column, used by the companion-paper readout
scripts.

Local index l = 8 b3 + 4 b2 + 2 b1 + b0. Gradient index 2 g + c for Gauss
point g and component c (x, y). Element (i, j) has index i N + j, i along x.
"""
from __future__ import annotations

import numpy as np



# ==========================================================================
# 1. the exact reference (no factorization)
# ==========================================================================
_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]
_XI = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
_Q = 1 / np.sqrt(3)
_GP = [(-_Q, -_Q), (_Q, -_Q), (_Q, _Q), (-_Q, _Q)]     # (xi_g, eta_g)


def _B(g, h):
    """2 x 4 gradient of the Q4 shape functions at Gauss point g."""
    r, s = _GP[g]
    Bm = np.zeros((2, 4))
    for a, (xa, ya) in enumerate(_XI):
        Bm[0, a] = xa * (1 + ya * s) / (2 * h)
        Bm[1, a] = ya * (1 + xa * r) / (2 * h)
    return Bm


def vhat_exact(N, k1, k2):
    """Vhat(k) = W_e^{1/2} Ghat(k) Shat(k)^{-1/2}, length 8, paper conventions."""
    if k1 == 0 and k2 == 0:
        return np.zeros(8, complex)
    h = 1 / N
    t1, t2 = 2 * np.pi * k1 / N, 2 * np.pi * k2 / N
    ph = [np.exp(1j * (t1 * cx + t2 * cy)) for cx, cy in _CORNERS]
    A = (h / 2) * np.concatenate([_B(g, h) @ np.array(ph) for g in range(4)])
    return A / np.linalg.norm(A)


def assemble_G(N):
    """Dense G (8 N^2 x N^2): row 8 e + 2 g + c, column node i N + j."""
    h = 1 / N
    G = np.zeros((8 * N * N, N * N))
    Bs = [_B(g, h) for g in range(4)]
    for i in range(N):
        for j in range(N):
            e = i * N + j
            for g in range(4):
                for a, (cx, cy) in enumerate(_CORNERS):
                    n = ((i + cx) % N) * N + (j + cy) % N
                    G[8 * e + 2 * g:8 * e + 2 * g + 2, n] += Bs[g][:, a]
    return G


def V_assembled(N):
    """V = W^{1/2} G (K^+)^{1/2}, assembled element by element (no symbol)."""
    Wh = 1 / (2 * N)                                    # sqrt(w), w = h^2/4
    A = Wh * assemble_G(N)
    K = A.T @ A
    ev, U = np.linalg.eigh(K)
    ih = np.where(ev > 1e-10 * ev.max(), 1 / np.sqrt(np.abs(ev)), 0)
    return A @ ((U * ih) @ U.T)


# ==========================================================================
# 2. local (4-qubit) building blocks
# ==========================================================================
def _R2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _pairs(pairs, t):
    M = np.eye(16)
    for i0, i1 in pairs:
        M[np.ix_([i0, i1], [i0, i1])] = _R2(t)
    return M


def Rtau(t):
    """Rotation by t in the planes (0,2), (1,3), (4,6), (5,7): qubit 1, b3 = 0."""
    return _pairs([(0, 2), (1, 3), (4, 6), (5, 7)], t)


def Rsplit():
    """Fixed rotation by pi/4 in the planes (2,3), (6,7): qubit 0, b1 = 1, b3 = 0."""
    return _pairs([(2, 3), (6, 7)], np.pi / 4)


def Romega(t):
    """Rotation by t in the planes (0,1), (4,5): qubit 0, b1 = 0, b3 = 0."""
    return _pairs([(0, 1), (4, 5)], t)


G_TAU = np.sqrt(2 / 3)


def exact_angles(N, k1, k2):
    f1, f2 = np.pi * k1 / N, np.pi * k2 / N
    a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
    return np.arctan2(b, a), np.arctan2(G_TAU * z, np.hypot(a, b))


def canonical(omega, tau):
    """Romega(omega) Rsplit Rtau(tau) |0>, length 16."""
    return (Romega(omega) @ Rsplit() @ Rtau(tau))[:, 0]


def Q_exact(N, k1, k2, omega=None):
    """Full 16 x 16 unitary of the two-angle factorization at exact angles."""
    om, tau = exact_angles(N, k1, k2)
    if omega is not None:
        om = omega
    return (np.exp(1j * np.pi * (k1 + k2) / N)
            * T_matrix() @ Romega(om) @ Rsplit() @ Rtau(tau))


def T_matrix():
    """The fixed unitary T, in closed form.

    Canonical local states and their images in the gradient space
    (row 2 g + c, c = x, y; g the Gauss point (xi_g, eta_g)):

        0 : (i/2) sum_g e_{g,x}                     mean x-gradient
        1 : (i/2) sum_g e_{g,y}                     mean y-gradient
        2 : -(sqrt 3 / 2) sum_g eta_g e_{g,x}       hourglass, x
        3 : -(sqrt 3 / 2) sum_g xi_g  e_{g,y}       hourglass, y

    The four images are orthonormal; T is completed to a unitary on the
    gradient block 0..7 and fixes the unused states 8..15.
    """
    img = np.zeros((8, 4), complex)
    for g, (xg, eg) in enumerate(_GP):
        img[2 * g, 0] = 0.5j
        img[2 * g + 1, 1] = 0.5j
        img[2 * g, 2] = -np.sqrt(3) / 2 * eg
        img[2 * g + 1, 3] = -np.sqrt(3) / 2 * xg
    assert np.allclose(img.conj().T @ img, np.eye(4), atol=1e-13)
    U, _, _ = np.linalg.svd(np.eye(8) - img @ img.conj().T)
    T = np.eye(16, dtype=complex)
    T[:8, :4] = img
    T[:8, 4:8] = U[:, :4]
    assert np.allclose(T.conj().T @ T, np.eye(16), atol=1e-12)
    return T


# ==========================================================================
# 3. the circuit
# ==========================================================================
class IsometryCircuit:
    """U_V for N = 2^n with F fractional bits (Sec. 3.8, Prop. gate).

    Q~(k) = e^{i(phi1+phi2)} T Qc(k) X_b3^[k = 0], where Qc is the multiplexed
    preparation of c(k)/||c(k)||, c = (a, b, zeta/sqrt3, zeta/sqrt3), of the
    companion paper (multiplexed_state_preparation.poisson_isometry) on the local qubits
    (b1, b0) = (t_0, t_1), T is the fixed unitary of T_matrix, and X_b3 moves
    k = 0 to the unused states 8..15. The zero-mode flag is an AND of the 2n
    negated wavenumber bits (2n - 1 Toffoli, uncomputed before the rotation
    stage, so its temporaries reuse those of the rotation stage). The
    arithmetic is simulated gate by gate; the rotation stage applies the angles
    its decision bits encode."""

    def __init__(self, n, F=20):
        from multiplexed_state_preparation import poisson_isometry
        if F < n:
            raise ValueError("need F >= n")
        self.n, self.N, self.F, self.I = n, 2 ** n, F, F
        self.sv = poisson_isometry(n, F)
        self.rot = self.sv.tree.rotation_counts()
        self.n_zero = 2 * n - 1
        self.n_wires = self.sv.c.n + max(self.n_zero, self.rot["temporaries"])

    # -- simulation -----------------------------------------------------------
    def simulate(self):
        """Q (N^2, 16, 16), the phase included, and a dict of checks."""
        from multiplexed_state_preparation.revcirc import BitSim
        sv, t, N, n = self.sv, self.sv.tree, self.N, self.n
        kk = np.arange(N * N)
        k1, k2 = kk // N, kk % N
        sim = BitSim(sv.c.n, N * N)
        for reg, val in zip(sv.k, (k1, k2)):
            sim.set_int(reg, val)
        sim.run(sv.c.gates[:sv.n_arith])
        th = t.angles(sim)                       # node angles from the decision bits
        sim.run(sv.c.gates[sv.n_arith:])
        work = [q for q in range(sv.c.n) if q not in sv.k[0] + sv.k[1]]
        dirty = int(sum(sim.get_bits(q).sum() for q in work))
        restored = bool(np.all(sim.get_int(sv.k[0], False) == k1) and
                        np.all(sim.get_int(sv.k[1], False) == k2))
        X8 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(8))
        T = T_matrix()
        Q = np.empty((N * N, 16, 16), complex)
        for i in range(N * N):
            R0 = _R2(th[(0, 0)][i])                                   # t_0
            U4 = np.kron(np.diag([1, 0]), _R2(th[(1, 0)][i])) + np.kron(np.diag([0, 1]), _R2(th[(1, 1)][i]))
            U = T @ np.kron(np.eye(4), U4 @ np.kron(R0, np.eye(2)))
            if i == 0:
                U = U @ X8
            Q[i] = np.exp(1j * np.pi * (k1[i] + k2[i]) / N) * U
        return Q, dict(work_bits_left_set=dirty, inputs_restored=restored)

    # -- resources -------------------------------------------------------------
    def resources(self):
        """Toffoli gates (compute plus uncompute), qubits, and rotations of U_V."""
        sv, r = self.sv, self.rot
        prep = sum(sv.toffoli().values()) + r["toffoli"]
        s = len(sv.tree.digits[(1, 1)])
        assert prep == 20 * self.F ** 2 + 59 * self.F + 21 + 2 * (self.F + 3) * (s - 1)
        return dict(N=self.N, F=self.F, I=self.I, width=sv.w,
                    toffoli_prep=prep, toffoli_zero=2 * self.n_zero,
                    toffoli_total=prep + 2 * self.n_zero,
                    work=self.n_wires - 2 * self.n, wires=self.n_wires + 4,
                    controlled_rotations=r["controlled_ry"], fixed_rotations=r["fixed_ry"],
                    flag_z=r["flag_z"], phase_gates=2 * self.n)


# ==========================================================================
# 4. QFT circuit and the dense end-to-end check
# ==========================================================================
def qft_matrix_from_gates(n):
    """Standard QFT (e^{+2 pi i jk/2^n}) built from H, controlled phases, swaps."""
    D = 2 ** n
    U = np.eye(D, dtype=complex)
    idx = np.arange(D)
    bit = lambda q: (idx >> (n - 1 - q)) & 1
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply_1q(U, q, M):
        b = bit(q)
        partner = idx ^ (1 << (n - 1 - q))
        return M[b, b][:, None] * U + M[b, 1 - b][:, None] * U[partner]

    for q in range(n):
        U = apply_1q(U, q, H)
        for t in range(q + 1, n):
            ph = np.exp(2j * np.pi / 2 ** (t - q + 1))
            U = np.where((bit(q) & bit(t))[:, None] == 1, ph * U, U)
    perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(D)])
    return U[perm]


def end_to_end_error(circ, Q):
    """|| (I x Pi_out) U_V (I x Pi_in) - V ||_2 from the gates, densely."""
    N, n = circ.N, circ.n
    F1s = qft_matrix_from_gates(n)
    F1 = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    qft_err = np.abs(F1s - F1.conj().T).max()          # paper's F is QFT^dag
    Fm = np.kron(F1, F1)
    V = V_assembled(N)
    BD = np.zeros((8 * N * N, N * N), complex)
    for k in range(N * N):
        BD[8 * k:8 * k + 8, k] = Q[k, :8, 0]
    UV = np.kron(Fm.conj().T, np.eye(8)) @ BD @ Fm
    return qft_err, np.linalg.norm(UV - V, 2), np.abs(UV.imag).max()
