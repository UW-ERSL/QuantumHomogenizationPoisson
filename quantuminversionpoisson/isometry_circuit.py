"""isometry_circuit -- a gate-level circuit U_V for the partial isometry V (scalar case).

V = W^{1/2} G (K^+)^{1/2} = (F^dag (x) I_8) blockdiag_k Vhat(k) F
(Proposition 2). Vhat(k) is a unit vector of length 8 for k != 0. This module
builds the wavenumber-controlled stage sum_k |k><k| (x) Q(k) from reversible
arithmetic, without tabulating anything, and checks it.

Factorization used (Proposition 6, Appendix B)
----------------------------------------------
With phi_j = pi k_j / N,

    a = sin phi1 cos phi2,   b = cos phi1 sin phi2,   zeta = sin phi1 sin phi2,
    r = sqrt(a^2 + b^2),

    Vhat(k) = e^{i (phi1 + phi2)} T . Romega(omega) . Rsplit . Rtau(tau) |0>,

    omega = atan2(b, a),   tau = atan2(sqrt(2/3) zeta, r),

where Rtau rotates the planes (|0>,|2>) and (|1>,|3>), Rsplit rotates
(|2>,|3>) by pi/4, Romega rotates (|0>,|1>), and T is a fixed unitary. All
three act only on the states with b3 = 0, so the unused states |8>..|15> are
fixed and Q(0) maps |0> to |8> exactly. The two angles come from two CORDIC
rotations of pi (k1 +- k2)/N and two CORDIC vectorings; the decision bits
control the rotations directly, and the arithmetic is then run in reverse.

Local index l = 8 b3 + 4 b2 + 2 b1 + b0. Gradient index 2 g + c for Gauss
point g and component c (x, y). Element (i, j) has index i N + j, i along x.
"""
from __future__ import annotations

import numpy as np

from .revcirc import (Circuit, BitSim, add_view, copy_view, load_select,
                      const_mult, cordic_rotate, cordic_vector, cordic_gain, to_int)


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
    """U_V for N = 2^n, F fractional bits, I CORDIC iterations."""

    def __init__(self, n, F=20, I=None, guard=4):
        self.n, self.N, self.F = n, 2 ** n, F
        self.I = I if I is not None else F
        self.w = F + guard
        if F < n:
            raise ValueError("need F >= n")
        self._build()

    # -- arithmetic -----------------------------------------------------------
    def _build(self):
        n, F, I, w = self.n, self.F, self.I, self.w
        c = Circuit()
        self.c = c
        self.k1, self.k2 = c.alloc(n), c.alloc(n)
        K = cordic_gain(I)
        Fz, wz = F, F + 1

        # angles pi (k1 + k2)/N and pi (k1 - k2)/N, in units of pi, mod 2
        def place(reg):
            v = [None] * wz
            for j in range(n):
                v[Fz - n + j] = reg[j]
            return v

        zp, zm = c.alloc(wz), c.alloc(wz)
        copy_view(c, place(self.k1), zp)
        add_view(c, zp, place(self.k2))
        copy_view(c, place(self.k1), zm)
        add_view(c, zm, place(self.k2), neg=True)

        cs = {}
        for name, z in (("p", zp), ("m", zm)):
            f = c.alloc(1)[0]
            c.cx(z[-1], f); c.cx(z[-2], f)       # |z| >= 1/2
            c.cx(f, z[-1])                       # z <- z + 1 (mod 2)
            x, y = c.alloc(w), c.alloc(w)
            load_select(c, x, to_int(1 / K, F, w), to_int(-1 / K, F, w), f)
            cordic_rotate(c, x, y, z, I, F, Fz)
            cs[name] = (x, y)                    # (cos, sin)
        (xp, yp), (xm, ym) = cs["p"], cs["m"]

        from .revcirc import shr
        a, b, zeta = c.alloc(w), c.alloc(w), c.alloc(w)
        copy_view(c, shr(yp, 1), a); add_view(c, a, shr(ym, 1))
        copy_view(c, shr(yp, 1), b); add_view(c, b, shr(ym, 1), neg=True)
        copy_view(c, shr(xm, 1), zeta); add_view(c, zeta, shr(xp, 1), neg=True)
        self.abz = (a, b, zeta)

        # omega = atan2(b, a): quadrant flag, then vectoring (a <- K r)
        fo = c.alloc(1)[0]
        c.cx(a[-1], fo)
        for j in range(w):
            c.cx(fo, a[j]); c.cx(fo, b[j])       # (a, b) <- ~(a, b) ~ -(a, b)
        domega = cordic_vector(c, a, b, I)
        # tau = atan2(sqrt(2/3) zeta, r): vectoring on (K r, sqrt(2/3) K zeta)
        y = const_mult(c, zeta, G_TAU * K, F)
        x = c.alloc(w)
        copy_view(c, a, x)
        dtau = cordic_vector(c, x, y, I)

        # zero-mode flag
        kk = self.k1 + self.k2
        for q in kk:
            c.x(q)
        acc = kk[0]
        for q in kk[1:]:
            t = c.alloc(1)[0]
            c.ccx(acc, q, t)
            acc = t
        for q in kk:
            c.x(q)
        self.flag0 = acc
        self.fomega, self.domega, self.dtau = fo, domega, dtau
        self.n_arith = len(c.gates)
        self.arith_counts = c.counts()
        c.gates += c.gates[::-1]                 # uncompute: exact reverse
        self.n_wires = c.n

    # -- local gate list (controlled on work bits) ----------------------------
    def local_gates(self):
        """[(matrix16, [(wire, value), ...]), ...] in application order."""
        al = np.arctan(2.0 ** -np.arange(self.I))
        A = al.sum()
        X2 = np.array([[0, 1], [1, 0]])
        g = [(np.kron(X2, np.eye(8)), [(self.flag0, 1)])]   # k = 0: |0> -> |8>
        # Rtau(tau), tau = A - 2 sum d_i al_i
        g.append((Rtau(A), []))
        for i, d in enumerate(self.dtau):
            g.append((Rtau(-2 * al[i]), [(d, 1)]))
        g.append((Rsplit(), []))
        # Romega(omega), omega = pi f + A - 2 sum d_i al_i
        g.append((Romega(A), []))
        for i, d in enumerate(self.domega):
            g.append((Romega(-2 * al[i]), [(d, 1)]))
        g.append((Romega(np.pi), [(self.fomega, 1)]))
        g.append((T_matrix(), []))
        return g

    # -- simulation -----------------------------------------------------------
    def simulate(self):
        """Run compute, read controls, apply local gates, run uncompute.

        Returns Q (N^2, 16, 16) including the phase e^{i(phi1+phi2)}, and a
        dict of checks.
        """
        N, n = self.N, self.n
        kk = np.arange(N * N)
        k1, k2 = kk // N, kk % N
        sim = BitSim(self.n_wires, N * N)
        sim.set_int(self.k1, k1)
        sim.set_int(self.k2, k2)
        sim.run(self.c.gates[:self.n_arith])
        Q = np.broadcast_to(np.eye(16, dtype=complex), (N * N, 16, 16)).copy()
        bits = {}
        for M, ctrl in self.local_gates():
            mask = np.ones(N * N, bool)
            for wire, val in ctrl:
                if wire not in bits:
                    bits[wire] = sim.get_bits(wire)
                mask &= bits[wire] == val
            Q[mask] = M @ Q[mask]
        ph = np.ones(N * N, complex)
        for j in range(n):
            ph *= np.where((k1 >> j) & 1, np.exp(1j * np.pi * 2 ** j / N), 1)
            ph *= np.where((k2 >> j) & 1, np.exp(1j * np.pi * 2 ** j / N), 1)
        Q *= ph[:, None, None]
        al = np.arctan(2.0 ** -np.arange(self.I))
        self.decoded = dict(
            omega=np.pi * sim.get_bits(self.fomega) + sum(
                (1 - 2 * sim.get_bits(d)) * a for d, a in zip(self.domega, al)),
            tau=sum((1 - 2 * sim.get_bits(d)) * a for d, a in zip(self.dtau, al)))
        sim.run(self.c.gates[self.n_arith:])
        work = [q for q in range(self.n_wires) if q not in self.k1 + self.k2]
        dirty = int(sum(sim.get_bits(q).sum() for q in work))
        restored = bool(np.all(sim.get_int(self.k1, False) == k1) and
                        np.all(sim.get_int(self.k2, False) == k2))
        return Q, dict(work_bits_left_set=dirty, inputs_restored=restored)

    # -- resources -------------------------------------------------------------
    def resources(self):
        ar = self.arith_counts
        nI = self.I
        rot = {"Ry on b1, ctrl d and b3=0 (tau)": nI,
               "Ry on b0, ctrl d and b3=b1=0 (omega)": nI + 1}
        fixed = {"uncontrolled Ry (tau, omega offsets)": 2,
                 "Rsplit: Ry(pi/4) on b0, ctrl b3=0, b1=1": 1,
                 "T: fixed 3-qubit unitary on b3=0": 1,
                 "zero mode: CX from flag to b3": 1}
        return dict(N=self.N, F=self.F, I=self.I, width=self.w,
                    wires=self.n_wires + 4, peak_live_wires=self.c.peak_live + 4,
                    toffoli_compute=ar["ccx"], toffoli_total=2 * ar["ccx"],
                    cnot_total=2 * ar["cx"], x_total=2 * ar["x"],
                    controlled_rotations=rot, fixed_local_ops=fixed,
                    phase_gates=2 * self.n)


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
