"""verify_lsq_readout -- least-squares readout of the effective conductivity.

    sigma^H / sigma_max = g^T (I - Pi) g,   g = Sigma^{1/2} ehat,   Pi = projector onto range(Vt),
    Vt = Sigma^{1/2} V   (singular values in [rho^{-1/2}, 1]).

Circuit (state-vector emulation, U_V from the gates, Gidney adders):
    PREP ehat -> U_Sigma^{1/2} (s_a) -> QSVT, even degree 2t, polynomial f(x) = r_t(x^2) on U_Vt^dag
    -> U_Sigma^{1/2} (s_c) -> overlap with ehat.
r_t is the degree-t Chebyshev polynomial on [1/rho, 1] normalized to r_t(0) = 1, so f(0) = 1,
|f| <= 1 on [-1, 1], f ~ 0 on [rho^{-1/2}, 1] and the block of the QSVT circuit is f(Vt) ~ I - Pi
with subnormalization 1.  Each query is one U_V or one U_V^dag.

    python verify_lsq_readout.py       # about 10 seconds; needs pyqsp (symmetric-QSP phases)
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, os.environ.get("QHP_REPO", os.path.dirname(os.path.abspath(__file__))))
import quantuminversionpoisson.revcirc as rc
rc.ADDER = "gidney"
from quantuminversionpoisson import IsometryCircuit, Q_exact, V_assembled
from quantuminversionpoisson.isometry_circuit import assemble_G

RUNS = 20000
Q_ZERO = np.kron(np.array([[0, 1], [1, 0]]), np.eye(8))        # k = 0: |0> -> |8>


# ---- polynomial and phases ----------------------------------------------------------
def cheb_r(t, a):
    z0 = (1 + a) / (1 - a)
    c = np.zeros(t + 1); c[t] = 1
    d0 = C.chebval(z0, c)
    return (lambda y: C.chebval((1 + a - 2 * y) / (1 - a), c) / d0), 1 / d0


def degree_for(a, eps):
    t = 1
    while cheb_r(t, a)[1] > eps:
        t += 1
    return t


def _phases(rho, eps_p, shift):
    """Reflection-convention phases with Re <0|U|0> = f(x); f = r_t(x^2), or the shifted
    f = (r_t(x^2) + e_t) / (1 + e_t) >= 0 with e_t = max_{[1/rho,1]} |r_t|.  Returns (phi, t, f, e_t)."""
    from pyqsp.sym_qsp_opt import newton_solver
    a = 1 / rho
    t = degree_for(a, eps_p)
    d = 2 * t
    r, e = cheb_r(t, a)
    fx = (lambda x: (r(x ** 2) + e) / (1 + e)) if shift else (lambda x: r(x ** 2))
    c = C.chebinterpolate(fx, d)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, seq = newton_solver(c[0::2], 0, crit=1e-13, maxiter=60)
    ph = np.array(seq.full_phases, float)
    ph[0] -= np.pi / 2                                          # Wx convention of the solver
    phi = ph - np.pi / 2                                        # Wx -> reflection convention
    phi[0] += np.pi / 4 + d * np.pi / 2
    phi[-1] += np.pi / 4
    return phi, t, fx, e


def phases(rho, eps_p):
    """Unshifted polynomial; returns (phi, t, r) with r(y) = r_t(y)."""
    phi, t, fx, e = _phases(rho, eps_p, False)
    return phi, t, cheb_r(t, 1 / rho)[0]


def scalar_response(phi, x):
    s = np.sqrt(1 - x * x)
    U = np.diag(np.exp(1j * phi[0] * np.array([1, -1])))
    for f in phi[1:]:
        U = U @ np.array([[x, s], [s, -x]]) @ np.diag(np.exp(1j * f * np.array([1, -1])))
    return U[0, 0]


# ---- emulation ------------------------------------------------------------------------
class Readout:
    """State axes: (position N^2, local 16, s_a, s_b, s_c, z)."""
    PI_G = (slice(None), slice(0, 8), slice(None), 0, slice(None))     # output of U_Vt, z removed
    PI_N = (slice(None), 0, slice(None), 0, slice(None))               # input of U_Vt

    def __init__(self, N, rho, Q):
        self.N, self.rho, self.Q = N, rho, Q
        chi = np.zeros((N, N)); chi[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1
        self.chi = chi.ravel()
        self.D = 2 * self.chi - 1                                       # -R_chi on the element index
        g0, g1 = (1 + rho ** -0.5) / 2, (1 - rho ** -0.5) / 2           # Sigma^{1/2} = g0 I + g1 (-R_chi)
        self.R = np.array([[np.sqrt(g0), -np.sqrt(g1)], [np.sqrt(g1), np.sqrt(g0)]])
        vf = self.chi.mean()
        self.sbar = vf + (1 - vf) / rho
        self.prep = np.zeros((N * N, 16)); self.prep[:, 0:8:2] = 1 / (2 * N)   # ehat, x-components

    def uv(self, s, dag=False):
        N = self.N
        x = s.reshape(N, N, 16, -1)
        x = np.fft.fft2(x, axes=(0, 1), norm="ortho").reshape(N * N, 16, -1)
        Qk = self.Q.conj().transpose(0, 2, 1) if dag else self.Q
        x = np.einsum("kij,kjr->kir", Qk, x).reshape(N, N, 16, -1)
        return np.fft.ifft2(x, axes=(0, 1), norm="ortho").reshape(s.shape)

    def usig(self, s, anc):
        x = np.moveaxis(s, anc, 0)
        y = np.einsum("ij,j...->i...", self.R, x)
        y[1] = y[1] * self.D.reshape(-1, *([1] * (y[1].ndim - 1)))
        y = np.einsum("ji,j...->i...", self.R, y)
        return np.moveaxis(y, 0, anc)

    def utv(self, s):                                                   # U_Vt = U_Sigma . U_V
        return self.usig(self.uv(s), 3)

    def utv_dag(self, s):
        return self.uv(self.usig(s, 3), dag=True)

    @staticmethod
    def refl(s, phi, region):
        """e^{i sgn phi (2P - I)}, sgn = +1 on z = 0 and -1 on z = 1."""
        out = np.empty_like(s)
        for zi, sg in ((0, 1.0), (1, -1.0)):
            x = s[..., zi]
            y = x * np.exp(-1j * sg * phi)
            y[region] = x[region] * np.exp(1j * sg * phi)
            out[..., zi] = y
        return out

    def qsvt(self, s, phi):
        d = len(phi) - 1
        for j in range(d, -1, -1):
            first = (d - j) % 2 == 0
            s = self.refl(s, phi[j], self.PI_G if first else self.PI_N)
            if j > 0:
                s = self.utv_dag(s) if first else self.utv(s)
        return s

    def block(self, v, phi):
        """Block <+|_z <Pi_g| U_Phi |Pi_g> |+>_z applied to a vector v of shape (N^2, 8)."""
        s = np.zeros((self.N ** 2, 16, 2, 2, 2, 2), complex)
        s[:, :8, 0, 0, 0, 0] = v / np.sqrt(2)
        s[:, :8, 0, 0, 0, 1] = v / np.sqrt(2)
        s = self.qsvt(s, phi)
        return ((s[..., 0] + s[..., 1]) / np.sqrt(2))[:, :8, 0, 0, 0]

    def amplitude(self, phi):
        s = np.zeros((self.N ** 2, 16, 2, 2, 2, 2), complex)
        s[:, :, 0, 0, 0, 0] = self.prep
        s = self.usig(s, 2)
        s[..., 1] = s[..., 0] / np.sqrt(2); s[..., 0] = s[..., 0] / np.sqrt(2)
        s = self.qsvt(s, phi)
        s = (s[..., 0] + s[..., 1]) / np.sqrt(2)                      # H on z, project on |0>
        s = self.usig(s, 4)
        return np.vdot(self.prep, s[:, :, 0, 0, 0])


# ---- references ------------------------------------------------------------------------
def references(N, rho):
    chi = np.zeros((N, N)); chi[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1
    s8 = np.repeat(np.where(chi.ravel() == 1, 1.0, 1 / rho), 8)
    A = assemble_G(N) / (2 * N)
    ehat = np.zeros(8 * N * N); ehat[0::2] = 1 / (2 * N)
    sbar = ehat @ (s8 * ehat)
    Kchi = A.T @ (s8[:, None] * A)
    f = A.T @ (s8 * ehat)
    v = -np.linalg.pinv(Kchi, hermitian=True) @ f
    r = ehat + A @ v
    sFE = r @ (s8 * r)
    Vt = np.sqrt(s8)[:, None] * V_assembled(N)
    U, sv, _ = np.linalg.svd(Vt, full_matrices=False)
    nz = sv > 1e-9
    return dict(sFE=sFE, sbar=sbar, g=np.sqrt(s8) * ehat, U=U[:, nz], sv=sv[nz])


def f_apply(ref, r, v):
    """f(Vt) v = v - U U^T v + U r(sv^2) U^T v on the gradient space."""
    c = ref["U"].T @ v
    return v - ref["U"] @ c + ref["U"] @ (r(ref["sv"] ** 2) * c)


def exact_Q(N, circ=None):
    Q = np.zeros((N * N, 16, 16), complex)
    Q[0] = Q_ZERO
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        om = circ.decoded["omega"][kk] if (circ is not None and k1 == k2 == N // 2) else None
        Q[kk] = Q_exact(N, k1, k2, om)
    return Q


def ae_samples(p, M, runs, rng):
    om = np.arcsin(np.sqrt(p)) / np.pi
    y = np.arange(M)

    def qpe(w):
        dl = w - y / M
        s = np.sin(np.pi * dl)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(s) < 1e-15, 1.0, np.sin(M * np.pi * dl) ** 2 / (M ** 2 * s ** 2))
    P = 0.5 * (qpe(om) + qpe(1 - om))
    P /= P.sum()
    return np.sin(np.pi * rng.choice(M, size=runs, p=P) / M) ** 2


# ---- 1. QSVT circuit against the dense polynomial (exact-angle U_V) ---------------------
def part1():
    print("1. QSVT circuit with exact-angle U_V against the dense polynomial f(Vt)")
    print(f"{'N':>3} {'rho':>5} {'eps_p':>7} {'d=2t':>5} {'scalar err':>10} {'block err':>10} "
          f"{'|A-poly|':>9} {'|A-FE|/sbar':>11} {'<= eps_p':>8}")
    rng = np.random.default_rng(3)
    for N, rho, eps_p in ((4, 3.0, 1 / 300), (8, 10.0, 1 / 300), (8, 10.0, 1e-6), (8, 100.0, 1 / 300)):
        phi, t, r = phases(rho, eps_p)
        xs = np.linspace(0, 1, 201)
        sc = max(abs(scalar_response(phi, x).real - r(x * x)) for x in xs)
        ref = references(N, rho)
        emu = Readout(N, rho, exact_Q(N))
        v = rng.standard_normal((N * N, 8)) + 1j * rng.standard_normal((N * N, 8))
        out = emu.block(v, phi)
        want = f_apply(ref, r, v.reshape(-1)).reshape(N * N, 8)
        be = np.abs(out - want).max()
        A = emu.amplitude(phi)
        g = ref["g"]
        Apoly = g @ f_apply(ref, r, g)
        print(f"{N:>3} {rho:>5.0f} {eps_p:>7.0e} {2 * t:>5} {sc:>10.1e} {be:>10.1e} "
              f"{abs(A - Apoly):>9.1e} {abs(A.real - ref['sFE']) / ref['sbar']:>11.1e} "
              f"{'yes' if abs(A.real - ref['sFE']) / ref['sbar'] <= eps_p else 'NO':>8}", flush=True)


# ---- 2. gate-level U_V: encoding error through 2t queries ---------------------------------
def part2():
    print("\n2. Gate-level U_V (Gidney adders): propagation of the encoding error delta")
    print(f"{'N':>3} {'rho':>4} {'F':>3} {'delta':>9} {'6N2^-F':>9} {'|A_gate-A_ex|':>13} "
          f"{'/(2t delta sbar)':>16} {'|A_gate-FE|/sbar':>16}")
    rho, eps_p = 10.0, 1 / 300
    phi, t, r = phases(rho, eps_p)
    for N in (4, 8):
        n = int(np.log2(N))
        for F in (12, 16, 20, 24, 28):
            circ = IsometryCircuit(n=n, F=F)
            Q, chk = circ.simulate()
            assert chk["work_bits_left_set"] == 0 and chk["inputs_restored"]
            Qe = exact_Q(N, circ)
            Qe[0] = Q[0]                       # k = 0: only the first column |0> -> |8> is fixed by V
            delta = max(np.linalg.norm(Q[k] - Qe[k], 2) for k in range(N * N))
            emu_g, emu_e = Readout(N, rho, Q), Readout(N, rho, Qe)
            Ag, Ae = emu_g.amplitude(phi).real, emu_e.amplitude(phi).real
            ref = references(N, rho)
            print(f"{N:>3} {rho:>4.0f} {F:>3} {delta:>9.2e} {6 * N * 2.0 ** -F:>9.2e} "
                  f"{abs(Ag - Ae):>13.2e} {abs(Ag - Ae) / (2 * t * delta * emu_g.sbar):>16.1e} "
                  f"{abs(Ag - ref['sFE']) / emu_g.sbar:>16.2e}", flush=True)
    c4 = IsometryCircuit(n=2, F=16); Qg, _ = c4.simulate()
    rc.ADDER = "cuccaro"
    c4c = IsometryCircuit(n=2, F=16); Qc, _ = c4c.simulate()
    rc.ADDER = "gidney"
    print(f"   Gidney against Cuccaro adders at N = 4, F = 16: max |Q_gidney - Q_cuccaro| = "
          f"{np.abs(Qg - Qc).max():.1e}")


# ---- 3. amplitude estimation --------------------------------------------------------------
def part3():
    print("\n3. Amplitude estimation sampled from its exact output distribution (20000 runs, seed 1)")
    print(f"{'N':>3} {'rho':>4} {'eps':>6} {'d':>4} {'F':>3} {'M':>7} {'queries':>9} {'|Ahat-A|<=pi/M':>14} "
          f"{'err<=eps sbar':>13} {'median5':>8}")
    rho = 10.0
    for N, eps in ((4, 1e-2), (8, 1e-2), (8, 1e-3)):
        eps_p = eps / 3
        phi, t, r = phases(rho, eps_p)
        d = 2 * t
        n = int(np.log2(N))
        F = int(np.ceil(np.log2(6 * N * 3 * d / eps)))
        Q, _ = IsometryCircuit(n=n, F=F).simulate()
        emu = Readout(N, rho, Q)
        A = emu.amplitude(phi).real
        ref = references(N, rho)
        M = int(np.ceil(3 * np.pi / (eps * emu.sbar)))
        rng = np.random.default_rng(1)
        Ahat = np.sqrt(ae_samples(A ** 2, M, RUNS, rng))
        err = np.abs(Ahat - ref["sFE"])
        med = np.median(Ahat.reshape(-1, 5), axis=1)
        print(f"{N:>3} {rho:>4.0f} {eps:>6.0e} {d:>4} {F:>3} {M:>7} {(2 * M - 1) * d:>9.2e} "
              f"{np.mean(np.abs(Ahat - A) <= np.pi / M):>14.3f} {np.mean(err <= eps * emu.sbar):>13.3f} "
              f"{np.mean(np.abs(med - ref['sFE']) <= eps * emu.sbar):>8.3f}", flush=True)



# ---- 5. shifted polynomial: A >= 0, large degree ------------------------------------------
def part5():
    print("\n5. Shifted polynomial f = (r_t + e_t)/(1 + e_t) >= 0 on the gate-level U_V: A >= 0 and s = (1 + e_t) A - e_t sbar")
    print(f"{'N':>3} {'rho':>6} {'d':>4} {'F':>3} {'scalar err':>10} {'A':>10} {'A >= 0':>6} {'s_FE':>10} "
          f"{'(s-s_FE)/sbar':>13} {'<= eps_p':>8}")
    eps_p = 1 / 300
    for N, rho in ((4, 10.0), (8, 100.0), (4, 1e3), (4, 1e4)):
        phi, t, fx, e = _phases(rho, eps_p, True)
        d = 2 * t
        xs = np.linspace(0, 1, 301)
        sc = max(abs(scalar_response(phi, x).real - fx(x)) for x in xs)
        F = int(np.ceil(np.log2(6 * N * 3 * d * (1 + e) / (3 * eps_p))))
        Q, chk = IsometryCircuit(n=int(np.log2(N)), F=F).simulate()
        assert chk["work_bits_left_set"] == 0 and chk["inputs_restored"]
        emu = Readout(N, rho, Q)
        A = emu.amplitude(phi)
        ref = references(N, rho)
        s_est = (1 + e) * A.real - e * emu.sbar
        err = (s_est - ref["sFE"]) / emu.sbar
        print(f"{N:>3} {rho:>6.0e} {d:>4} {F:>3} {sc:>10.1e} {A.real:>10.3e} {'yes' if A.real >= 0 else 'NO':>6} "
              f"{ref['sFE']:>10.3e} {err:>13.2e} {'yes' if abs(err) <= eps_p else 'NO':>8}", flush=True)

# ---- 4. resources at N = 1024 ---------------------------------------------------------------
def part4():
    print("\n4. Logical resources of one readout, square, N = 1024, rho = 10 (Gidney adders)")
    rho, eps, N = 10.0, 1e-2, 1024
    vf = 0.25
    sbar = vf + (1 - vf) / rho
    phi, t, r = phases(rho, eps / 3)
    d = 2 * t
    F = int(np.ceil(np.log2(6 * N * 3 * d / eps)))
    M = int(np.ceil(3 * np.pi / (eps * sbar)))
    res = IsometryCircuit(n=10, F=F).resources()
    per_q = res["toffoli_total"] + 2                                # one U_V and one controlled R_chi
    tot = (2 * M - 1) * d * per_q
    old = 4.4e6 * 51926
    print(f"   d = {d}, F = {F}, M = {M}, queries = {(2 * M - 1) * d:.3e}, Toffoli per query = {per_q:,}, "
          f"qubits = {res['wires'] + 5}")
    print(f"   total Toffoli = {tot:.3e}  (current paper text: {old:.3e}; ratio {old / tot:.0f})")


if __name__ == "__main__":
    t0 = time.time()
    print("----- output --------------------------------------------------")
    part1(); part2(); part3(); part4(); part5()
    print(f"---------------------------------------------------------------  [{time.time() - t0:.0f} s]")
