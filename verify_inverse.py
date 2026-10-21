"""verify_inverse -- the preconditioned inverse applied to specific heat sources (Sec. 4.4).

    python verify_inverse.py           # about 2 minutes

u_Q = sigma_max^{-1} (K^+)^{1/2} q_t(Mtilde) (K^+)^{1/2} f, with q_t the odd
QSVT polynomial of Sec. 3.7, is compared with the finite element solution
u = (K^chi)^+ f from a sparse direct solve, in the energy norm of K^chi.
Eq. (energy-error) bounds the relative error by eps_p for every source with
zero mean. Square inclusion, v_f = 1/4, sigma_max = 1.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from verify_square import assemble, square_chi, circuit_block
from quantuminversionpoisson.fourier_symbol import symbol_function, apply_mode_function


def k_function(f, N):
    """x -> f(K) x by FFT, zero on the constant field."""
    T = symbol_function(f, N)
    return lambda x: apply_mode_function(T, x)


# ---- odd QSVT polynomial q_t at kappa = rho ---------------------------------
def q_poly(rho, eps):
    a = 1 / rho ** 2
    s = (1 + a) / (1 - a)
    t = int(np.ceil(0.5 * rho * np.log(2 / eps)))
    Ts = np.cosh(t * np.arccosh(s))

    def q(x):
        z = (1 + a - 2 * x ** 2) / (1 - a)
        Tz = np.where(np.abs(z) <= 1, np.cos(t * np.arccos(np.clip(z, -1, 1))),
                      np.cosh(t * np.arccosh(np.maximum(np.abs(z), 1))))
        return (1 - Tz / Ts) / x
    return q, 2 * t - 1


# ---- sources, solutions, errors ---------------------------------------------
def zero_mean(f):
    return f - f.mean()


def sources(N, A, St, Kh, lam, U):
    n = N * N
    i, j = np.divmod(np.arange(n), N)
    x, y = i / N, j / N
    wd = np.tile(np.array([1.0, 0.0]) / (2 * N), 4 * n)        # sqrt(w) e_x at every Gauss point
    nz = lam > 1e-9
    f = {"smooth": np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
         "random": np.random.default_rng(0).standard_normal(n),
         "cell": -A.T @ (St * wd),                              # macroscopic gradient (1, 0)
         "worst": Kh(U[:, nz][:, np.argmin(lam[nz])])}          # eigenvector at 1/rho
    return {k: zero_mean(v) for k, v in f.items()}


def fe_solve(Kc, f):
    keep = np.arange(1, Kc.shape[0])                            # pin node 0, then remove the mean
    u = np.zeros_like(f)
    u[keep] = spla.spsolve(Kc[keep][:, keep].tocsc(), f[keep])
    return zero_mean(u)


def energy_error(Kc, u, x):
    e = x - u
    return np.sqrt(e @ (Kc @ e) / (u @ (Kc @ u)))


def apply_inverse(q, Kih, lam, U, sbar, f):
    lt = np.where(np.abs(lam) > 1e-9, lam, sbar)                # spectrum of Mtilde
    return Kih(U @ (q(lt) * (U.conj().T @ Kih(f))))


LOADS = ("smooth", "random", "cell", "worst")


def inverse_case(N, rho, eps, F_bits=()):
    """Energy-norm errors of u_Q for the four sources, the QSVT degree d, and
    the errors with Mtilde from the gate-level circuit for each F in F_bits."""
    q, d = q_poly(rho, eps)
    A = assemble(N).tocsr()
    chi = square_chi(N)
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    Kc = (A.T @ sp.diags(St) @ A).tocsr()                       # K^chi / sigma_max
    Kih, Kh = (k_function(g, N) for g in (lambda e: e ** -0.5, np.sqrt))
    M = Kih(Kc @ Kih(np.eye(N * N)))
    lam, U = np.linalg.eigh((M + M.T) / 2)
    sbar = chi.mean() + (1 - chi.mean()) / rho
    F_ = sources(N, A, St, Kh, lam, U)
    sols = {k: fe_solve(Kc, f) for k, f in F_.items()}
    errs = {k: energy_error(Kc, sols[k], apply_inverse(q, Kih, lam, U, sbar, F_[k])) for k in LOADS}
    circ = {}
    for Fb in F_bits:
        Mc, chk = circuit_block(N, rho, Fb)
        lc, Uc = np.linalg.eigh((Mc + Mc.conj().T) / 2)
        circ[Fb] = {k: energy_error(Kc, sols[k], apply_inverse(q, Kih, lc, Uc, sbar,
                                                               F_[k].astype(complex)).real) for k in LOADS}
    return dict(d=d, errs=errs, circuit=circ)


def residual(rho, eps):
    """Largest |x q_t(x) - 1| on [1/rho, 1], that is, 1/T_t(s)."""
    a = 1 / rho ** 2
    t = int(np.ceil(0.5 * rho * np.log(2 / eps)))
    return 1 / np.cosh(t * np.arccosh((1 + a) / (1 - a)))


if __name__ == "__main__":
    eps, rho = 1e-6, 10.0
    print("----- output --------------------------------------------------")
    print(f"residual of q_t: {residual(rho, eps):.3e}")
    print(f"{'N':>3} {'F':>4} {'d':>5}  " + "  ".join(f"{k:>9}" for k in LOADS))
    for N in (8, 16, 32, 64):
        r = inverse_case(N, rho, eps, F_bits=(24, 32) if N <= 16 else ())
        print(f"{N:3d} {'-':>4} {r['d']:5d}  " +
              "  ".join(f"{r['errs'][k]:9.2e}" for k in LOADS), flush=True)
        for Fb, e in r["circuit"].items():
            print(f"{N:3d} {Fb:4d} {r['d']:5d}  " +
                  "  ".join(f"{e[k]:9.2e}" for k in LOADS), flush=True)
    print("---------------------------------------------------------------")
