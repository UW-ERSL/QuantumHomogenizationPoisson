"""fourier_symbol -- the Fourier symbol of the homogeneous periodic Q4 conduction cell.

The unit-conductivity stiffness matrix K of the periodic N x N Q4 mesh is
circulant, so the spatial DFT diagonalizes it, with one scalar per mode,

    Shat(k) = a1 + a2,
    a1 = (2/3)(1 - cos t1)(2 + cos t2),   a2 = (2/3)(1 - cos t2)(2 + cos t1),

with t_j = 2 pi k_j / N. These are the eigenvalues of K1 = circ(-1,2,-1) and
M1 = (1/6) circ(1,4,1) substituted into K = K1 (x) M1 + M1 (x) K1. The symbol
is zero only at k = 0, the constant field; inverse and inverse square root are
set to zero there (Moore-Penrose pseudo-inverse on the zero-mean subspace).

    from quantuminversionpoisson.fourier_symbol import symbol, apply_mode_function
    S = symbol(N)                                   # (N, N)
    y = apply_mode_function(S ** -0.5, x)           # (K^+)^{1/2} x, by FFT

Node index i N + j, with i along x.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


def symbol(N: int) -> np.ndarray:
    """Shat(k1, k2) = a1 + a2. Shape (N, N), real, nonnegative."""
    t = 2 * np.pi * np.arange(N) / N
    t1, t2 = t[:, None], t[None, :]
    return (2 / 3) * ((1 - np.cos(t1)) * (2 + np.cos(t2))
                      + (1 - np.cos(t2)) * (2 + np.cos(t1)))


def symbol_function(f, N: int) -> np.ndarray:
    """f(Shat) at every mode, zero at the zero mode."""
    S = symbol(N)
    ok = S > TOL * S.max()
    return np.where(ok, f(np.where(ok, S, 1.0)), 0.0)


def apply_mode_function(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the mode-wise scalar operator T (N, N) to a vector or to columns."""
    N = T.shape[0]
    X = np.fft.fft2(v.reshape(N, N, -1), axes=(0, 1))
    out = np.fft.ifft2(T[..., None] * X, axes=(0, 1))
    return out.reshape(v.shape) if np.iscomplexobj(v) else out.real.reshape(v.shape)


def q4_element() -> np.ndarray:
    """Unit-conductivity Q4 element on a square, 2 x 2 Gauss (h cancels in 2D)."""
    g = 1 / np.sqrt(3)
    xi_a = np.array([-1, 1, 1, -1]); eta_a = np.array([-1, -1, 1, 1])
    Ke = np.zeros((4, 4))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = np.vstack([xi_a * (1 + eta_a * eta), eta_a * (1 + xi_a * xi)]) / 4
            Ke += (2 * dN).T @ (2 * dN) / 4          # J = h/2 I, det J = h^2/4
    return Ke


def reference(N: int) -> np.ndarray:
    """K assembled element by element, independently of the symbol."""
    K = np.zeros((N * N, N * N))
    Ke = q4_element()
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i in range(N):
        for j in range(N):
            nodes = [((i + a) % N) * N + (j + b) % N for a, b in corners]
            K[np.ix_(nodes, nodes)] += Ke
    return K


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"{'N':>4} {'sym vs assembly':>16} {'(S^-1/2)^2 S - P':>17} {'min nonzero / 2(1-cos 2pi/N)':>30}")
    for N in (4, 8, 16, 32):
        K = reference(N)
        S = symbol(N)
        Kd = apply_mode_function(S, np.eye(N * N))
        Sh = symbol_function(lambda s: s ** -0.5, N)
        P = Sh ** 2 * S
        P[0, 0] = 1.0
        smin = S[S > 1e-12].min() / (2 * (1 - np.cos(2 * np.pi / N)))
        print(f"{N:>4} {np.abs(Kd - K).max():>16.2e} {np.abs(P - 1).max():>17.2e} {smin:>30.12f}")
    print("---------------------------------------------------------------")
