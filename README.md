# QuantumInversionPoisson

Fast-inversion block encoding of preconditioned periodic heat conduction with
mesh-independent condition number.

For a periodic two-phase Q4 conduction cell with conductivity contrast `rho`,
the operator preconditioned by the homogeneous operator `K`,

    M = (K^+)^{1/2} K^chi (K^+)^{1/2} / sigma_max,

factors as `V^T Sigmatilde V`, where `V = W^{1/2} G (K^+)^{1/2}` is a partial
isometry and `Sigmatilde` is diagonal. Because `K` is circulant, `V` is a
quantum Fourier transform pair around one unit vector of length 8 per
wavenumber. The resulting block encoding of `M` has subnormalization 1 and
condition number at most `rho` for every mesh size `N`. This package builds
the wavenumber-controlled stage of that circuit at gate level and verifies it.

The two-phase operator itself, and the O(1) oracle for a square inclusion, come
from the companion package [`pyblockencode`](https://github.com/UW-ERSL/PyBlockEncode)
(`POISSON2D_2PHASE`, subnormalization 16/3 for `K^chi / sigma_max`).

## Install

    pip install git+https://github.com/UW-ERSL/QuantumInversionPoisson

or, from a clone, `pip install -r requirements.txt`. Only numpy and scipy are
required; Qiskit and Qiskit Aer are needed only for the optional cross-checks,
and pyqsp only for `verify_qiskit_qsvt.py`.

## Quick start

```python
from quantuminversionpoisson import IsometryCircuit, vhat_exact
import numpy as np

circ = IsometryCircuit(n=4, F=24)                 # N = 16, 24 fixed-point bits
Q, checks = circ.simulate()                       # every wavenumber, gate by gate
N = circ.N
err = max(np.linalg.norm(Q[k1 * N + k2, :8, 0] - vhat_exact(N, k1, k2))
          for k1 in range(N) for k2 in range(N))
print(checks, err, circ.resources()["toffoli_total"])
```

    python -m quantuminversionpoisson.fourier_symbol   # symbol against assembly, seconds
    python verify_isometry.py          # circuit checks, about 15 seconds
    python verify_isometry.py --aer    # adds the Qiskit Aer cross-check, about 4 minutes
    python verify_square.py            # Sec. 4 numerics, about 4 minutes
    python verify_full_unitary.py      # full-unitary error of U_V, about 6 minutes
    python verify_qsvt_degree.py       # QSVT degree and sup norm, a few seconds
    python verify_inverse.py           # preconditioned inverse on four sources, about 1 minute
    python verify_readout.py           # Sec. 4: readout of the effective conductivity, about 3 minutes
    python verify_qiskit_qsvt.py       # the readout as one Qiskit circuit with QSVT phases, about 20 seconds
    python makeTables.py               # Tables 1-3 of the paper as LaTeX in tables/

## Layout

| File | Contents |
|---|---|
| `quantuminversionpoisson/fourier_symbol.py` | Closed-form scalar symbol of `K`, functions of `K` by FFT, `K` assembled element by element |
| `quantuminversionpoisson/revcirc.py` | X/CX/CCX reversible fixed-point arithmetic (Cuccaro adders, CORDIC), bit-packed simulator; unchanged from the elasticity package |
| `quantuminversionpoisson/isometry_circuit.py` | Closed-form factorization of `Vhat(k)` into two angles; the gate-level circuit `U_V`; dense reference `V` |
| `verify_isometry.py` | Arithmetic, factorization, circuit, and scaling checks |
| `verify_isometry_aer.py` | The exported circuit in Qiskit Aer (MPS) for every wavenumber at N = 4 |
| `verify_square.py` | Condition numbers of `K^chi` and of the preconditioned operator, and the encoding of `M`, for a square inclusion |
| `verify_full_unitary.py` | Full 16 x 16 error of the wavenumber-controlled stage against the exact-angle unitary (42 cases) |
| `verify_inverse.py` | The preconditioned inverse applied to four heat sources, against the finite element solution in the energy norm, with FFT-PCG iteration counts |
| `verify_readout.py` | Sec. 4: the quadratic form for the effective conductivity against the finite element cell solution; a statevector emulation of the readout circuit `Z_ab` on the gate-level `U_V`; amplitude estimation sampled from its exact output distribution, against the error and query count of Prop. readout-cost |
| `verify_qiskit_qsvt.py` | The readout circuit of Sec. 4 built entirely in Qiskit at N = 4, rho = 3: `U_V` (QFT, multiplexed `Q(k)`), `U_Sigma`, `U_M`, the QSVT sequence with pyqsp phases for `q_t`, flags; blocks checked against `Mtilde` and `q_t(Mtilde)/alpha_q`, and the all-zero probability sampled in Aer |
| `verify_qsvt_degree.py` | Degree and sup norm of the odd QSVT inverse polynomial against the bounds of Sec. 3.5 (independent of the PDE) |
| `makeTables.py` | Writes the `tabular` of each table in Sec. 4 to `tables/*.tex`, after asserting every claim the text makes about it |

## Factorization

With `phi_j = pi k_j / N`, `a = sin phi1 cos phi2`, `b = cos phi1 sin phi2`,
`zeta = sin phi1 sin phi2` and `r = (a^2 + b^2)^{1/2}`,

    Vhat(k) = e^{i(phi1 + phi2)} T . Romega(omega) . Rsplit . Rtau(tau) |0>,

    omega = atan2(b, a),   tau = atan2(sqrt(2/3) zeta, r),

where `Rtau` rotates the planes (|0>,|2>) and (|1>,|3>), `Rsplit` rotates
(|2>,|3>) by pi/4, `Romega` rotates (|0>,|1>), and `T` is a fixed unitary
whose first four columns are the mean gradients and the two hourglass
patterns. All three rotations act only on states with b3 = 0, so `Q(0)` maps
|0> to |8> exactly. The two angles come from two CORDIC rotations of
`pi (k1 +- k2) / N` and two CORDIC vectorings; the decision bits control the
rotations directly, and the arithmetic is then reversed so that every work
wire returns to zero.

## What is verified

- Symbol against the assembled `K` to 4.4e-16, N = 4 ... 32; smallest nonzero
  eigenvalue `2 (1 - cos(2 pi / N))`, attained at k = (1, 0).
- Adders exhaustively; CORDIC to its truncation error.
- Closed-form factorization to 1.1e-15 for every `k` at `N` = 8 and 64.
- Gate-level simulation for every `k`, `N` = 4 ... 256, `F` = 12, 16, ..., 32
  (42 cases): work register clean and inputs restored in every case; error
  `delta <= 6 N 2^-F` (largest measured `delta 2^F / N` = 5.73, at N = 4,
  from the wavenumber (N/2, N/2)), where `delta` is the full 16 x 16 error
  against the exact-angle unitary, at most 1.26 times the block error.
- Dense `U_V`, with the QFT built from gates, against `V` assembled element by
  element (N = 8, 16), to 1.1e-6 at F = 24.
- The exported 594-qubit circuit in Qiskit Aer (MPS) for every `k` at N = 4,
  agreeing to 6.5e-10.
- The readout of Sec. 4 as one 15-qubit Qiskit circuit (N = 4, rho = 3,
  eps_p = 1e-2, d = 15, alpha_q = 3.373): block of `U_M` equals `Mtilde` to
  1.7e-15; QSP phases (pyqsp symmetric-QSP Newton solver, mapped from the Wx
  to the reflection convention) reproduce `q_t/alpha_q` on [-1, 1] to 3.1e-12;
  block of the QSVT circuit equals `q_t(Mtilde)/alpha_q` to 2.5e-12; the
  all-zero amplitude equals `psi^T q_t(Mtilde) psi / alpha_q` and gives
  `sigma^H` = 0.434186 against the finite element 0.434028 (polynomial error
  2.4e-3 < eps_p in the correction); 2e5 Aer shots give 0.4288 +- 0.0074.
  `Q(k)` enters at exact angles; its arithmetic is checked by
  `verify_isometry_aer.py`.
- Square inclusion, `v_f = 1/4`, `N` = 8 ... 64, `rho` = 1.5 ... 1e4:
  `kappa(M) = rho` exactly (the bound is attained), while the condition number
  of `K^chi` seen by QSVT (`alpha = 16/3`) grows by 3.80, 3.96, 3.99 per
  doubling of `N`; the block of `U_V^dag U_Sigma U_V` from the gate-level
  circuit matches `M + sbar Pi_0` to below `2 delta`, and its extreme
  eigenvalues equal `1/rho` and 1 to 13 digits.
- Preconditioned inverse (exact `Mtilde`, and the gate-level block at N = 8, 16):
  relative energy-norm error <= eps_p = 1e-6 for smooth, random, cell-problem
  and worst-case sources, N = 8 ... 64, rho = 10; the worst-case source
  attains the residual 8.69e-7 of the polynomial; the gate-level block changes
  the errors by at most 1.2e-7 (F = 24) and 2.8e-9 (F = 32). FFT-PCG reaches
  the same error in 9 to 10 iterations, against d = 145 QSVT queries.
- Effective conductivity (Sec. 4), square inclusion, `v_f = 1/4`:
  `sigma^H / sigma_max = sbar - psi^T Mtilde^{-1} psi`, `psi = V^T Sigmatilde ehat`,
  agrees with the finite element cell solution to 1.6e-10 (relative) for
  `N` = 8 ... 64 and `rho` = 1.5 ... 1e4; `V^T ehat = 0` to 8.3e-17; the
  polynomial changes the correction term by at most 0.40 eps_p of it;
  `sigma^V / sigma^H` = 2.10 (`rho` = 10) and 1442 (`rho` = 1e4) at N = 64.
- Readout circuit `Z_ab = (U_psi,a)^dag U_q U_psi,b`, emulated on the
  statevector with `U_V` from the gates (N = 4, 8; F = 24, 32), flag qubits on
  the local register, separate `U_Sigma` ancillas, and `U_q` a unitary with
  block `q_t(Mtilde_c)/C` and a random unitary elsewhere: the all-zero
  amplitude equals the product of the blocks to 1.7e-17 and the exact value
  to 1.4e-10 (F = 24); with one shared ancilla and no flags it is wrong
  (1.78e-2 against 9.43e-3 at N = 4).
- Amplitude estimation on the Hadamard test (sampled exactly, 2000 runs,
  seed 1), rho = 10: with eps_p = eps/3, F from `6 N 2^-F`, and
  `M = ceil(9 pi C / (eps sbar))`, the error in `sigma^H / sigma_max` is at
  most `eps sbar` in 96.7% to 100% of runs and in every median of 5 runs;
  at eps = 1e-2: d = 63, C = 12.08, M = 105,113, (2M - 1) d = 1.32e7 queries
  to `U_M`.
- QSVT inverse polynomial (odd, Gribling et al. Cor. 12): degree
  `d <= rho ln(2/eps_p) + 1`; sup norm 1.30 rho (eps_p = 1e-3) and
  1.76 rho (eps_p = 1e-6).

## Cost per application of U_V

Toffoli ~ 41 F^2 + 168 F + 117 + 4 log2 N, within 0.54% for 12 <= F <= 36
(the CSD digits of the constant sqrt(2/3) depend on F, so the count is not an
exact quadratic), with `F = ceil(log2(6 N / delta))` for full-unitary error
`delta`; 2F + 1 controlled single-qubit rotations. For N = 1024 and
delta = 1e-6: F = 33, 50,430 Toffolis (circuit count), 3,370 qubits.

## What is NOT established

1. **Rotation synthesis:** the rotations are simulated exactly; Clifford+T
   synthesis is not included.
2. **Qubit count:** intermediate CORDIC values are kept until the final
   uncomputation; pebbling is not applied.
3. **Error constant:** `delta <= 6 N 2^-F` is measured in every case tested;
   the paper proves `delta <= C N 2^-F` with an unspecified absolute constant.
4. **QSVT phases at scale:** the phase factors are computed and run in
   Qiskit only at N = 4, rho = 3 (d = 15), with `Q(k)` at exact angles; the
   statevector emulation in `verify_readout.py` (N = 4, 8, rho = 10) uses a
   unitary with the block `q_t(Mtilde)/alpha_q` in place of the phase sequence.
5. **Readout of the full field:** only the effective conductivity is read
   out; the temperature field is not.

## Conventions

Scalar temperature, one unknown per node, `N_dof = N^2`. The DFT has entries
`N^-1 e^{-2 pi i k.j/N}`. Local index `2 g + c` for Gauss point `g` and
gradient component `c` (x, y); node and element `(i, j)` have index
`i N + j`, with `i` along `x`. `W = w I`, `w = h^2 / 4`.
