# QuantumHomogenizationPoisson

Code for the paper *A Preconditioned Quantum Algorithm for Periodic
Homogenization of Two-Phase Conduction* (K. Suresh). Every number the paper
reports is reproduced by the scripts below: the numerical experiments of
Sec. 5 by `experiments.py`, the circuit figures and checks of Secs. 3-4 by
`makeFigures.py` and the `verify_*.py` scripts.

For a periodic two-phase Q4 conduction cell with conductivity contrast `rho`,
the cell-problem operator preconditioned by the homogeneous operator `K`,

    M = (K^+)^{1/2} K^chi (K^+)^{1/2} / sigma_max,

factors as `V^T Sigmatilde V`, where `V = W^{1/2} G (K^+)^{1/2}` is a partial
isometry and `Sigmatilde` is diagonal. Because `K` is circulant, `V` is a
quantum Fourier transform pair around one unit vector of length 8 per
wavenumber. The resulting block encoding of `M` has subnormalization 1 and
condition number at most `rho` for every mesh size `N`. The effective
conductivity is the Voigt bound minus the quadratic form
`psi^T Mtilde^{-1} psi`, read out as one amplitude by amplitude estimation.

This repository builds the wavenumber-controlled stage of `U_V` at gate level,
the readout circuit (statevector emulation, and one Qiskit circuit with QSVT
phases), verifies both against the finite element solution of the cell
problem, and runs the numerical experiments of the paper on three dyadic cells
(square, cross, laminate) up to N = 1024. The Python package keeps its
original name, `quantuminversionpoisson`.

The baseline condition number of Sec. 5.2 uses the subnormalization
`alpha = 16/3` of the encoding of `K^chi / sigma_max` in the companion package
[`pyblockencode`](https://github.com/UW-ERSL/PyBlockEncode) (`POISSON2D_2PHASE`).
The value is set in `verify_square.py` and `experiments.py`; when
pyblockencode is installed, `verify_square.py` cross-checks it.

## Install

    git clone https://github.com/UW-ERSL/QuantumHomogenizationPoisson
    cd QuantumHomogenizationPoisson
    pip install -r requirements.txt

Only numpy and scipy are required for the package; matplotlib for
`makeFigures.py`; pyblockencode (optional, not in `requirements.txt`) for the
cross-check of `alpha` in `verify_square.py`; Qiskit and Qiskit Aer for the Aer cross-checks; and pyqsp
for `verify_qiskit_qsvt.py`. On Debian or Ubuntu system Python, pyqsp fails to
build (`AttributeError: install_layout` from the system setuptools); install
in a virtual environment:

    python -m venv venv && . venv/bin/activate
    pip install -U setuptools wheel && pip install -r requirements.txt

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

    python -m quantuminversionpoisson.fourier_symbol   # Sec. 2.2: symbol against assembly, seconds
    python verify_isometry.py          # Secs. 3.3-3.4: circuit checks, about 15 seconds
    python verify_isometry.py --aer    # adds the Qiskit Aer cross-check, about 4 minutes
    python verify_full_unitary.py      # Sec. 3.4: full-unitary error of U_V, about 6 minutes
    python verify_qsvt_degree.py       # Sec. 3.6: QSVT degree and sup norm, a few seconds
    python verify_square.py            # Sec. 3.5: condition numbers and gate-level encoding of M, about 4 minutes
    python verify_inverse.py           # check: preconditioned inverse on four sources, about 1 minute
    python verify_readout.py           # Sec. 4: readout of the effective conductivity, about 2 minutes
    python verify_qiskit_qsvt.py       # Sec. 4.2: the readout as one Qiskit circuit with QSVT phases, about 20 seconds
    python experiments.py              # Sec. 5: all five experiments, tables/ and figs/, about 6 minutes
    python experiments.py mesh         # one experiment: geometry, mesh, contrast, accuracy, crossover
    python makeFigures.py              # all other figures to figs/, about 6 minutes
    python makeFigures.py circuits     # only the two circuit figures, seconds
    python makeFigures.py isometry     # encoding error and Toffoli count of U_V (Sec. 3.4), about 4 minutes
    python makeTables.py               # checks of the former Sec. 5 (N <= 64) as tables; not in the paper

`tables/`, `figs/` and `experiments.json` are generated, not committed. The
paper uses `figs/spectrum.pdf` (fig:spectrum), `figs/readout_circuit.pdf`
(fig:readout), `figs/qsvt_circuit.pdf` (fig:qsvt) and `figs/isometry_cost.pdf`
(fig:isometry-cost) from `makeFigures.py`, and `figs/cells.pdf`, `mesh.pdf`,
`contrast.pdf`, `accuracy.pdf`, `crossover.pdf` with `tables/geometry.tex`,
`contrast.tex`, `resources.tex` from `experiments.py`; `makeFigures.py` also writes
`figs/qsvt_poly.pdf` and `figs/angles.pdf`, which the current paper does not
use. The circuit figures are drawn from the circuits that `readout()` and
`u_q()` of `verify_qiskit_qsvt.py` build, with `U_V`, `U_M`, `U_q` and
`-R_chi` as labeled blocks; they are written as quantikz
(`figs/*_circuit.tex`) and need pdflatex with the quantikz package. Table 1 of the paper (symbols) is not generated.

## Layout

| File | Contents |
|---|---|
| `quantuminversionpoisson/fourier_symbol.py` | Closed-form scalar symbol of `K` (Eq. symbol-entries), functions of `K` by FFT, `K` assembled element by element |
| `quantuminversionpoisson/revcirc.py` | X/CX/CCX reversible fixed-point arithmetic (Cuccaro adders, CORDIC), bit-packed simulator; unchanged from the elasticity package |
| `quantuminversionpoisson/isometry_circuit.py` | Closed-form factorization of `Vhat(k)` into two angles; the gate-level circuit `U_V`; dense reference `V` |
| `verify_isometry.py` | Arithmetic, factorization, circuit, and scaling checks |
| `verify_isometry_aer.py` | The exported circuit in Qiskit Aer (MPS) for every wavenumber at N = 4 |
| `verify_full_unitary.py` | Full 16 x 16 error of the wavenumber-controlled stage against the exact-angle unitary (42 cases), and the Toffoli count |
| `verify_qsvt_degree.py` | Degree and sup norm of the odd QSVT inverse polynomial against the bounds of Sec. 3.6 (independent of the PDE) |
| `verify_square.py` | Condition numbers of `K^chi`, of a separately encoded preconditioner, and of the preconditioned operator; the encoding of `Mtilde` from the gate-level circuit; square inclusion |
| `verify_inverse.py` | The preconditioned inverse applied to four heat sources, against the finite element solution in the energy norm, with FFT-PCG iteration counts |
| `verify_readout.py` | The quadratic form for the effective conductivity (Eq. sigmaH-pre) against the finite element cell solution; a statevector emulation of the readout circuit (Fig. readout) on the gate-level `U_V`; amplitude estimation sampled from its exact output distribution, against Eqs. readout-error and readout-queries |
| `verify_qiskit_qsvt.py` | The readout circuit built entirely in Qiskit at N = 4, rho = 3: `U_V` (QFT, multiplexed `Q(k)`), `U_Sigma`, `U_M`, the QSVT sequence with pyqsp phases for `q_t`, flags; blocks checked against `Mtilde` and `q_t(Mtilde)/alpha_q`, and the all-zero probability sampled in Aer |
| `experiments.py` | The five experiments of Sec. 5 on the square, cross and laminate, with operators applied matrix-free up to N = 1024; the `R_chi` gate networks, checked on every element; writes `experiments.json`, `tables/*.tex` and `figs/*.pdf`, after asserting every number Sec. 5 quotes |
| `makeTables.py` | Tables of the checks that formed Sec. 5 before 2026-10-13 (N <= 64), with the claims asserted; not used by the current paper |
| `makeFigures.py` | Writes the spectrum figure of Sec. 2.2, the isometry-cost figure of Sec. 3.4, and the readout and QSVT circuit figures of Secs. 4.2 and 3.6 to `figs/`, after asserting every number the text quotes about them (15 qubits; d queries to `U_M`; `delta <= 6 N 2^-F` in 42 cases, largest 5.73; Toffoli fit within 0.54%; 50,430 Toffolis and 3,370 qubits at N = 1024, F = 33) |

The scripts write `alpha_q`, the subnormalization of the QSVT polynomial, as
`C`. `verify_readout.py` emulates the readout for both macroscopic gradients
(`Z_ab`, `a, b` in `{1, 2}`); the paper uses the single gradient `e = (1, 0)`,
whose amplitude is `A = Z_11`.

## Factorization (Sec. 3.4)

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

## Implementation checks

**Symbol and isometry (Secs. 2.2, 3.3-3.4).**

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

**Preconditioned operator and its inverse (Secs. 3.5-3.6).**

- Square inclusion, `v_f = 1/4`, `N` = 8 ... 64, `rho` = 1.5 ... 1e4:
  `kappa(Mtilde) = rho` exactly (the bound is attained), while the condition
  number of `K^chi` seen by QSVT (`alpha = 16/3`) grows by 3.80, 3.96, 3.99
  per doubling of `N`, and the lower bound `kappa_P` for a separately encoded
  preconditioner is larger still; the block of `U_V^dag U_Sigma U_V` from the
  gate-level circuit matches `Mtilde = M + sbar Pi_0` to below `2 delta`, and
  its extreme eigenvalues equal `1/rho` and 1 to 13 digits.
- QSVT inverse polynomial (odd, Gribling et al. Cor. 12): degree
  `d <= rho ln(2/eps_p) + 1`; sup norm 1.30 rho (eps_p = 1e-3) and
  1.76 rho (eps_p = 1e-6).
- Preconditioned inverse (exact `Mtilde`, and the gate-level block at N = 8, 16):
  relative energy-norm error <= eps_p = 1e-6 for smooth, random, cell-problem
  and worst-case sources, N = 8 ... 64, rho = 10; the worst-case source
  attains the residual 8.69e-7 of the polynomial; the gate-level block changes
  the errors by at most 1.2e-7 (F = 24) and 2.8e-9 (F = 32). FFT-PCG reaches
  the same error in 9 to 10 iterations, against d = 145 QSVT queries.

**Readout of the effective conductivity (Sec. 4).**

- Quadratic form, square inclusion, `v_f = 1/4`:
  `sigma^H / sigma_max = sbar - psi^T Mtilde^{-1} psi`, `psi = V^T Sigmatilde ehat`,
  agrees with the finite element cell solution to 1.6e-10 (relative) for
  `N` = 8 ... 64 and `rho` = 1.5 ... 1e4; `V^T ehat = 0` to 8.3e-17; the
  polynomial changes the correction term by at most 0.40 eps_p of it;
  `sigma^V / sigma^H` = 2.10 (`rho` = 10) and 1442 (`rho` = 1e4) at N = 64.
- Readout circuit, emulated on the statevector with `U_V` from the gates
  (N = 4, 8; F = 24, 32), flag qubits on the local register, separate
  `U_Sigma` ancillas for the forward and reverse preparations, and `U_q` a
  unitary with block `q_t(Mtilde)/alpha_q` and a random unitary elsewhere: the
  all-zero amplitude equals the product of the blocks to 1.7e-17 and the exact
  value to 1.4e-10 (F = 24); with one shared ancilla and no flags it is wrong
  (1.78e-2 against 9.43e-3 at N = 4).
- Amplitude estimation on the all-zero probability `A^2` of the readout
  circuit (`A >= 0`, no controlled copy; sampled exactly, 2000 runs, seed 1),
  rho = 10: with eps_p = eps/3, F from `6 N 2^-F`, and
  `M = ceil(3 pi alpha_q / (eps sbar))`, the error in `sigma^H / sigma_max` is
  at most `eps sbar` in 94.5% to 99.4% of runs and in every median of 5 runs;
  at eps = 1e-2 (Example 4): d = 63, alpha_q = 12.08, sbar = 0.325,
  M = 35,038, (2M - 1) d = 4.41e6 queries to `U_M`.
- The readout as one 15-qubit Qiskit circuit (N = 4, rho = 3, eps_p = 1e-2,
  d = 15, alpha_q = 3.373): block of `U_M` equals `Mtilde` to 1.7e-15; QSP
  phases (pyqsp symmetric-QSP Newton solver, mapped from the Wx to the
  reflection convention) reproduce `q_t/alpha_q` on [-1, 1] to 3.1e-12; block
  of the QSVT circuit equals `q_t(Mtilde)/alpha_q` to 2.5e-12; the all-zero
  amplitude equals `psi^T q_t(Mtilde) psi / alpha_q` and gives
  `sigma^H / sigma_max` = 0.434186 against the finite element 0.434028
  (relative error 2.4e-3 < eps_p in the correction); 2e5 Aer shots give
  0.4288 +- 0.0074 (95%). `Q(k)` enters at exact angles; its arithmetic is
  checked by `verify_isometry_aer.py`.

## Numerical experiments (Sec. 5)

All with operators applied matrix-free (FFT for functions of `K`, sparse
`W^{1/2} G`), N up to 1024 (N_dof ~ 1.05e6), on the square (v_f = 1/4),
the cross (two bands of width N/8, v_f = 15/64, connected) and the laminate
(chi = 1 for x < 1/2, exact sigma^H). `R_chi` costs 2, 9 and 0 Toffolis.

- **Cells (N = 256):** `kappa(Mtilde) = rho` to six digits for every cell at
  rho = 10 and 1e4; quadratic form against FE to 2e-12 or better; laminate
  against Reuss and Voigt to 1e-13; `sigma^V / sigma^H_11` = 2.10, 1.36, 3.03
  (rho = 10) and 1444, 1.81, 2501 (rho = 1e4).
- **Mesh (rho = 10, N = 16 ... 1024):** `kappa(Mtilde) = 10` to ten digits;
  `kappa(K^chi)` grows by 4.00 per doubling to 1.1e6; the error of `q_t`
  (d = 145, eps_p = 1e-6) stays in [4.1e-7, 8.6e-7]; FFT-PCG needs 6 to 9
  iterations (1 for the laminate); Toffolis per query grow from 74,158 to
  106,622.
- **Contrast (N = 1024, eps = 1e-2):** Toffolis per readout 4.7e11 (rho = 10)
  to 1.5e18 (rho = 1e4) relative to Voigt; relative to sigma^H at rho = 1e4,
  9.6e21 for the square and 3.4e18 for the cross.
- **Accuracy (N = 256, rho = 10, 2000 runs, seed 1):** the readout meets the
  bound in at least 94.1% of runs for eps = 1e-1 ... 1e-4; queries grow as
  eps^-1.20. At N = 1024: 4.7e11, 9.2e12, 1.5e14 Toffolis for eps = 1e-2,
  1e-3, 1e-4; 3,545 to 4,833 logical qubits.
- **Crossover (square, rho = 10, eps = 1e-2):** FFT-PCG, 2 iterations,
  about 0.6 s at N = 1024 (Intel Xeon 2.8 GHz, 2 cores); quantum readout
  4.7e11 Toffolis, 925 days at 170 us per Toffoli (Babbush et al.); crossover
  at about 1e14 unknowns (170 us) and 1e12 unknowns (1 us) on that machine
  (1.1e14 and 6e11 to 7e11 in repeated runs); the crossover scales with the
  classical speed of the machine that runs the script.

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
3. **Error constant:** `delta <= 6 N 2^-F` is measured in every case tested
   (Sec. 3.4); the paper states `F = log2 N + log2(1/delta) + O(1)` (Sec. 3.4)
   and does not prove the constant.
4. **QSVT phases at scale:** the phase factors are computed and run in
   Qiskit only at N = 4, rho = 3 (d = 15), with `Q(k)` at exact angles; the
   statevector emulation in `verify_readout.py` (N = 4, 8, rho = 10) uses a
   unitary with the block `q_t(Mtilde)/alpha_q` in place of the phase sequence.
5. **Readout of the full field:** only the effective conductivity is read
   out; the temperature field is not.
6. **Geometry and dimension:** only dyadic cells (square, cross, laminate),
   isotropic phases, and D = 2 are implemented; non-dyadic inclusions need
   material oracles that are not constructed here.
7. **Crossover:** the classical time is measured on one machine and
   extrapolated as N_dof log N_dof beyond N = 1024; the quantum time assumes a
   logical Toffoli time (170 us, Babbush et al.; or 1 us) and excludes
   rotation synthesis.

## Conventions

Scalar temperature, one unknown per node, `N_dof = N^2`. The DFT has entries
`N^-1 e^{-2 pi i k.j/N}`. Local index `2 g + c` for Gauss point `g` and
gradient component `c` (x, y); node and element `(i, j)` have index
`i N + j`, with `i` along `x`. `W = w I`, `w = h^2 / 4`. Section, table, and
figure numbers refer to the paper as of 2026-10-13.
