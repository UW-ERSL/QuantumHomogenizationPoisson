# QuantumHomogenizationPoisson

Code for the paper *A Preconditioned Quantum Linear Solver for Periodic
Two-Phase Heat Conduction with Mesh-Independent Condition Number*
(K. Suresh). Every number the paper reports is reproduced by the scripts
below: every table by `makeTables.py`, every figure by `makeFigures.py`,
and the checks of Secs. 2-3 and Appendix A by the `verify_*.py` scripts.

For a periodic two-phase Q4 conduction cell with conductivity contrast `rho`,
the cell-problem operator preconditioned by the homogeneous operator `K`,

    M = (K^+)^{1/2} K^chi (K^+)^{1/2} / sigma_max,

factors as `V^T Sigmatilde V`, where `V = W^{1/2} G (K^+)^{1/2}` is a partial
isometry and `Sigmatilde` is diagonal. Because `K` is circulant, `V` is a
quantum Fourier transform pair around one unit vector of length 8 per
wavenumber. The resulting block encoding of `M` has subnormalization 1 and
condition number at most `rho` for every mesh size `N`, and QSVT with the
odd inverse polynomial `q_t` of degree `d = 2t - 1` block-encodes its inverse
with relative energy-norm error at most `eps_p + 2 d C delta`.

The paper stops at the block encoding of the inverse. Reading out a quantity
of interest (the effective conductivity) is the subject of a companion paper;
the readout scripts are kept in this repository for that paper (see
"Companion-paper code" below).

This repository builds the wavenumber-controlled stage of `U_V` at gate level,
verifies the encoding of `M` and the inverse against the finite element
solution, and runs the numerical experiments of the paper on three dyadic
cells (square, cross, laminate) up to N = 1024. The Python package keeps its
original name, `quantuminversionpoisson`.

The baseline condition number of Sec. 5.2 (Fig. mesh a) uses the
subnormalization `alpha = 16/3` of the encoding of `K^chi / sigma_max` in the
companion package [`pyblockencode`](https://github.com/UW-ERSL/PyBlockEncode)
(`POISSON2D_2PHASE`). The value is set in `verify_square.py` and
`cells.py`; when pyblockencode is installed, `verify_square.py`
cross-checks it.

## Install

    git clone https://github.com/UW-ERSL/QuantumHomogenizationPoisson
    cd QuantumHomogenizationPoisson
    pip install -r requirements.txt

Only numpy and scipy are required for the package; matplotlib for
`makeFigures.py`; pyblockencode (optional, not in `requirements.txt`) for the
cross-check of `alpha` in `verify_square.py`; Qiskit and Qiskit Aer for the Aer cross-checks; and pyqsp
for the companion-paper script `verify_qiskit_qsvt.py` and the circuit
figures of `makeFigures.py circuits`; neither is needed for the paper. On Debian or Ubuntu system Python, pyqsp fails to
build (`AttributeError: install_layout` from the system setuptools); install
in a virtual environment:

    python -m venv venv && . venv/bin/activate
    pip install -U setuptools wheel && pip install -r requirements.txt

## Quick start

The two listings of Sec. 4 of the paper. The first builds the square cell at
N = 256 and rho = 10, applies the inverse to the cell-problem source, and
computes the resources of the inverse at N = 1024; replace `"square"` by
`"cross"` or `"laminate"` for the other two cells.

```python
import cells

cell = cells.Cell("square", N=256, rho=10.0)       # 65,536 unknowns
lo, hi = cells.extreme_eigs(cell)                  # extreme eigenvalues of Mtilde
err, d = cells.inverse_error(cell, eps_p=1e-6)     # q_t(Mtilde), matrix-free
print(hi / lo, d, err)                             # 10.0, 145, 5.03e-07

v = cells.inverse_resources("square", N=1024, rho=10.0, eps_p=1e-6)
print(v["F"], v["toffoli_per_query"], v["qubits"]) # 45, 89022, 5701
```

The second checks the gate-level circuit `U_V` at N = 16 and F = 24 against
`Vhat(k)` at every wavenumber.

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
    python verify_isometry.py          # Sec. 3.5, App. A: circuit checks, about 15 seconds
    python verify_isometry.py --aer    # adds the Qiskit Aer cross-check, about 4 minutes
    python verify_full_unitary.py      # Sec. 3.8, App. A: full-unitary error of U_V, about 6 minutes
    python verify_qsvt_degree.py       # Sec. 3.7: QSVT degree and sup norm, a few seconds
    python verify_square.py            # Sec. 3.6: condition numbers and gate-level encoding of M, about 4 minutes
    python verify_inverse.py           # Sec. 3.7: the inverse on four heat sources, about 1 minute
    python makeTables.py               # every table of the paper to tables/, about 1 minute
    python makeTables.py inverse       # one table; --list for the names
    python makeFigures.py              # every figure of the paper to figs/, about 10 minutes
    python makeFigures.py mesh         # one figure; --list for the names

`tables/` and `figs/` are generated, not committed. The paper uses five tables,
`uv_cost` (tab:uv-cost, App. A.2), `geometry` (Sec. 5.1), `be_M` (Sec. 5.3),
`inverse` (Sec. 5.4) and `resources` (Sec. 5.5), and four figures,
`spectrum` (Sec. 2.2), `cells` (Sec. 5.1), `mesh` (Sec. 5.2) and
`isometry_cost` (App. A). Both scripts assert every number the text quotes
about them before writing. `python makeFigures.py circuits qsvt_poly angles`
draws figures the paper does not use: the readout and QSVT circuit figures
of the companion paper (`circuit_figures.py`, quantikz, needs Qiskit and
pyqsp), the polynomial `q_t`, and the two angles over the wavenumbers.
Table 1 of the paper (symbols) is not generated. Files in
`tables/` or `figs/` left by earlier versions (for example `crossover.pdf`,
`contrast.tex`) are not used.

## Layout

| File | Contents |
|---|---|
| `quantuminversionpoisson/fourier_symbol.py` | Closed-form scalar symbol of `K` (Eq. symbol-entries), functions of `K` by FFT, `K` assembled element by element |
| `quantuminversionpoisson/revcirc.py` | X/CX/CCX reversible fixed-point arithmetic (Gidney adders by default, `ADDER = "cuccaro"` for the ripple-carry count; CORDIC), bit-packed simulator; unchanged from the elasticity package |
| `docs/angle_oracle.md` | The angle oracle of `U_V` in detail: CORDIC modes, reversible arithmetic, the five stages, a trace at k = (3, 1), and the encoding-error bound with its proof (App. A of the paper in brief) |
| `quantuminversionpoisson/isometry_circuit.py` | Closed-form factorization of `Vhat(k)` into two angles; the gate-level circuit `U_V`; dense reference `V` |
| `verify_isometry.py` | Arithmetic, factorization, circuit, and scaling checks |
| `verify_isometry_aer.py` | The exported circuit in Qiskit Aer (MPS) for every wavenumber at N = 4 |
| `verify_full_unitary.py` | (App. A) Full 16 x 16 error of the wavenumber-controlled stage against the exact-angle unitary (42 cases), and the Toffoli count |
| `verify_qsvt_degree.py` | Degree and sup norm of the odd QSVT inverse polynomial against the bounds of Sec. 3.7 (independent of the PDE) |
| `verify_square.py` | Condition numbers of `K^chi`, of a separately encoded preconditioner, and of the preconditioned operator; the encoding of `Mtilde` from the gate-level circuit; square inclusion |
| `verify_inverse.py` | The preconditioned inverse applied to four heat sources, against the finite element solution in the energy norm, with exact `Mtilde` and with the gate-level block |
| `verify_readout.py` | (companion paper) The quadratic form for the effective conductivity (Eq. sigmaH-pre) against the finite element cell solution; a statevector emulation of the readout circuit (Fig. readout) on the gate-level `U_V`; amplitude estimation sampled from its exact output distribution, against Eqs. readout-error and readout-queries |
| `verify_qiskit_qsvt.py` | (companion paper) The readout circuit built entirely in Qiskit at N = 4, rho = 3: `U_V` (QFT, multiplexed `Q(k)`), `U_Sigma`, `U_M`, the QSVT sequence with pyqsp phases for `q_t`, flags; blocks checked against `Mtilde` and `q_t(Mtilde)/alpha_q`, and the all-zero probability sampled in Aer |
| `verify_lsq_readout.py` | (companion paper) The least-squares readout `sigma^H/sigma_max = g^T (I - Pi) g` (even polynomial `f = (r_t + e_t)/(1 + e_t)`, one `U_V` per query): dense identity check, gate-level `U_V` emulation with pyqsp phases to d = 640, amplitude estimation from its exact distribution, shifted polynomial `A >= 0` |
| `cells.py` | Module, not a script: the three test cells (square, cross, laminate) with operators applied matrix-free up to N = 1024, the `R_chi` gate networks checked on every element, the inverse by a Clenshaw recurrence on the Chebyshev interpolant of `q_t`, the choice of F, and the resource counts |
| `makeTables.py` | The five tables of the paper: `uv_cost` (from `IsometryCircuit.stage_counts()`), `geometry`, `be_M`, `inverse` (N <= 64, dense checks), `resources`; every number the text quotes about them asserted |
| `makeFigures.py` | The four figures of the paper: `spectrum`, `cells`, `mesh` (N = 16 ... 1024, asserting every number of Sec. 5.2), `isometry_cost` (`delta <= 6 N 2^-F` in 42 cases, largest 5.73; Toffoli fit within 0.43%); on request, the figures the paper does not use |
| `circuit_figures.py` | The readout and QSVT circuit figures of the companion paper, called by `makeFigures.py circuits` |

The paper and the scripts write the subnormalization of the QSVT polynomial,
`max |q_t|` on [-1, 1], as `C` (`alpha_q` in the companion-paper scripts).

## Factorization (App. A.1)

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

The gate-level detail that the paper leaves out, the two CORDIC modes with a
worked example, the changes for reversible arithmetic, the five stages of the
angle oracle bit by bit, the register values at one wavenumber, and the
encoding-error bound with its proof, is in
[`docs/angle_oracle.md`](docs/angle_oracle.md).

## Implementation checks

**Symbol and isometry (Secs. 2.2, 3.4-3.5, App. A).**

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

**Preconditioned operator and its inverse (Secs. 3.6-3.7, 5.3-5.4).**

- Square inclusion, `v_f = 1/4`, `N` = 8 ... 64, `rho` = 1.5 ... 1e4:
  `kappa(Mtilde) = rho` exactly (the bound is attained), while the condition
  number of `K^chi` seen by QSVT (`alpha = 16/3`) grows by 3.80, 3.96, 3.99
  per doubling of `N`, and the lower bound `kappa_P` for a separately encoded
  preconditioner is larger still; the block of `U_V^dag U_Sigma U_V` from the
  gate-level circuit matches `Mtilde = M + sbar Pi_0` to below `2 delta`, and
  its extreme eigenvalues equal `1/rho` and 1 to 13 digits.
- QSVT inverse polynomial (odd, Gribling et al. Cor. 12):
  `t = ceil((rho/2) ln(2/eps_p))`, `d = 2t - 1`; at rho = 10, d = 77 with
  `C = max |q_t|` = 13.19 = 1.32 rho (eps_p = 1e-3) and d = 145 with
  C = 17.64 = 1.76 rho (eps_p = 1e-6).
- Preconditioned inverse (exact `Mtilde`, and the gate-level block at N = 8, 16):
  relative energy-norm error <= eps_p = 1e-6 for smooth, random, cell-problem
  and worst-case sources, N = 8 ... 64, rho = 10; the worst-case source
  attains the residual 8.69e-7 of the polynomial; the gate-level block changes
  the errors by at most 1.2e-7 (F = 24) and 2.8e-9 (F = 32).

## Numerical experiments (Sec. 5)

All with operators applied matrix-free (FFT for functions of `K`, sparse
`W^{1/2} G`, Clenshaw recurrence for `q_t`), N up to 1024 (N_dof ~ 1.05e6),
on the square (v_f = 1/4), the cross (two bands of width N/8, v_f = 15/64,
connected) and the laminate (chi = 1 for x < 1/2). `R_chi` costs 2, 9 and 0
Toffolis.

- **Cells (N = 256, Sec. 5.1):** `rho lambda_min(Mtilde)`, `lambda_max` and
  `kappa(Mtilde) / rho` equal 1 to six digits for every cell at rho = 10 and
  1e4.
- **Mesh (rho = 10, N = 16 ... 1024, Sec. 5.2):** `kappa(Mtilde) = 10` to ten
  digits; `kappa(K^chi)` seen by QSVT grows by 4.00 per doubling to 1.2e6;
  the inverse at the fixed degree d = 145 (eps_p = 1e-6) has relative
  energy-norm error 4.1e-7 to 8.6e-7 for the cell-problem source of each cell;
  with F from `F = ceil(log2(12 N d C / eps_p))`, F = 39 ... 45 and the
  Toffolis per `U_V` grow from 33,722 to 44,510.
- **Gate-level encoding and the inverse (N <= 64, Secs. 5.3-5.4):** see the
  checks above (`makeTables.py be_M inverse`).
- **Resources (N = 1024, rho = 10, Sec. 5.5):** per query to the encoding of
  `Mtilde` (two `U_V`, one `R_chi`): 89,022 Toffolis and 590 rotations at
  eps_p = 1e-6 (d = 145, F = 45), 51,926 and 546 at eps_p = 1e-3 (d = 77,
  F = 34); the inverse costs 1.3e7 and 4.0e6 Toffolis on 5,701 and 3,578
  logical qubits. Not included: rotation synthesis, the QSVT phase rotations,
  and loading of the microstructure beyond `R_chi`.

## Cost per application of U_V

Toffoli ~ 20.5 F^2 + 64 F + 41 + 4 log2 N, within 0.43% for 12 <= F <= 36
(the CSD digits of the constant sqrt(2/3) depend on F, so the count is not an
exact quadratic), Gidney adders; with `revcirc.ADDER = "cuccaro"` the count is
about 2.05 times larger. `F = ceil(log2(6 N / delta))` for full-unitary error
`delta`; 2F + 1 controlled single-qubit rotations. For the inverse,
`delta = eps_p / (2 d C)`, so `F = ceil(log2(12 N d C / eps_p))`. For N = 1024:
F = 34, 25,962 Toffolis (circuit count), 3,575 wires; F = 45, 44,510 Toffolis,
5,698 wires.
The four CORDIC runs, computed and uncomputed, use exactly 20 F^2 + 48 F
Toffolis (42,660 at F = 45, 96%); `IsometryCircuit.stage_counts()` gives the
count and the work-register qubits by stage (tab:uv-cost of App. A.2).
Rotations per application: 2F + 4 in the angle oracle, 4 n (n - 1) / 2
controlled phases in the two 2D Fourier transforms, and 2n phase gates for
e^{i(phi1+phi2)}.

## What is NOT established

1. **Rotation synthesis:** the rotations are simulated exactly; Clifford+T
   synthesis is not included.
2. **Qubit count:** intermediate CORDIC values are kept until the final
   uncomputation; pebbling is not applied.
3. **Error constant:** `delta <= 6 N 2^-F` is measured in every case tested
   (App. A); the paper states `F = log2 N + log2(1/delta) + O(1)` and does not
   prove the constant.
4. **QSVT phases at scale:** the inverse is applied as the polynomial
   `q_t(Mtilde)` (Clenshaw recurrence, or the dense block of the gate-level
   circuit); the phase factors of `q_t` at d = 145 are not computed.
5. **Readout:** the paper builds the block encoding of the inverse only; no
   quantity of interest is read out (companion paper).
6. **Geometry and dimension:** only dyadic cells (square, cross, laminate),
   isotropic phases, and D = 2 are implemented; non-dyadic inclusions need
   material oracles that are not constructed here.
7. **Gate counts:** single-qubit rotations are counted before synthesis into
   Clifford and T gates; no comparison with classical solvers is made, and for
   problems of practical interest classical solvers are vastly superior.

## Conventions

Scalar temperature, one unknown per node, `N_dof = N^2`. The DFT has entries
`N^-1 e^{-2 pi i k.j/N}`. Local index `2 g + c` for Gauss point `g` and
gradient component `c` (x, y); node and element `(i, j)` have index
`i N + j`, with `i` along `x`. `W = w I`, `w = h^2 / 4`. Section, table, and
figure numbers refer to the paper as of 2026-10-18.

## Companion-paper code

The scripts `verify_readout.py`, `verify_qiskit_qsvt.py`,
`verify_lsq_readout.py`, and the circuit figures of `makeFigures.py circuits`
implement the readout of the effective conductivity, which is the subject of
a companion paper and is not used by the current paper. Their results, as of
2026-10-18:

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
  at eps = 1e-2: d = 63, alpha_q = 12.08, sbar = 0.325,
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
