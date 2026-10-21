# The angle oracle of U_V: CORDIC realization and encoding error

This document holds the detail of the gate-level circuit that the paper
summarizes in Appendix A: the CORDIC modes with a worked example, the changes
needed for reversible arithmetic, the five stages of the angle oracle bit by
bit, the register values at one wavenumber, and the encoding-error bound with
its proof. The circuit is `quantuminversionpoisson/isometry_circuit.py`; its
gate count by stage is `IsometryCircuit.stage_counts()`, written to
`tables/uv_cost.tex` by `python makeTables.py uv_cost`.

Notation follows the paper: $\phi_j = \pi k_j/N$, $F$ fractional bits,
$I = F$ CORDIC iterations, registers of $w = F + 4$ bits, and

$$
\omega(\mathbf k) = \operatorname{atan2}(\cos\phi_1\sin\phi_2,\ \sin\phi_1\cos\phi_2),\qquad
\tau(\mathbf k) = \operatorname{atan2}\bigl(\sqrt{2/3}\,\sin\phi_1\sin\phi_2,\ r(\mathbf k)\bigr),
$$

with $r(\mathbf k) = (\sin^2\phi_1\cos^2\phi_2 + \cos^2\phi_1\sin^2\phi_2)^{1/2}$.

## 1. The angles as polar angles

Each angle is the polar angle of a planar vector,

$$
\mathbf p_\omega(\mathbf k) = (\sin\phi_1\cos\phi_2,\ \cos\phi_1\sin\phi_2),\qquad
\mathbf p_\tau(\mathbf k) = \bigl(r(\mathbf k),\ \sqrt{2/3}\,\sin\phi_1\sin\phi_2\bigr),
$$

with $\|\mathbf p_\omega\| = r$. Equivalently, $\omega$ is the polar angle of
the point $(\cot\phi_2, \cot\phi_1)$ and $\tau = \operatorname{atan2}(\sqrt{2/3}, R)$
with $R$ its distance from the origin.

In one dimension, Wang et al. [1] avoid arithmetic for the Poisson equation by
writing each reciprocal eigenvalue of the Laplacian as a product of squared
sines, each prepared by a rotation whose angle is linear in the mode index.
The construction rests on a product formula for one variable, the prepared
amplitude is proportional to $\lambda_{\min}/\lambda_j$, and its extension to
several dimensions, where the eigenvalues are sums over the directions, is
left open.

## 2. The two CORDIC modes

All registers hold two's-complement integers $Z$ read as $Z/2^F$, on
$w = F + 4$ bits. CORDIC [2] replaces every multiplication by a shift. With
$\alpha_i = \arctan 2^{-i}$, a *micro-rotation* by $\pm\alpha_i$ maps $(x, y)$
to $(x \mp 2^{-i}y,\ y \pm 2^{-i}x)$ and multiplies the length by
$\sqrt{1 + 2^{-2i}}$. After $I$ micro-rotations, whatever their signs, the
length has grown by the gain $K = \prod_{i<I}\sqrt{1 + 2^{-2i}} = 1.6468$, and
any angle of magnitude at most $\sum_i \alpha_i = 1.743$ can be reached. The
construction uses $I = F$. The sign of each micro-rotation is a decision bit
$d_i$, and the two modes differ in how the bits are chosen.

- **Rotation mode:** the target angle is given in an $(F+1)$-bit register $z$.
  The bit $d_i = [z < 0]$ is chosen so that the micro-rotation by
  $(1 - 2d_i)\alpha_i$ moves $(x, y)$ toward the target while the same angle
  is subtracted from $z$, which is driven to zero. Started at
  $(x, y) = (1/K, 0)$, it returns $(\cos, \sin)$ of the target.
- **Vectoring mode:** the input is a vector $(x_0, y_0)$ with $x_0 \ge 0$. The
  bit $d_i = [y < 0]$ is chosen so that each micro-rotation reduces $|y|$.
  After $I$ iterations $y \approx 0$, the rotation performed is
  $\theta = \sum_i (1 - 2d_i)\alpha_i = \operatorname{atan2}(y_0, x_0)$, and $x$
  holds $K(x_0^2 + y_0^2)^{1/2}$. The angle is held only in the decision bits.

The listing implements vectoring mode with integer registers and compares the
result with the exact values for the vector $(3, 4)$.

```python
import math

F = 16
def vec(x, y):
    th = 0
    for i in range(F):
        a = round(math.atan(2**-i) * 2**F)
        if y >= 0: x, y, th = x + (y >> i), y - (x >> i), th + a
        else:      x, y, th = x - (y >> i), y + (x >> i), th - a
    return th / 2**F, x / 2**F

th, x = vec(3 * 2**F, 4 * 2**F)
print(f"CORDIC  angle {th:.6f}  length x gain {x:.6f}")
K = math.prod(math.sqrt(1 + 4.0**-i) for i in range(F))
print(f"exact   angle {math.atan2(4, 3):.6f}  length x gain {5 * K:.6f}")
```

```
----- output --------------
CORDIC  angle 0.927292  length x gain 8.233856
exact   angle 0.927295  length x gain 8.233801
```

The arithmetic returns the angle to $3 \times 10^{-6}$ and the length times the
gain to $6 \times 10^{-5}$, with no multiplication.

## 3. From classical to reversible arithmetic

A reversible circuit cannot overwrite a register: every operation must be
invertible, and every intermediate value must later be erased by running the
circuit backward [3]. Three changes to the listing follow.

1. **Adders:** each addition is a reversible adder on $w$-bit registers. The
   adders of Gidney [4] compute each carry as a temporary logical AND whose
   uncomputation by measurement requires no Toffoli gate; each such AND is
   counted as one Toffoli gate, so a $w$-bit addition costs $w - 1$ Toffoli
   gates.
2. **Shifted copies:** a step is reversible once the shifted copy of $x$ is
   kept, so each iteration keeps a copy of $x$ shifted by $i$ bits until the
   uncomputation. These copies are most of the work register.
3. **Decision bits:** the sign bit of $z$ (rotation mode) or of $y$ (vectoring
   mode) is a qubit that selects between addition and subtraction in the step;
   it is kept as $d_i$, and in vectoring mode these bits are the output.

## 4. The five stages, bit by bit

The first four stages compute into the work register, and the fifth uses the
result and then reverses the first four.

1. **Load the angles:** writing the $n = \log_2 N$ bits of $k_j$ into
   positions $F - n$ to $F - 1$ of a register gives $k_j/N$ with no gates. One
   adder and one subtractor form $z_\pm = (k_1 \pm k_2)/N$, the angles
   $\phi_1 \pm \phi_2$ in units of $\pi$, taken modulo $2$.
2. **Cosines and sines:** rotation mode, started at $(x, y) = (1/K, 0)$, runs
   on each of $z_+$ and $z_-$. A flag $[|z_\pm| \ge 1/2]$, read from the two
   leading bits, shifts $z_\pm$ by $1$ and starts $x$ at $-1/K$ instead of
   $1/K$, which brings the angle into the range of convergence. After $I$
   iterations, $(x_\pm, y_\pm) = (\cos\pi z_\pm, \sin\pi z_\pm)$ to about
   $2^{-F}$.
3. **Sum to product:** shifts by one bit and three additions give
   $a = (y_+ + y_-)/2 = \sin\phi_1\cos\phi_2$,
   $b = (y_+ - y_-)/2 = \cos\phi_1\sin\phi_2$, and
   $\zeta = (x_- - x_+)/2 = \sin\phi_1\sin\phi_2$.
4. **Two angles:** vectoring mode runs twice. Vectoring $(a, b)$, after a flag
   $f_\omega$ that negates the pair when $a < 0$, gives the decision bits of
   $\omega$ and leaves $Kr$ in $x$; vectoring $(Kr, \sqrt{2/3}\,K\zeta)$, the
   second component formed by shifts and adds over the signed digits of
   $\sqrt{2/3}\,K$, gives the decision bits of $\tau$. These decision bits and
   $f_\omega$ form $g(\mathbf k)$.
5. **Rotations and uncomputation:** since $\theta = A - 2\sum_i d_i \alpha_i$
   with $A = \sum_i \alpha_i$, a Givens rotation by $\theta$ is the fixed
   rotation by $A$ followed by the rotations by $-2\alpha_i$, each controlled
   by $d_i$; $f_\omega$ controls a further rotation by $\pi$. No register holds
   $\omega$ or $\tau$. A flag set at $\mathbf k = \mathbf 0$ maps $|0\rangle$
   to $|8\rangle$, the phase $e^{i(\phi_1 + \phi_2)}$ is $2n$ phase gates on
   the bits of $\mathbf k$, and stages 1 to 4 are then run in reverse.

Inverse trigonometric functions have been computed into angle registers
before: the arc cotangent of an eigenvalue for the controlled rotation of a
quantum Poisson solver, digit by digit with $O(m^3)$ gates for $m$-bit
registers [5], and the arcsine by reversible CORDIC [6]. Stage 5 uses the
decision bits of vectoring mode as controls instead, as in [6] for state
preparation, so the rotations need $2F + 4$ controlled or fixed single-qubit
rotations per application of $U_V$ and no angle register.

## 5. Values at one wavenumber

The output below emulates stages 2 to 4 in integer arithmetic with $F = 12$ at
$N = 16$ and $\mathbf k = (3, 1)$, with the registers and decision rules of
Section 4; it is an emulation of the stages, not the gate-level circuit.

```
----- output --------------
N = 16, F = 12, k = (3, 1)
stage 2  cos(phi1+phi2)  0.707031 exact  0.707107
         sin(phi1+phi2)  0.706543 exact  0.707107
         cos(phi1-phi2)  0.923340 exact  0.923880
         sin(phi1-phi2)  0.383057 exact  0.382683
stage 3  a         0.544678 exact  0.544895
         b         0.161621 exact  0.162212
         zeta      0.108154 exact  0.108386
stage 4  omega     0.288574 exact  0.289339  bits 011000100110
         tau       0.153809 exact  0.154421  bits 011010110000
```

The fixed-point values agree with the exact values to about $10^{-3}$, and the
largest errors, about $8 \times 10^{-4}$ and $6 \times 10^{-4}$, are in
$\omega$ and $\tau$. At $\mathbf k = (1, 1)$ and $N = 1024$, stage 3 gives
$a = b = 3.07 \times 10^{-3}$ and $\zeta = 9.41 \times 10^{-6}$, and stage 4
gives $\omega = \pi/4$ and $\tau = 0.00177$. The inputs to vectoring are of
order $|\mathbf k|/N$ and $(|\mathbf k|/N)^2$ while their absolute error is of
order $2^{-F}$, which is the origin of the factor $N$ in the encoding error.

## 6. Encoding error

Rounding to $F$ fractional bits introduces the *encoding error* $\delta$, the
largest spectral-norm difference over the wavenumbers between the gate-level
$Q(\mathbf k)$ and a unitary at exact angles. Two facts determine it: the
circuit is exact except for the two angles, and the angle of a planar vector
is ill-conditioned where the vector is short.

**Proposition (encoding error).** Let $N \ge 4$ and $\mathbf k \ne \mathbf 0$.
Let $\tilde Q(\mathbf k)$ be the circuit with computed angles
$\tilde\omega, \tilde\tau$, and $Q_{\mathrm{ex}}(\mathbf k)$ the circuit with
the exact $\tau$ and the exact $\omega$, except that $\omega$ is replaced by
$\tilde\omega$ at $\mathbf k = (N/2, N/2)$. Angles are compared modulo $2\pi$.

1. **Circuit:** $\|\tilde Q(\mathbf k) - Q_{\mathrm{ex}}(\mathbf k)\| \le
   |\tilde\omega - \omega| + |\tilde\tau - \tau|$.
2. **Conditioning:** $\|\mathbf p_\omega(\mathbf k)\| \ge 2\sqrt2/N$ for
   $\mathbf k \notin \{\mathbf 0, (N/2,N/2)\}$, and
   $\|\mathbf p_\tau(\mathbf k)\| \ge 2\sqrt2/N$ for every
   $\mathbf k \ne \mathbf 0$.
3. **Accuracy of the arithmetic (hypothesis):** if for every $\mathbf k$ and
   each $\theta \in \{\omega, \tau\}$
   $$|\tilde\theta(\mathbf k) - \theta(\mathbf k)|\,\|\mathbf p_\theta(\mathbf k)\| \le c_\theta\,2^{-F},$$
   that is, the computed angle has an error that is a fixed multiple of
   $2^{-F}$ relative to the length of the vector whose angle it is, then
   $$\delta \le \frac{c_\theta}{\sqrt2}\,N\,2^{-F}.$$

**Proof.** (i) The phase $e^{i(\phi_1+\phi_2)}$ is applied by exact phase
gates, and the fixed unitaries $T$ and the $\pi/4$ rotation are exact, so the
two circuits differ only in the rotations by $\tau$ and $\omega$. A rotation by
$t$ differs from the rotation by $t'$ in the same plane by
$2|\sin((t-t')/2)| \le |t-t'|$, and the differences of the factors of a product
of unitaries add. (ii) With $c_j = \cos(2\pi k_j/N)$, a direct expansion gives
$r^2 = \tfrac12(1 - c_1 c_2)$. If $k_1 = 0$ and $k_2 \ne 0$, then $c_1 = 1$ and
$c_2 \le c_* = \cos(2\pi/N)$; if $k_1 = N/2$ and $k_2 \ne N/2$, then $c_1 = -1$
and $c_2 \ge -c_*$; in both cases $1 - c_1 c_2 \ge 1 - c_*$. If neither index
is $0$ or $N/2$, then $|c_1|, |c_2| \le c_*$ and
$1 - c_1 c_2 \ge 1 - c_*^2 \ge 1 - c_*$; the cases with $k_2 \in \{0, N/2\}$
follow by symmetry. Hence $r^2 \ge \tfrac12(1 - c_*) = \sin^2(\pi/N)$, and
$\sin(\pi/N) \ge (2\sqrt2/\pi)(\pi/N) = 2\sqrt2/N$ for $N \ge 4$, by concavity
of $\sin$ on $[0, \pi/4]$. Since
$\|\mathbf p_\tau\|^2 = r^2 + \tfrac23 \sin^2\phi_1\sin^2\phi_2 \ge r^2$, the
bound on $\mathbf p_\tau$ follows for $\mathbf k \ne (N/2,N/2)$; at
$\mathbf k = (N/2,N/2)$, $r = 0$ and $\|\mathbf p_\tau\| = \sqrt{2/3} \ge 2\sqrt2/N$.
At $\mathbf k = (N/2,N/2)$ the exact $\tau$ equals $\pi/2$, the first column of
$Q$ does not depend on $\omega$, and both circuits are completions of the same
first column, which is why $\tilde\omega$ may be used there. (iii) By (i),
(ii), and the hypothesis,
$\delta \le c_\omega 2^{-F}/\|\mathbf p_\omega\| + c_\tau 2^{-F}/\|\mathbf p_\tau\|
\le (c_\omega + c_\tau)\,N\,2^{-F}/(2\sqrt2)$; with $c_\theta$ the larger of
the two, this is the stated bound. $\square$

**Measured constants.** The proposition proves the structure of the error,
linear in $N$ and exponentially small in $F$, and reduces it to one measured
quantity, the constant $c_\theta$, which is a property of the arithmetic. In
all 42 cases tested ($N = 4$ to $256$, $F = 12, 16, \dots, 32$), the largest
measured values are $c_\omega = 16.5$ and $c_\tau = 20.4$, so the bound holds
with $c_\theta = 21$ and gives $\delta \le 14.9\,N\,2^{-F}$. The measured
encoding error is smaller, $\delta \le 6N\,2^{-F}$, with largest
$\delta\,2^{F}/N = 5.73$ at $N = 4$ (`python makeFigures.py isometry`); it is
smaller than the bound because the two angle errors peak at different
wavenumbers.

**Where the factor $N$ comes from.** The factor $N$ is a loss of significance
of the kind met when two nearly equal numbers are subtracted in floating
point. It arises at the two wavenumbers at which the vector normalized by the
arithmetic becomes short, $\mathbf k = \mathbf 0$ and $\mathbf k = (N/2,N/2)$:
there the components of $\mathbf p_\omega$ have magnitude of order
$|\mathbf k|/N$, and an absolute error of order $2^{-F}$ becomes an angle error
of order $N 2^{-F}$. Computing these components to relative precision, for
example as $x \cdot (\sin x/x)$ with $x = \pi k_j/N$ exact, followed by a
normalizing shift, would remove the factor $N$ and the $\log_2 N$ bits of $F$
that it requires; this is not implemented.

## 7. Gate counts

Each iteration of rotation mode makes two $w$-bit additions and one update of
the $(F + 1)$-bit angle register, and each iteration of vectoring mode two
$w$-bit additions. With $w - 1$ Toffoli gates per $w$-bit addition, a
rotation-mode run uses $I(2(w - 1) + F) = F(3F + 6)$ Toffoli gates
($6{,}345$ at $F = 45$) and a vectoring run $2I(w - 1) = 2F(F + 3)$
($4{,}320$), and the uncomputation repeats both. The four runs, computed and
uncomputed, use $20F^2 + 48F$ Toffoli gates, $42{,}660$ of the $44{,}510$ of
one application of $U_V$ at $N = 1024$; the remaining $1{,}850$ are the adders
of stages 1 and 3, the constant product, and the zero-mode flag. The work
register holds most of the $5{,}698$ qubits because each iteration keeps a
shifted copy of $x$ until the uncomputation (Section 3).

## References

1. S. Wang et al., Quantum Poisson solver without arithmetic (2024), `wang2024quantum` in the paper's bibliography.
2. J. E. Volder, The CORDIC trigonometric computing technique, IRE Trans. Electron. Comput. EC-8 (1959) 330-334.
3. C. H. Bennett, Logical reversibility of computation, IBM J. Res. Dev. 17 (1973) 525-532.
4. C. Gidney, Halving the cost of quantum addition, Quantum 2 (2018) 74.
5. S. Wang et al., Quantum fast Poisson solver: the algorithm and complete and modular circuit design, Quantum Inf. Process. 19 (2020), `wang2020quantum` in the paper's bibliography.
6. I. Burge, M. Barbeau, J. Garcia-Alfaro, Quantum CORDIC: Arcsin on a budget, arXiv:2411.14434 (2024).
