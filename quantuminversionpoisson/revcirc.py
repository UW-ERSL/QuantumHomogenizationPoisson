"""revcirc -- gate-level reversible fixed-point arithmetic, with a bit-exact simulator.

Gates are X, CX and CCX only, so every arithmetic stage is a classical
reversible circuit. The simulator runs all inputs at once, bit-packed into
uint64 words, so a circuit can be checked exhaustively over every wavenumber.

Fixed-point convention: a register is a list of wires, least significant
first, read as a two's-complement integer Z; its value is Z / 2^F.
"""
from __future__ import annotations

import numpy as np


class Circuit:
    def __init__(self):
        self.n = 0
        self.gates = []          # ('x', t) | ('cx', c, t) | ('ccx', c1, c2, t)
                                 # | ('and', c1, c2, t): temporary AND onto t = 0 (Gidney)
                                 # | ('andu', c1, c2, t): its measurement-based uncompute
        self.pool = []           # freed wires, known to be zero
        self.peak_live = 0
        self.live = 0

    # -- wires ---------------------------------------------------------------
    def alloc(self, w):
        out = []
        for _ in range(w):
            if self.pool:
                out.append(self.pool.pop())
            else:
                out.append(self.n)
                self.n += 1
        self.live += w
        self.peak_live = max(self.peak_live, self.live)
        return out

    def free(self, wires):
        """Return wires that the caller guarantees are back to zero."""
        self.pool.extend(wires)
        self.live -= len(wires)

    # -- gates ---------------------------------------------------------------
    def x(self, t):
        self.gates.append(("x", t))

    def cx(self, c, t):
        self.gates.append(("cx", c, t))

    def ccx(self, a, b, t):
        self.gates.append(("ccx", a, b, t))

    def and_(self, a, b, t):
        self.gates.append(("and", a, b, t))

    def andu(self, a, b, t):
        self.gates.append(("andu", a, b, t))

    def counts(self, start=0, stop=None):
        """Gate counts. In a sequence and its exact reverse, every temporary AND
        is computed once and uncomputed once, in either order, so the
        Toffoli-equivalent count of a compute-uncompute pair is
        2 * ccx + (and + andu): one AND (4 T) per pair, uncompute free."""
        g = self.gates[start:stop]
        return {k: sum(1 for q in g if q[0] == k) for k in ("x", "cx", "ccx", "and", "andu")}


# ==========================================================================
# simulator
# ==========================================================================
class BitSim:
    """All 2^nin basis inputs at once; wire j holds one bit per input."""

    def __init__(self, n_wires, n_inputs):
        self.B = n_inputs
        self.words = (n_inputs + 63) // 64
        self.s = np.zeros((n_wires, self.words), np.uint64)
        self.ones = np.full(self.words, np.uint64(0xFFFFFFFFFFFFFFFF))

    def set_int(self, wires, values):
        """Load integer values (one per input) into the given wires."""
        values = np.asarray(values, dtype=np.int64)
        for j, w in enumerate(wires):
            bits = ((values >> j) & 1).astype(np.uint8)
            self.s[w] = np.packbits(np.pad(bits, (0, 64 * self.words - self.B)),
                                    bitorder="little").view(np.uint64)

    def get_bits(self, w):
        return np.unpackbits(self.s[w].view(np.uint8),
                             bitorder="little")[:self.B].astype(np.int64)

    def get_int(self, wires, signed=True):
        v = np.zeros(self.B, np.int64)
        for j, w in enumerate(wires):
            v |= self.get_bits(w) << j
        if signed and len(wires):
            top = 1 << (len(wires) - 1)
            v = np.where(v & top, v - (top << 1), v)
        return v

    def run(self, gates):
        """Apply gates. A temporary AND acts as a CCX onto a target that must be
        0 when it is computed and must equal the AND when it is uncomputed (the
        measurement-based uncompute leaves only a phase, fixed by a CZ); both
        conditions are asserted for every input."""
        s = self.s
        for g in gates:
            if g[0] in ("and", "andu"):
                prod = s[g[1]] & s[g[2]]
                assert np.all(s[g[3]] == (0 if g[0] == "and" else prod)), g
                s[g[3]] ^= prod
            elif g[0] == "ccx":
                s[g[3]] ^= s[g[1]] & s[g[2]]
            elif g[0] == "cx":
                s[g[2]] ^= s[g[1]]
            else:
                s[g[1]] ^= self.ones


# ==========================================================================
# arithmetic
# ==========================================================================
ZERO = None          # placeholder in operand views: a constant 0 bit


def shr(reg, i):
    """Arithmetic right shift by i, as a view of length len(reg)."""
    w = len(reg)
    return [reg[j + i] if j + i < w else reg[w - 1] for j in range(w)]


def shl(reg, i):
    """Left shift by i (zero fill), as a view of length len(reg)."""
    return [ZERO if j < i else reg[j - i] for j in range(len(reg))]


def shift(reg, s):
    return shl(reg, s) if s >= 0 else shr(reg, -s)


def copy_view(c, view, dst):
    for v, d in zip(view, dst):
        if v is not ZERO:
            c.cx(v, d)


def cuccaro(c, a, b, cin):
    """b <- b + a + cin (mod 2^w). a and cin are restored. 2w Toffolis."""
    w = len(a)

    def maj(x, y, z):
        c.cx(z, y); c.cx(z, x); c.ccx(x, y, z)

    def uma(x, y, z):
        c.ccx(x, y, z); c.cx(z, x); c.cx(x, y)

    maj(cin, b[0], a[0])
    for i in range(1, w):
        maj(a[i - 1], b[i], a[i])
    for i in range(w - 1, 0, -1):
        uma(a[i - 1], b[i], a[i])
    uma(cin, b[0], a[0])


ADDER = "gidney"     # or "cuccaro" (the Cuccaro ripple-carry adder, about twice the Toffoli count)


def reverse(gates):
    """Exact inverse of a gate list: reversed order, with each temporary AND
    compute and uncompute exchanged."""
    sw = {"and": "andu", "andu": "and"}
    return [(sw[g[0]],) + g[1:] if g[0] in sw else g for g in gates[::-1]]


def gidney(c, a, b, cin):
    """b <- b + a + cin (mod 2^w), a and cin restored [Gidney 2018].

    w - 1 temporary ANDs compute the carries; each is uncomputed by
    measurement at no Toffoli cost. Toffoli-equivalent cost w - 1.
    """
    w = len(a)
    car = [cin] + c.alloc(w - 1)               # car[i] = carry into bit i
    for i in range(w - 1):
        c.cx(car[i], a[i]); c.cx(car[i], b[i])
        c.and_(a[i], b[i], car[i + 1])
        c.cx(car[i], car[i + 1])                # car[i+1] = MAJ(a_i, b_i, car_i)
    c.cx(a[w - 1], b[w - 1]); c.cx(car[w - 1], b[w - 1])
    for i in range(w - 2, -1, -1):
        c.cx(car[i], car[i + 1])
        c.andu(a[i], b[i], car[i + 1])
        c.cx(car[i], a[i]); c.cx(a[i], b[i])   # a restored, b <- a ^ b ^ car
    c.free(car[1:])


def _add(c, a, b, cin):
    (gidney if ADDER == "gidney" else cuccaro)(c, a, b, cin)


def add_view(c, dst, view, neg=None):
    """dst <- dst + view, or dst - view.

    neg: None (add), True (subtract), or a wire e (subtract when e = 1).
    The operand is copied into a temporary register that is cleared afterwards.
    """
    w = len(dst)
    tmp = c.alloc(w)
    cin = c.alloc(1)[0]
    copy_view(c, view, tmp)
    if neg is True:
        for t in tmp:
            c.x(t)
        c.x(cin)
    elif neg is not None:
        for t in tmp:
            c.cx(neg, t)
        c.cx(neg, cin)
    _add(c, tmp, dst, cin)
    if neg is True:
        c.x(cin)
        for t in tmp:
            c.x(t)
    elif neg is not None:
        c.cx(neg, cin)
        for t in tmp:
            c.cx(neg, t)
    copy_view(c, view, tmp)
    c.free(tmp + [cin])


def to_int(v, F, w):
    """Two's-complement bit pattern of round(v 2^F) on w bits."""
    return int(np.round(v * 2 ** F)) & ((1 << w) - 1)


def load_select(c, reg, v0, v1, ctrl):
    """reg (zero) <- v0 if ctrl = 0 else v1 (integer bit patterns)."""
    for j, r in enumerate(reg):
        b0, b1 = (v0 >> j) & 1, (v1 >> j) & 1
        if b0:
            c.x(r)
        if b0 != b1:
            c.cx(ctrl, r)


def add_select(c, dst, v0, v1, ctrl):
    """dst <- dst + (v0 if ctrl = 0 else v1), constants as bit patterns."""
    tmp = c.alloc(len(dst))
    cin = c.alloc(1)[0]
    load_select(c, tmp, v0, v1, ctrl)
    _add(c, tmp, dst, cin)
    load_select(c, tmp, v0, v1, ctrl)
    c.free(tmp + [cin])


def csd(value, F, top=4):
    """Canonical signed digits of value on weights 2^top ... 2^-F."""
    n = int(np.round(value * 2 ** F))
    digits, j = [], -F
    while n != 0:
        if n & 1:
            d = 2 - (n & 3)
            digits.append((j, d))
            n -= d
        n >>= 1
        j += 1
    return digits


def const_mult(c, src, M, F):
    """Fresh register <- M * src, by shift-and-add over the CSD digits of M."""
    out = c.alloc(len(src))
    for j, d in csd(M, F):
        add_view(c, out, shift(src, j), neg=None if d > 0 else True)
    return out


# ==========================================================================
# CORDIC
# ==========================================================================
def cordic_gain(I):
    return float(np.prod(np.sqrt(1 + 2.0 ** (-2 * np.arange(I)))))


def cordic_rotate(c, x, y, z, I, F, Fz):
    """Rotation mode: rotate (x, y) by the angle pi * z. Returns decision bits.

    Decision d_i = 1 when z_i < 0 (clockwise micro-rotation). Copies of
    x >> i are kept as garbage, removed by the final uncompute.
    """
    w, wz = len(x), len(z)
    dbits = []
    for i in range(I):
        d = c.alloc(1)[0]
        c.cx(z[-1], d)
        dbits.append(d)
        t = c.alloc(w - i)                      # x >> i, sign extended on read
        copy_view(c, x[i:], t)
        tv = t + [t[-1]] * i
        # s = +1 (d = 0): x -= y >> i, y += x >> i, z -= alpha_i
        c.x(d)
        add_view(c, x, shr(y, i), neg=d)        # subtract when d = 0
        c.x(d)
        add_view(c, y, tv, neg=d)               # subtract when d = 1
        a = np.arctan(2.0 ** -i) / np.pi
        add_select(c, z, to_int(-a, Fz, wz), to_int(a, Fz, wz), d)
    return dbits


def cordic_vector(c, x, y, I):
    """Vectoring mode: drive y to zero. Returns decision bits.

    d_i = 1 when y_i < 0. The accumulated angle is
    atan2(y0, x0) = sum_i (1 - 2 d_i) atan(2^-i), for x0 >= 0.
    """
    w = len(x)
    dbits = []
    for i in range(I):
        d = c.alloc(1)[0]
        c.cx(y[-1], d)
        dbits.append(d)
        t = c.alloc(w - i)
        copy_view(c, x[i:], t)
        tv = t + [t[-1]] * i
        # s = +1 (d = 0): x += y >> i, y -= x >> i
        add_view(c, x, shr(y, i), neg=d)        # subtract when d = 1
        c.x(d)
        add_view(c, y, tv, neg=d)               # subtract when d = 0
        c.x(d)
    return dbits
