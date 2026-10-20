"""circuit_figures.py -- the readout and QSVT circuit figures of the companion
paper (figs/readout_circuit.pdf, figs/qsvt_circuit.pdf), drawn from the circuits
that readout() and u_q() of verify_qiskit_qsvt.py build, with U_V, U_M, U_q and
the material reflection as labeled blocks. They are written as quantikz
(figs/*_circuit.tex) and compiled with pdflatex. Needs Qiskit and pyqsp; called
by `python makeFigures.py circuits`.
"""
import os
import re
import shutil
import subprocess

import numpy as np

from qiskit.circuit import ControlledGate, Gate, Parameter
import verify_qiskit_qsvt as vq


class Box(Gate):
    """Labeled block standing for a subcircuit; its inverse carries the dagger label."""
    def __init__(self, n, label, inv_label=None):
        super().__init__("box", n, [], label=label)
        self.inv_label = label if inv_label is None else inv_label

    def inverse(self, annotated=False):
        return Box(self.num_qubits, self.inv_label, self.label)


class CBox(ControlledGate):
    """Singly controlled Box; self-inverse as used here (-R_chi)."""
    def __init__(self, n, label):
        super().__init__("cbox", n + 1, [], num_ctrl_qubits=1,
                         base_gate=Box(n, label), label=label)

    def inverse(self, annotated=False):
        return CBox(self.num_qubits - 1, self.base_gate.label)


def tex_param(p):
    s = str(p).replace("*", "")
    s = re.sub(r"phi_(\d+)", r"\\phi_{\1}", s)
    return s.replace("theta", r"\theta")


def quantikz(circ, rows, labels, bundles, meters):
    """quantikz source of circ. rows[q] is the row of qubit q, labels[r] the
    label of row r, bundles[r] the number of qubits merged into row r; a
    single-qubit gate applied to every qubit of a bundled row in succession is
    drawn once, with superscript tensor power."""
    ops, data, i = [], circ.data, 0
    while i < len(data):
        ins = data[i]
        q = [circ.find_bit(b).index for b in ins.qubits]
        name, op = ins.operation.name, ins.operation
        if len(q) == 1 and bundles[rows[q[0]]] > 1:              # merge over the bundle
            k = bundles[rows[q[0]]]
            grp = data[i:i + k]
            assert all(g.operation.name == name and len(g.qubits) == 1
                       and rows[circ.find_bit(g.qubits[0]).index] == rows[q[0]] for g in grp), name
            base = {"h": "H", "x": "X"}[name]
            ops.append(("g", [rows[q[0]]], rf"{base}^{{\otimes m}}"))
            i += k
            continue
        r = [rows[x] for x in q]
        if name in ("h", "x"):
            ops.append(("g", r, name.upper()))
        elif name == "ry":
            ops.append(("g", r, rf"R_y({tex_param(op.params[0])})"))
        elif name == "rzz":
            assert abs(r[0] - r[1]) == 1
            ops.append(("rzz", sorted(r), rf"R_{{ZZ}}({tex_param(op.params[0])})"))
        elif name == "box":
            span = sorted(set(r))
            assert span == list(range(span[0], span[-1] + 1)), op.label
            ops.append(("box", span, op.label))
        elif name == "cbox":
            span = sorted(set(r[1:]))
            assert span == list(range(span[0], span[-1] + 1))
            ops.append(("cbox", [r[0]] + span, op.base_gate.label))
        elif name.startswith("mcx"):
            st = op.ctrl_state
            ops.append(("mcx", r, [(st >> j) & 1 for j in range(len(r) - 1)]))
        else:
            raise ValueError(name)
        i += 1

    nrow = len(labels)
    last, cols = [0] * nrow, []
    for kind, r, _ in reversed(ops):                             # as late as possible
        span = range(min(r), max(r) + 1)
        c = max(last[x] for x in span)
        for x in span:
            last[x] = c + 1
        cols.append(c)
    ncol = max(last)
    cols = [ncol - 1 - c for c in reversed(cols)]
    grid = [[""] * ncol for _ in range(nrow)]
    for (kind, r, lab), c in zip(ops, cols):
        if kind == "g":
            grid[r[0]][c] = rf"\gate{{{lab}}}"
        elif kind in ("box", "rzz"):
            grid[r[0]][c] = rf"\gate[{len(r)}]{{{lab}}}"
        elif kind == "cbox":
            grid[r[0]][c] = rf"\ctrl{{{r[-1] - r[0]}}}"
            grid[r[1]][c] = rf"\gate[{len(r) - 1}]{{{lab}}}"
        elif kind == "mcx":
            *ctl, tgt = r
            nodes = sorted(r)
            for x, closed in zip(ctl, lab):
                j = nodes.index(x)
                nxt = nodes[j + 1] if x < tgt else nodes[j - 1]
                grid[x][c] = (r"\ctrl" if closed else r"\octrl") + f"{{{nxt - x}}}"
            grid[tgt][c] = r"\targ{}"
    lines = []
    for x in range(nrow):
        head = rf"\lstick{{{labels[x]}}} & " + (rf"\qwbundle{{m}}" if bundles[x] > 1 else "")
        tail = r" & \meter{}" if meters else r" & "
        lines.append(head + " & " + " & ".join(grid[x]) + tail)
    return ("\\begin{quantikz}[column sep=0.18cm, row sep={0.7cm,between origins}]\n"
            + " \\\\\n".join(lines) + "\n\\end{quantikz}\n")


def compile_circuit(body, name):
    tex = ("\\documentclass[border=2pt]{standalone}\n\\usepackage{amsmath,amssymb}\n"
           "\\usepackage{quantikz}\n\\begin{document}\n" + body + "\\end{document}\n")
    open(f"figs/{name}.tex", "w").write(tex)
    if shutil.which("pdflatex") is None:
        print(f"figs/{name}.tex written; pdflatex not found, PDF not built")
        return
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"],
                         cwd="figs", capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:]
    for ext in ("aux", "log"):
        os.remove(f"figs/{name}.{ext}")
    print(f"figs/{name}.pdf")


def draw_circuits():
    """figs/readout_circuit.pdf and figs/qsvt_circuit.pdf from verify_qiskit_qsvt.py."""
    from qiskit import QuantumCircuit
    m = vq.m
    UV = Box(m + 4, r"\mathcal{U}_V", r"\mathcal{U}_V^{\dagger}")
    UM = Box(m + 5, r"\mathcal{U}_M", r"\mathcal{U}_M^{\dagger}")
    Uq = Box(m + 7, r"\mathcal{U}_q")
    th = Parameter("theta")

    # U_Sigma: Ry(theta) on its ancilla, -R_chi controlled by it, Ry(-theta)
    ref = vq.u_sigma(np.ones(vq.N * vq.N), vq.RHO)
    assert [d.operation.name for d in ref.data] == ["ry", "diagonal", "ry"]
    assert all(ref.find_bit(ref.data[j].qubits[0]).index == m for j in (0, 2))
    US = QuantumCircuit(m + 1, name="U_Sigma")
    US.ry(th, m)
    US.append(CBox(m, r"-R_{\chi}"), [m] + list(range(m)))
    US.ry(-th, m)

    # readout: pos (m), loc (4), qa, z, c, s1, s2, f1, f2 of verify_qiskit_qsvt.py,
    # labeled s_0, z_1, z_2, s_1, s_2, f_1, f_2 in the paper
    R = vq.readout(UV, US, Uq).decompose(gates_to_decompose=["U_Sigma", "U_Sigma_dg"])
    assert R.num_qubits == 15                                                  # "15 qubits"
    rows = [0] * m + list(range(1, 12))
    labels = [r"position", r"local $\ell_0$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$",
              r"$s_0$", r"$z_1$", r"$z_2$", r"$s_1$", r"$s_2$", r"$f_1$", r"$f_2$"]
    compile_circuit(quantikz(R, rows, labels, [m] + [1] * 11, meters=True), "readout_circuit")

    # U_q: d queries to U_M and U_M^dag between d + 1 projector phases
    for d in (3, 15):
        Q = vq.u_q(UM, [Parameter(f"phi_{j}") for j in range(d + 1)])
        assert sum(x.operation.name == "box" for x in Q.data) == d             # "d queries to U_M"
    Q = vq.u_q(UM, [Parameter(f"phi_{j}") for j in range(4)])
    rows = [0] * m + list(range(1, 8))
    labels = [r"position", r"local $\ell_0$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$",
              r"$s_0$", r"$z_1$", r"$z_2$"]
    compile_circuit(quantikz(Q, rows, labels, [m] + [1] * 7, meters=False), "qsvt_circuit")
