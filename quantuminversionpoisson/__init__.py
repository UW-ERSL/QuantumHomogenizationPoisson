"""quantuminversionpoisson -- fast-inversion block encoding of preconditioned periodic heat conduction.

The preconditioned operator M = (K^+)^{1/2} K^chi (K^+)^{1/2} / sigma_max of a
periodic two-phase Q4 conduction cell factors as V^T Sigmatilde V with V a
partial isometry, and V = (F^dag x I_8) blockdiag_k Vhat(k) F, with Vhat(k) a
unit vector of length 8. The block encoding of M has subnormalization 1 and
condition number at most rho for every mesh size N.

    from quantuminversionpoisson import IsometryCircuit
    circ = IsometryCircuit(n=4, F=24)            # N = 16, 24 fixed-point bits
    Q, checks = circ.simulate()                  # gate-level, every wavenumber
    print(circ.resources())                      # Toffoli, wires, rotations
"""
from .fourier_symbol import symbol, symbol_function, apply_mode_function, reference
from .isometry_circuit import (IsometryCircuit, vhat_exact, V_assembled, T_matrix,
                               exact_angles, canonical, Q_exact, end_to_end_error)
from .revcirc import Circuit, BitSim

__all__ = ["symbol", "symbol_function", "apply_mode_function", "reference",
           "IsometryCircuit", "vhat_exact", "V_assembled", "T_matrix",
           "exact_angles", "canonical", "Q_exact", "end_to_end_error", "Circuit", "BitSim"]
