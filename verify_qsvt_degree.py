"""verify_qsvt_degree -- degree and sup norm of the odd QSVT inverse polynomial (Sec. 3.7); independent of the PDE.

    python verify_qsvt_degree.py       # a few seconds

x q_t(x) = 1 - r_t(x^2), r_t the degree-t Chebyshev polynomial on [1/rho^2, 1]
with r_t(0) = 1 (Gribling et al., Cor. 12). Reports the smallest degree
d = 2t - 1 with relative residual eps_p, against the bound rho ln(2/eps_p) + 1,
and C = max_[-1,1] |q_t| against the bound of Eq. (eq:qsvt-norm).
"""
import numpy as np


def T(t, z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    i = np.abs(z) <= 1
    out[i] = np.cos(t * np.arccos(z[i]))
    out[~i] = np.sign(z[~i]) ** t * np.cosh(t * np.arccosh(np.abs(z[~i])))
    return out


print(f"{'rho':>7} {'eps_p':>6} {'d':>7} {'bound':>7} {'C/rho':>6} {'bound':>6} {'argmax*rho':>10}")
x = np.linspace(1e-12, 1.0, 4_000_001)
for rho in (1.5, 10, 100, 1e3, 1e4):
    for eps_p in (1e-3, 1e-6):
        a = 1 / rho ** 2
        s = (1 + a) / (1 - a)
        t = int(np.ceil(np.arccosh(1 / eps_p) / np.arccosh(s)))
        r = T(t, (1 + a - 2 * x ** 2) / (1 - a)) / np.cosh(t * np.arccosh(s))
        q = (1 - r) / x
        k = np.argmax(np.abs(q))
        Cb = max(1 + eps_p, np.sqrt(0.5 * np.log(2 / eps_p) + 1 / rho))
        print(f"{rho:7g} {eps_p:6.0e} {2 * t - 1:7d} {rho * np.log(2 / eps_p) + 1:7.0f} "
              f"{abs(q[k]) / rho:6.3f} {Cb:6.3f} {x[k] * rho:10.3f}")
