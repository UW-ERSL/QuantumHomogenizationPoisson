from setuptools import setup

setup(
    name="quantuminversionpoisson",
    version="0.1.0",
    description="Fast-inversion block encoding of preconditioned periodic heat conduction",
    url="https://github.com/UW-ERSL/QuantumInversionPoisson",
    packages=["quantuminversionpoisson"],
    python_requires=">=3.10",
    install_requires=["numpy", "scipy"],
    extras_require={"aer": ["qiskit>=1.0", "qiskit-aer"]},
)
