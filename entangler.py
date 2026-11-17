"""Piece 3: the fixed entangler U_ent = e^{-iHT} (QCL Eq. 4).

H = sum_j a_j X_j + sum_{j>k} J_jk Z_j (x) Z_k, with a_j, J_jk ~ U[-1, 1], drawn once.
Built as one 2^n x 2^n matrix on all qubits at once, then exponentiated once.

Run `python entangler.py` for the checks.
"""
from functools import reduce

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def on_qubits(ops, n):
    """Kronecker product over n qubits: ops = {qubit: 2x2 matrix}, identity on the rest.

    on_qubits({0: Z, 1: Z}, 6) = Z (x) Z (x) I (x) I (x) I (x) I  (qubit 0 leftmost, as in PennyLane).
    """
    return reduce(np.kron, [ops.get(q, I2) for q in range(n)])


def ising_parts(n, rng):
    """The two halves of H as 2^n x 2^n matrices: H_X = sum a_j X_j, H_ZZ = sum J_jk Z_j Z_k."""
    a = rng.uniform(-1, 1, size=n)
    J = rng.uniform(-1, 1, size=(n, n))  # only k < j is used: 15 pairs for n = 6
    H_X = sum(a[j] * on_qubits({j: X}, n) for j in range(n))
    H_ZZ = sum(J[j, k] * on_qubits({j: Z, k: Z}, n) for j in range(n) for k in range(j))
    return H_X, H_ZZ


def entangler(rng, n=6, T=10.0):
    """U_ent = e^{-iHT}: one fixed 2^n x 2^n unitary. Call once, reuse in every layer."""
    H_X, H_ZZ = ising_parts(n, rng)
    return expm(-1j * T * (H_X + H_ZZ))


if __name__ == "__main__":
    n, T = 6, 10.0
    H_X, H_ZZ = ising_parts(n, np.random.default_rng(0))
    H = H_X + H_ZZ
    U = expm(-1j * T * H)
    dim = 2**n

    print("H shape", H.shape, " Hermitian:", np.allclose(H, H.conj().T))
    print("U shape", U.shape, " unitary (U^dag U = I):", np.allclose(U.conj().T @ U, np.eye(dim)))
    print("H_ZZ diagonal (phases only):", np.allclose(H_ZZ, np.diag(np.diag(H_ZZ))))

    # The halves don't commute, so e^{-iHT} is not (X layer) then (ZZ layer).
    split = expm(-1j * T * H_X) @ expm(-1j * T * H_ZZ)
    print("max |e^{-iHT} - e^{-iT H_X} e^{-iT H_ZZ}| =", np.abs(U - split).max().round(3))

    # Thin alternating slices (Trotterisation) approach e^{-iHT} as n_slices grows.
    for n_slices in (10, 100, 1000):
        dt = T / n_slices
        slice_ = expm(-1j * dt * H_X) @ expm(-1j * dt * H_ZZ)
        err = np.abs(np.linalg.matrix_power(slice_, n_slices) - U).max()
        print(f"  {n_slices:5d} slices: max error {err:.2e}")

    print("same seed, same U:", np.allclose(entangler(np.random.default_rng(0)), U))
