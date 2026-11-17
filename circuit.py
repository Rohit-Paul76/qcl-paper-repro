"""Pieces 2 + 4 + readout: the QCL circuit of paper Fig. 2 (N = 6 qubits, D = 6 layers).

encode x  ->  [ U_ent (fixed),  R_X R_Z R_X(theta) on every qubit ] x D  ->  measure <Z>
"""
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 6
DEPTH = 6
THETA_SHAPE = (DEPTH, N_QUBITS, 3)   # 3 angles per qubit per layer = 108

dev = qml.device("default.qubit", wires=N_QUBITS)


def encode(x_per_qubit):
    """Piece 2: qubit j gets R_Y(sin^-1 x_j), then R_Z(cos^-1 x_j^2). No trainable numbers."""
    for j, xj in enumerate(x_per_qubit):
        qml.RY(pnp.arcsin(xj), wires=j)
        qml.RZ(pnp.arccos(xj**2), wires=j)


def layers(theta, U_ent):
    """Piece 4: D layers of [fixed U_ent on all qubits, then trained R_X R_Z R_X on each qubit]."""
    for d in range(DEPTH):
        qml.QubitUnitary(U_ent, wires=range(N_QUBITS))
        for j in range(N_QUBITS):
            qml.RX(theta[d, j, 0], wires=j)
            qml.RZ(theta[d, j, 1], wires=j)
            qml.RX(theta[d, j, 2], wires=j)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def fitting_circuit(x, theta, U_ent):
    """Fig. 3: the same x on all 6 qubits; returns <Z_1>. x can be an array (one run per entry)."""
    encode([x] * N_QUBITS)
    layers(theta, U_ent)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def classification_circuit(X, theta, U_ent):
    """Fig. 4: qubit j gets feature j mod 2 of each point; returns (<Z_1>, <Z_2>). X: (n, 2)."""
    encode([X[:, j % 2] for j in range(N_QUBITS)])
    layers(theta, U_ent)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
