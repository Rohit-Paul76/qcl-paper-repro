"""Piece 5 (+ 6): losses and the BFGS loop. The circuit runs on the simulator; BFGS runs in SciPy.

params is one flat vector: Fig. 3 = [108 angles, a]; Fig. 4 = [108 angles] (a fixed at 1).
"""
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from circuit import THETA_SHAPE, fitting_circuit, classification_circuit

N_THETA = int(np.prod(THETA_SHAPE))


def fitting_output(params, x, U_ent):
    """y(x) = a * <Z_1>(x)."""
    theta, a = params[:N_THETA].reshape(THETA_SHAPE), params[N_THETA]
    return a * fitting_circuit(x, theta, U_ent)


def fitting_loss(params, x, y, U_ent):
    """Quadratic loss: sum_i (y_i - a <Z_1>(x_i))^2."""
    return pnp.sum((y - fitting_output(params, x, U_ent)) ** 2)


def class_probs(params, X, U_ent):
    """softmax(<Z_1>, <Z_2>) per point: (n, 2), rows sum to 1."""
    z1, z2 = classification_circuit(X, params.reshape(THETA_SHAPE), U_ent)
    q = pnp.stack([z1, z2], axis=1)
    e = pnp.exp(q)
    return e / pnp.sum(e, axis=1, keepdims=True)


def classification_loss(params, X, labels, U_ent):
    """Cross-entropy: -sum_i sum_k f_k(x_i) log y_k(x_i), teacher one-hot (1,0) or (0,1)."""
    return -pnp.sum(labels * pnp.log(class_probs(params, X, U_ent)))


def run_bfgs(loss, params0, maxiter=500, log_every=25):
    """Minimise loss(params) with SciPy BFGS; exact gradient from PennyLane (backprop)."""
    grad = qml.grad(loss)
    t0, it = time.time(), [0]

    def f(p):
        return float(loss(pnp.array(p, requires_grad=False)))

    def jac(p):
        return np.asarray(grad(pnp.array(p, requires_grad=True)), dtype=float)

    def callback(p):
        it[0] += 1
        if it[0] % log_every == 0:
            print(f"    iter {it[0]:4d}  loss {f(p):.3e}  ({time.time() - t0:.0f} s)", flush=True)

    res = minimize(f, params0, jac=jac, method="BFGS", callback=callback,
                   options={"maxiter": maxiter})
    print(f"  BFGS done: {res.nit} iterations, loss {res.fun:.3e}, {time.time() - t0:.0f} s "
          f"({res.message})", flush=True)
    return res.x
