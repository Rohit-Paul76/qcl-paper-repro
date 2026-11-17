"""Reproduce QCL Fig. 3: fit x^2, e^x, sin(pi x), |x| with one fixed circuit, retraining theta and a.

Run: ~/.venvs/qcl/bin/python fig3.py   ->  figures/fig3_fitting.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from circuit import THETA_SHAPE
from data import TARGETS, fitting_data
from entangler import entangler
from train import N_THETA, fitting_loss, fitting_output, run_bfgs

U_ent = entangler(np.random.default_rng(0))      # drawn once; same for every target
data_rng = np.random.default_rng(1)
init_rng = np.random.default_rng(2)
x_plot = np.linspace(-1, 1, 200)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, f) in zip(axes.ravel(), TARGETS.items()):
    print(f"target {name}", flush=True)
    x_train, y_train = fitting_data(f, data_rng)
    params0 = np.append(init_rng.uniform(0, 2 * np.pi, N_THETA), 1.0)   # theta ~ U[0, 2pi], a = 1

    y_initial = fitting_output(params0, x_plot, U_ent)
    params = run_bfgs(lambda p: fitting_loss(p, x_train, y_train, U_ent), params0)
    y_final = fitting_output(params, x_plot, U_ent)
    print(f"  trained a = {params[N_THETA]:.3f}", flush=True)

    ax.plot(x_train, y_train, "o", ms=3, color="tab:blue", label="teacher")
    ax.plot(x_plot, y_initial, "--", color="grey", label="initial")
    ax.plot(x_plot, y_final, "-", color="tab:red", lw=2, label="final")
    ax.set_title(name); ax.set_xlabel("x"); ax.legend(fontsize=8)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.suptitle("QCL Fig. 3 reproduction (N = 6, D = 6, T = 10)")
fig.tight_layout(); fig.savefig(out / "fig3_fitting.png", dpi=140)
print("saved", out / "fig3_fitting.png")
