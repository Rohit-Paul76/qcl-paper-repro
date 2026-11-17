"""Reproduce QCL Fig. 4: classify a blob (class 0) inside a ring (class 1).

Run: ~/.venvs/qcl/bin/python fig4.py   ->  figures/fig4_classification.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import classification_data
from entangler import entangler
from train import N_THETA, class_probs, classification_loss, run_bfgs

U_ent = entangler(np.random.default_rng(0))      # same fixed entangler as Fig. 3
X, labels = classification_data(np.random.default_rng(3))
params0 = np.random.default_rng(4).uniform(0, 2 * np.pi, N_THETA)   # a fixed at 1: no extra param

params = run_bfgs(lambda p: classification_loss(p, X, labels, U_ent), params0)
y1 = np.asarray(class_probs(params, X, U_ent))[:, 1]
acc = np.mean((y1 > 0.5) == (labels[:, 1] == 1))
print(f"training accuracy: {acc:.3f}")

g = np.linspace(-1, 1, 41)
GX, GY = np.meshgrid(g, g)
grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
Y1 = np.asarray(class_probs(params, grid, U_ent))[:, 1].reshape(GX.shape)

cls = labels[:, 1].astype(int)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
for ax in (a, b):
    ax.scatter(X[cls == 0, 0], X[cls == 0, 1], s=10, c="tab:blue", label="class 0")
    ax.scatter(X[cls == 1, 0], X[cls == 1, 1], s=10, c="tab:red", label="class 1")
    ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
a.set_title("(a) teacher data"); a.legend(fontsize=8)
im = b.contourf(GX, GY, Y1, levels=20, cmap="coolwarm", vmin=0, vmax=1, alpha=0.8)
b.contour(GX, GY, Y1, levels=[0.5], colors="k", linewidths=1.5)
fig.colorbar(im, ax=b, label="y_1 = P(class 1)")
b.set_title(f"(b) trained output, 0.5 contour (train acc {acc:.2f})")

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out / "fig4_classification.png", dpi=140)
print("saved", out / "fig4_classification.png")
