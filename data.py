"""Piece 1: teacher data for QCL Figs. 3 and 4.

Run `python data.py` to plot the data and check it (figures/data_check.png).
"""
import numpy as np

# Fig. 3 targets. The caption says sin x, but the plotted curve is sin(pi x).
TARGETS = {
    "x^2": lambda x: x**2,
    "e^x": np.exp,
    "sin(pi x)": lambda x: np.sin(np.pi * x),
    "|x|": np.abs,
}


def fitting_data(f, rng, n=100):
    """Fig. 3: n points uniform on [-1, 1] (sin^-1 x needs |x| <= 1), sorted, and y = f(x)."""
    x_train = np.sort(rng.uniform(-1, 1, size=n))
    return x_train, f(x_train)


def classification_data(rng, n_per_class=100):
    """Fig. 4: blob (class 0) inside a ring (class 1).

    X: (2 n_per_class, 2), every coordinate in [-1, 1] (each one goes into sin^-1).
    labels: (2 n_per_class, 2), one-hot teacher vectors (1, 0) or (0, 1).
    """
    blob = rng.normal(0.0, 0.15, size=(n_per_class, 2))
    r = rng.uniform(0.6, 0.9, size=n_per_class)
    phi = rng.uniform(0, 2 * np.pi, size=n_per_class)
    ring = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    X = np.clip(np.concatenate([blob, ring]), -1, 1)

    labels = np.zeros((2 * n_per_class, 2))
    labels[:n_per_class, 0] = 1
    labels[n_per_class:, 1] = 1
    return X, labels


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for ax, (name, f) in zip(axes, TARGETS.items()):
        x_train, y_train = fitting_data(f, rng)
        print(f"{name:10s} x_train {x_train.shape}  y_train {y_train.shape}  "
              f"x in [{x_train.min():.2f}, {x_train.max():.2f}]")
        ax.plot(x_train, y_train, "o", ms=3)
        ax.set_title(name)

    X, labels = classification_data(rng)
    print(f"Fig. 4     X {X.shape}  labels {labels.shape}  "
          f"coords in [{X.min():.2f}, {X.max():.2f}]")
    cls = labels.argmax(axis=1)
    axes[4].scatter(X[cls == 0, 0], X[cls == 0, 1], s=8, c="tab:blue", label="class 0")
    axes[4].scatter(X[cls == 1, 0], X[cls == 1, 1], s=8, c="tab:red", label="class 1")
    axes[4].set_aspect("equal"); axes[4].set_title("Fig. 4 data"); axes[4].legend(fontsize=8)

    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "data_check.png", dpi=120)
    print("saved", out / "data_check.png")
