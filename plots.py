from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from tqdm import tqdm


def animate_history(history: list[dict[str, Any]], labels: np.ndarray, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num="history-animation", figsize=(16, 4))
    else:
        fig, ax = figax

    camera = Camera(fig)

    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")

    for sol in tqdm(history, desc="Обработка решений"):
        step, t = sol["step"], sol["time"]
        pos = sol["pos"]
        ax.plot(*pos[~labels].T, ls="", marker=".", markersize=1, c="r")
        ax.plot(*pos[labels].T, ls="", marker=".", markersize=1, c="b")
        ax.set_title(f"Шаг {step}, время {t:.5f}")

        camera.snap()

    return camera.animate()


if __name__ == "__main__":
    import pickle

    with open("history.pkl", "rb") as f:
        history = pickle.load(f)

    labels = history[0]["labels"]

    ani = animate_history(history, labels)
    ani.save("history-1.gif", dpi=300, fps=30)
    plt.show()
