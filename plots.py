import locale

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")


def animate_motion(positions, labels, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num="motion-animation", figsize=(16, 4))
    else:
        fig, ax = figax

    camera = Camera(fig)
    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.grid(False)
    for xy in positions:
        ax.plot(*xy[~labels].T, ls="", marker=".", markersize=1, c="r")
        ax.plot(*xy[labels].T, ls="", marker=".", markersize=1, c="b")
        camera.snap()

    return camera.animate()


def make_motion_frame(step, time, positions, labels, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num=f"frame-{step}", figsize=(16, 4))
    else:
        fig, ax = figax

    ax.plot(*positions[~labels].T, ls="", marker=".", markersize=1, c="r")
    ax.plot(*positions[labels].T, ls="", marker=".", markersize=1, c="b")
    ax.set(
        xlabel="$x$",
        ylabel="$y$",
        title=f"Итерация ${step}$, время ${time:.2f}$",
        aspect="equal",
    )
    ax.grid(False)

    return fig, ax


def animate_pressure(x, pressures, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num="pressure-animation", figsize=(8, 4))
    else:
        fig, ax = figax

    camera = Camera(fig)
    ax.set(xlabel="$x$", ylabel="$p$", ylim=(0, 1))
    p_max = np.max(pressures)
    for p in pressures:
        ax.plot(x, p / p_max, c="g")
        camera.snap()

    return camera.animate()


def make_pressure_frame(step, t, x, p, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num=f"pressure-{step}", figsize=(8, 4))
    else:
        fig, ax = figax

    ax.plot(x, p, c="g")
    ax.set(
        xlabel="$x$",
        ylabel="$p$",
        ylim=(0, 1),
        title=f"Итерация ${step}$, время ${t:.2f}$",
    )

    return fig, ax


if __name__ == "__main__":
    from pathlib import Path

    import h5py

    data_path = Path("data") / "history.h5"
    print(f"Чтение данных из '{data_path}'...\n")
    with h5py.File(data_path, "r") as f:
        labels = f["labels"][:]
        steps = f["steps"][:]
        times = f["times"][:]
        positions = f["positions"][:]
        velocities = f["velocities"][:]
        x_centers = f["x_centers"][:]
        pressures = f["pressures"][:]
        densities = f["densities"][:]
        temperatures = f["temperatures"][:]
        velocities_x = f["velocities_x"][:]

    plt.style.use("sciart.mplstyle")

    print("Создание анимации...")
    ani = animate_motion(positions, labels)
    pics_dir = Path("pics")
    pics_dir.mkdir(exist_ok=True, parents=True)
    path = pics_dir / "motion.gif"
    ani.save(path, dpi=150, fps=25)
    print(f"Анимация сохранена в '{path}'\n")

    print("Сохранение отдельных кадров...")
    frames_dir = pics_dir / "frames"
    motion_frames_dir = frames_dir / "motion"
    motion_frames_dir.mkdir(exist_ok=True, parents=True)
    every = 3
    for step, t, pos in zip(steps[::every], times[::every], positions[::every]):
        fig, ax = make_motion_frame(step, t, pos, labels)
        fig.savefig(motion_frames_dir / f"motion_{step}")
        plt.close(fig)
    print(f"Кадры сохранены в '{motion_frames_dir}'\n")

    print("Создание анимации давления...")
    ani_pressure = animate_pressure(x_centers, pressures)
    path = pics_dir / "pressure.gif"
    ani_pressure.save(path, dpi=150, fps=25)
    print(f"Анимация сохранена в '{path}'\n")

    print("Сохранение отдельных кадров...")
    pressure_frames_dir = frames_dir / "pressure"
    pressure_frames_dir.mkdir(exist_ok=True, parents=True)
    p_max = np.max(pressures)
    for step, t, p in zip(steps[::every], times[::every], pressures[::every]):
        fig, ax = make_pressure_frame(step, t, x_centers, p / p_max)
        fig.savefig(pressure_frames_dir / f"pressure_{step}")
        plt.close(fig)
    print(f"Кадры сохранены в '{pressure_frames_dir}'\n")

    print("Готово!")
