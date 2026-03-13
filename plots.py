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


def animate_field(x, values, ylabel, color, num_id, figax=None):
    """Универсальная анимация скалярного поля вдоль x."""
    if figax is None:
        fig, ax = plt.subplots(num=f"{num_id}-animation", figsize=(8, 4))
    else:
        fig, ax = figax

    camera = Camera(fig)
    ax.set(xlabel="$x$", ylabel=ylabel, ylim=(0, 1))
    v_max = np.max(np.abs(values))
    for v in values:
        ax.plot(x, v / v_max if v_max != 0 else v, c=color)
        camera.snap()

    return camera.animate()


def make_field_frame(step, t, x, values, ylabel, color, num_id, figax=None):
    """Универсальный кадр скалярного поля вдоль x."""
    if figax is None:
        fig, ax = plt.subplots(num=f"{num_id}-{step}", figsize=(8, 4))
    else:
        fig, ax = figax

    ax.plot(x, values, c=color)
    ax.set(
        xlabel="$x$",
        ylabel=ylabel,
        ylim=(0, 1),
        title=f"Итерация ${step}$, время ${t:.2f}$",
    )

    return fig, ax


def animate_pressure(x, pressures, figax=None):
    return animate_field(x, pressures, "$p$", "g", "pressure", figax)


def make_pressure_frame(step, t, x, p, figax=None):
    return make_field_frame(step, t, x, p, "$p$", "g", "pressure", figax)


def animate_density(x, densities, figax=None):
    return animate_field(x, densities, r"$\mathrm{\rho}$", "b", "density", figax)


def make_density_frame(step, t, x, rho, figax=None):
    return make_field_frame(step, t, x, rho, r"$\mathrm{\rho}$", "b", "density", figax)


def animate_temperature(x, temperature, figax=None):
    return animate_field(x, temperature, "$T$", "orange", "temperature", figax)


def make_temperature_frame(step, t, x, T, figax=None):
    return make_field_frame(step, t, x, T, "$T$", "orange", "temperature", figax)


def animate_velocities_x(x, velocities_x, figax=None):
    return animate_field(x, velocities_x, "$v_x$", "purple", "velocities_x", figax)


def make_velocities_x_frame(step, t, x, vx, figax=None):
    return make_field_frame(step, t, x, vx, "$v_x$", "purple", "velocities_x", figax)


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

    pics_dir = Path("pics")
    pics_dir.mkdir(exist_ok=True, parents=True)
    frames_dir = pics_dir / "frames"
    every = 3

    # --- Движение частиц ---
    print("Создание анимации движения...")
    ani = animate_motion(positions, labels)
    path = pics_dir / "motion.gif"
    ani.save(path, dpi=150, fps=25)
    print(f"Анимация сохранена в '{path}'\n")

    motion_frames_dir = frames_dir / "motion"
    motion_frames_dir.mkdir(exist_ok=True, parents=True)
    print("Сохранение кадров движения...")
    for step, t, pos in zip(steps[::every], times[::every], positions[::every]):
        fig, ax = make_motion_frame(step, t, pos, labels)
        fig.savefig(motion_frames_dir / f"motion_{step}")
        plt.close(fig)
    print(f"Кадры сохранены в '{motion_frames_dir}'\n")

    # --- Общий цикл по скалярным полям ---
    fields = [
        ("pressure", pressures, animate_pressure, make_pressure_frame, "pressure"),
        ("density", densities, animate_density, make_density_frame, "density"),
        (
            "temperature",
            temperatures,
            animate_temperature,
            make_temperature_frame,
            "temperature",
        ),
        (
            "velocities_x",
            velocities_x,
            animate_velocities_x,
            make_velocities_x_frame,
            "velocities_x",
        ),
    ]

    for name, data, animate_fn, frame_fn, fname_prefix in fields:
        print(f"Создание анимации {name}...")
        ani_field = animate_fn(x_centers, data)
        path = pics_dir / f"{name}.gif"
        ani_field.save(path, dpi=150, fps=25)
        print(f"Анимация сохранена в '{path}'\n")

        field_frames_dir = frames_dir / name
        field_frames_dir.mkdir(exist_ok=True, parents=True)
        print(f"Сохранение кадров {name}...")
        v_max = np.max(np.abs(data))
        for step, t, v in zip(steps[::every], times[::every], data[::every]):
            fig, ax = frame_fn(step, t, x_centers, v / v_max if v_max != 0 else v)
            fig.savefig(field_frames_dir / f"{fname_prefix}_{step}")
            plt.close(fig)
        print(f"Кадры сохранены в '{field_frames_dir}'\n")

    print("Готово!")
