import locale

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")


def animate_motion(positions, labels, figax=None):
    if figax is None:
        fig, ax = plt.subplots(num="motion-animation", figsize=(16, 4))
    else:
        fig, ax = figax

    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.grid(False)
    (line_r,) = ax.plot([], [], ls="", marker=".", markersize=1, c="r")
    (line_b,) = ax.plot([], [], ls="", marker=".", markersize=1, c="b")

    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]
    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())

    def update(frame):
        xy = positions[frame]
        line_r.set_data(xy[~labels, 0], xy[~labels, 1])
        line_b.set_data(xy[labels, 0], xy[labels, 1])
        return line_r, line_b

    return FuncAnimation(fig, update, frames=len(positions), blit=True)


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
        title=f"Итерация ${step}$, время ${round(time, 1)}$",
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

    ax.set(xlabel="$x$", ylabel=ylabel, ylim=(None, 1))
    v_max = np.max(np.abs(values))
    norm_values = values / v_max if v_max != 0 else values
    (line,) = ax.plot(x, norm_values[0], c=color)

    def update(frame):
        line.set_ydata(norm_values[frame])
        return (line,)

    return FuncAnimation(fig, update, frames=len(values), blit=True)


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
        ylim=(None, 1),
        title=f"Итерация ${step}$, время ${round(t, 1)}$",
    )

    return fig, ax


def animate_combined(positions, labels, x, densities, pressures, figax=None):
    """Общая анимация: частицы, плотность, давление (3 строки)."""
    if figax is None:
        fig, axes = plt.subplots(
            3, 1, num="combined-animation", figsize=(16, 10), sharex=True
        )
    else:
        fig, axes = figax

    ax_motion, ax_density, ax_pressure = axes

    ax_motion.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    ax_motion.grid(False)
    ax_density.set(xlabel="$x$", ylabel=r"$\mathrm{\rho}$", ylim=(None, 1))
    ax_pressure.set(xlabel="$x$", ylabel="$p$", ylim=(None, 1))

    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]
    ax_motion.set_xlim(None, all_x.max())
    ax_motion.set_ylim(None, all_y.max())

    rho_max = np.max(np.abs(densities))
    p_max = np.max(np.abs(pressures))
    norm_rho = densities / rho_max if rho_max != 0 else densities
    norm_p = pressures / p_max if p_max != 0 else pressures

    (line_r,) = ax_motion.plot([], [], ls="", marker=".", markersize=1, c="r")
    (line_b,) = ax_motion.plot([], [], ls="", marker=".", markersize=1, c="b")
    (line_rho,) = ax_density.plot(x, norm_rho[0], c="b")
    (line_p,) = ax_pressure.plot(x, norm_p[0], c="g")

    def update(frame):
        xy = positions[frame]
        line_r.set_data(xy[~labels, 0], xy[~labels, 1])
        line_b.set_data(xy[labels, 0], xy[labels, 1])
        line_rho.set_ydata(norm_rho[frame])
        line_p.set_ydata(norm_p[frame])
        return line_r, line_b, line_rho, line_p

    return FuncAnimation(fig, update, frames=len(positions), blit=True)


def make_combined_frame(step, t, positions, labels, x, rho, p, figax=None):
    """Общий кадр: частицы, плотность, давление (3 строки)."""
    if figax is None:
        fig, axes = plt.subplots(
            3, 1, num=f"combined-{step}", figsize=(16, 10), sharex=True
        )
    else:
        fig, axes = figax

    ax_motion, ax_density, ax_pressure = axes

    ax_motion.plot(*positions[~labels].T, ls="", marker=".", markersize=1, c="r")
    ax_motion.plot(*positions[labels].T, ls="", marker=".", markersize=1, c="b")
    ax_motion.set(
        xlabel="$x$",
        ylabel="$y$",
        title=f"Итерация ${step}$, время ${round(t, 1)}$",
        aspect="equal",
    )
    ax_motion.grid(False)

    ax_density.plot(x, rho, c="b")
    ax_density.set(xlabel="$x$", ylabel=r"$\mathrm{\rho}$", ylim=(None, 1))

    ax_pressure.plot(x, p, c="g")
    ax_pressure.set(xlabel="$x$", ylabel="$p$", ylim=(None, 1))

    fig.tight_layout()
    return fig, axes


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
    every = 5

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

    # --- Общая анимация ---
    print("Создание общей анимации (motion + density + pressure)...")
    ani_combined = animate_combined(positions, labels, x_centers, densities, pressures)
    path = pics_dir / "combined.gif"
    ani_combined.save(path, dpi=150, fps=25)
    print(f"Анимация сохранена в '{path}'\n")

    combined_frames_dir = frames_dir / "combined"
    combined_frames_dir.mkdir(exist_ok=True, parents=True)
    print("Сохранение кадров общей анимации...")
    rho_max = np.max(np.abs(densities))
    p_max = np.max(np.abs(pressures))
    for step, t, pos, rho, p in zip(
        steps[::every],
        times[::every],
        positions[::every],
        densities[::every],
        pressures[::every],
    ):
        fig, axes = make_combined_frame(
            step,
            t,
            pos,
            labels,
            x_centers,
            rho / rho_max if rho_max != 0 else rho,
            p / p_max if p_max != 0 else p,
        )
        fig.savefig(combined_frames_dir / f"combined_{step}")
        plt.close(fig)
    print(f"Кадры сохранены в '{combined_frames_dir}'\n")

    print("Готово!")
