import matplotlib.pyplot as plt
from celluloid import Camera


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
        title=f"Итерация {step}, время {time:.3f}",
        aspect="equal",
    )
    ax.grid(False)

    return fig, ax


if __name__ == "__main__":
    from pathlib import Path

    import h5py

    data_path = Path("data") / "history.h5"
    print(f"Чтение данных из '{data_path}'...")
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
    gif_path = pics_dir / "motion.gif"
    ani.save(gif_path, dpi=150, fps=25)
    print(f"Анимация сохранена в '{gif_path}'")

    print("Сохранение отдельных кадров...")
    frames_dir = pics_dir / "frames"
    frames_dir.mkdir(exist_ok=True, parents=True)
    every = 5
    for step, t, pos in zip(steps[::every], times[::every], positions[::every]):
        fig, ax = make_motion_frame(step, t, pos, labels)
        fig.savefig(frames_dir / f"motion_{step}")
        plt.close(fig)
    print(f"Кадры сохранены в '{frames_dir}'")

    print("Готово!")
