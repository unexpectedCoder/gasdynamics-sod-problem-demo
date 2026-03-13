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


if __name__ == "__main__":
    from pathlib import Path

    import h5py

    data_dir = Path("data")
    with h5py.File(data_dir / "history.h5", "r") as f:
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

    ani = animate_motion(positions, labels)
    pics_dir = Path("pics")
    pics_dir.mkdir(exist_ok=True, parents=True)
    ani.save(pics_dir / "motion.gif", dpi=150, fps=30)
