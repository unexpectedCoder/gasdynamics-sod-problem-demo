import matplotlib.pyplot as plt
import numpy as np


def accurate_solution(
    x: np.ndarray,
    speed: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    relative: bool,
    figax=None,
):
    if figax is None:
        fig, ax = plt.subplots(num="accurate-solution")
    else:
        fig, ax = figax

    ax.plot(x, speed, label="$u / u_2$" if relative else "$u$")
    ax.plot(x, pressure, label=r"$p / p_\text{L}$" if relative else "$p$")
    ax.plot(
        x,
        density,
        label=r"$\mathrm{\rho} / \mathrm{\rho}_\text{L}$" if relative else r"$\rho$",
    )
    ax.set(xlabel="Координата $x$", ylabel="Параметры газа")

    return fig, ax


if __name__ == "__main__":
    import locale
    from pathlib import Path

    # 1. Загрузка решения
    res_dir = Path("results")
    sol = np.load(res_dir / "solution.npz")
    x = sol["x"]
    u = sol["speed"]
    p = sol["pressure"]
    rho = sol["density"]
    relative = sol["relative"]
    # 2. Построение графиков
    locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")
    plt.style.use("sciart.mplstyle")
    fig, ax = accurate_solution(x, u, p, rho, relative)
    ax.legend()
    # 3. Сохранение графика
    pics_dir = Path("pics")
    pics_dir.mkdir(parents=True, exist_ok=True)
    save_path = pics_dir / "accurate_solution.svg"
    fig.savefig(save_path)
    print(f"График точного решения сохранён в '{save_path}'")
    # (4. Отображение)
    plt.show()
