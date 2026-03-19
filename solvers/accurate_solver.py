from typing import Any

import numpy as np
from scipy.optimize import fsolve


class AccurateSolver:
    def __init__(
        self,
        x: np.ndarray,
        t: float,
        rho: tuple[float, float],
        p: tuple[float, float],
        gamma: float,
    ):
        self.x = x
        self.t = t
        self.rho_l, self.rho_r = rho
        self.p_l, self.p_r = p
        self.gamma = gamma

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        x_from, x_to, num_points = (
            config["mesh"]["from"],
            config["mesh"]["to"],
            config["mesh"]["num_points"],
        )
        x = np.linspace(x_from, x_to, num_points)
        t = config["t_end"]
        rho = config["left"]["rho"], config["right"]["rho"]
        p = config["left"]["p"], config["right"]["p"]
        gamma = config["gamma"]
        return cls(x, t, rho, p, gamma)

    def solve(self, relative=False):
        zones_params, zones = self.calc_zones(self.x, self.t)
        return {
            "x": self.x,
            "speed": self.speed(zones_params, zones, relative),
            "pressure": self.pressure(zones_params, zones, relative),
            "density": self.density(zones_params, zones, relative),
            "relative": relative,
        }

    def sonic(self, pressure, density):
        return np.sqrt(self.gamma * pressure / density)

    def _zones_parameters(self, x):
        rho_1, rho_2, p_2, u_2 = x

        p_l, rho_l = self.p_l, self.rho_l
        p_r, rho_r = self.p_r, self.rho_r
        gamma = self.gamma

        c_l = self.sonic(p_l, rho_l)
        c_2 = self.sonic(p_2, rho_2)

        D = (p_2 - p_r) / (rho_r * u_2)

        return [
            rho_1 - rho_r * D / (D - u_2),
            rho_2**gamma - (p_2 * rho_l**gamma / p_l),
            p_2
            - p_r
            * ((gamma + 1) * rho_1 - (gamma - 1) * rho_r)
            / ((gamma + 1) * rho_r - (gamma - 1) * rho_1),
            u_2 - 2 / (gamma - 1) * (c_l - c_2),
        ]

    def calc_zones_parameters(self):
        rho_r, rho_l = self.rho_r, self.rho_l
        p_l = self.p_l

        c_l = self.sonic(p_l, rho_l)
        u = c_l

        sol = fsolve(self._zones_parameters, [rho_r, rho_l, p_l, u])
        return {
            "rho_1": sol[0],
            "rho_2": sol[1],
            "p_2": sol[2],
            "u_2": sol[3],
        }

    def calc_zones(self, x: np.ndarray, t: float):
        params = self.calc_zones_parameters()
        rho_2 = params["rho_2"]
        p_2 = params["p_2"]
        u_2 = params["u_2"]

        p_l, rho_l = self.p_l, self.rho_l
        c_l = self.sonic(p_l, rho_l)
        c_2 = self.sonic(p_2, rho_2)

        p_r, rho_r = self.p_r, self.rho_r
        D = (p_2 - p_r) / (rho_r * u_2)

        return params, {
            1: x <= -c_l * t,
            2: (x > -c_l * t) & (x <= (u_2 - c_2) * t),
            3: ((u_2 - c_2) * t < x) & (x <= u_2 * t),
            4: (u_2 * t < x) & (x <= D * t),
            5: x > D * t,
        }

    def speed(
        self,
        zones_params: dict[str, float],
        zones: dict[int, np.ndarray],
        relative=False,
    ):
        p_l, rho_l = self.p_l, self.rho_l
        c_l = self.sonic(p_l, rho_l)
        u_2 = zones_params["u_2"]
        gamma = self.gamma

        u = np.zeros_like(self.x)
        x_2 = self.x[zones[2]]
        u[zones[2]] = 2 / (gamma + 1) * (c_l + x_2 / self.t)
        u[zones[3]], u[zones[4]] = u_2, u_2

        return u / u_2 if relative else u

    def pressure(
        self,
        zones_params: dict[str, float],
        zones: dict[int, np.ndarray],
        relative=False,
    ):
        p_l, p_r = self.p_l, self.p_r
        rho_l = self.rho_l
        gamma = self.gamma

        p = np.full_like(self.x, p_l)
        c_l = self.sonic(p_l, rho_l)
        p_2 = zones_params["p_2"]
        u = self.speed(zones_params, zones)[zones[2]]

        p[zones[2]] = p_l * (1 - 0.5 * (gamma - 1) * u / c_l) ** (
            2 * gamma / (gamma - 1)
        )
        p[zones[3]], p[zones[4]] = p_2, p_2
        p[zones[5]] = p_r

        return p / p_l if relative else p

    def density(
        self,
        zones_params: dict[str, float],
        zones: dict[int, np.ndarray],
        relative=False,
    ):
        rho_l, rho_r = self.rho_l, self.rho_r
        p_l = self.p_l
        gamma = self.gamma
        c_l = self.sonic(p_l, rho_l)

        u = self.speed(zones_params, zones)[zones[2]]
        rho_1, rho_2 = zones_params["rho_1"], zones_params["rho_2"]

        rho = np.full_like(self.x, rho_l)
        rho[zones[2]] = rho_l * (1 - 0.5 * (gamma - 1) * u / c_l) ** (2 / (gamma - 1))
        rho[zones[3]] = rho_2
        rho[zones[4]] = rho_1
        rho[zones[5]] = rho_r

        return rho / rho_l if relative else rho


if __name__ == "__main__":
    from pathlib import Path

    # 1. Инициализация
    print("Инициализация решателя...")
    solver = AccurateSolver(
        x=np.linspace(-0.5, 0.5, 501),
        t=0.1,
        rho=(1.0, 0.125),
        p=(1.0, 0.1),
        gamma=1.4,
    )
    # 2. Решение
    print("Запуск решателя...")
    solution = solver.solve(relative=True)
    # 3. Сохранение численных результатов
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "solution.npz"
    np.savez(save_path, **solution)
    print(f"Результаты решения сохранены в '{save_path}'")
    # 4. Визуализация в отдельном модуле plots.py
