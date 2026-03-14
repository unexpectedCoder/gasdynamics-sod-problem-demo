from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.random import default_rng
from scipy.spatial import KDTree
from tqdm import trange

from model_params import ModelParams


@dataclass(frozen=True)
class Potential:
    sigma: float = 1.0
    epsilon: float = 1.0

    @property
    def r_cut(self):
        return 2 ** (1 / 6) * self.sigma


class Model:
    def __init__(self, mp: ModelParams, pot=Potential(), rng=default_rng()):
        self.mp = mp
        self.potential = pot
        self.rng = rng

        bounds_l = 0.0, mp.Lx / 2, 0.0, mp.Ly
        pos_l = self._place_particles(mp.rho_l, bounds_l)
        bounds_r = mp.Lx / 2, mp.Lx, 0.0, mp.Ly
        pos_r = self._place_particles(mp.rho_r, bounds_r)
        self.positions = np.concatenate([pos_l, pos_r])

        labels_l = np.zeros(len(pos_l), dtype=bool)
        labels_r = np.ones(len(pos_r), dtype=bool)
        self.labels = np.concatenate([labels_l, labels_r])

        v_l = self._assign_velocities(len(pos_l), mp.T_l)
        v_r = self._assign_velocities(len(pos_r), mp.T_r)
        self.velocities = np.concatenate([v_l, v_r])

    def _place_particles(self, rho: float, bounds: tuple | list):
        x_min, x_max, y_min, y_max = bounds
        a = 1 / np.sqrt(rho)
        xs = np.arange(x_min + a / 2, x_max, a)
        ys = np.arange(y_min + a / 2, y_max, a)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        positions += self.rng.uniform(-0.1 * a, 0.1 * a, positions.shape)
        return positions

    def _assign_velocities(self, n: int, T: float, macro_velo=0.0):
        sigma_v = np.sqrt(T)
        vx = self.rng.normal(macro_velo, sigma_v, n)
        vy = self.rng.normal(0.0, sigma_v, n)
        # Вычтем среднюю скорость (убрать дрейф):
        vx -= vx.mean() - macro_velo
        vy -= vy.mean()
        return np.column_stack([vx, vy])

    def _compute_forces(self, pairs: np.ndarray):
        pos = self.positions

        acc = np.zeros_like(pos)
        if len(pairs) == 0:
            return acc

        # Векторы между частицами в парах
        dr = pos[pairs[:, 0]] - pos[pairs[:, 1]]
        r2 = np.sum(dr**2, axis=1)

        # Отсечение (на всякий случай)
        mask = r2 < self.potential.r_cut**2
        dr = dr[mask]
        r2 = r2[mask]
        valid_pairs = pairs[mask]

        # Предотвращение деления на 0
        r2 = np.maximum(r2, 1e-10)

        # Леннард-Джонсовские коэффициенты
        inv_r2 = self.potential.sigma**2 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2

        # Скалярная сила / r  (для удобства умножения на dr)
        # F(r)/r = 24ε/r² · [2(σ/r)^12 − (σ/r)^6]
        f_over_r = 24 * self.potential.epsilon / r2 * (2 * inv_r12 - inv_r6)
        force = f_over_r[:, np.newaxis] * dr

        # Третий закон Ньютона: F_ij = −F_ji
        # Используем np.add.at для атомарного накопления
        np.add.at(acc, valid_pairs[:, 0], force)
        np.add.at(acc, valid_pairs[:, 1], -force)

        return acc

    def _find_neighbors(self):
        tree = KDTree(self.positions)
        # Замечание: query_pairs возвращает каждую пару ровно один раз (i < j).
        # Это ключевое — вычисляем силу один раз и прибавляем ±F к обеим частицам (3-й закон Ньютона)
        return tree.query_pairs(r=self.potential.r_cut, output_type="ndarray")

    def _apply_walls(self):
        Lx, Ly = self.mp.Lx, self.mp.Ly
        pos, vel = self.positions, self.velocities

        # Левая стенка
        mask = pos[:, 0] < 0
        pos[mask, 0] = -pos[mask, 0]
        vel[mask, 0] = -vel[mask, 0]

        # Правая стенка
        mask = pos[:, 0] > Lx
        pos[mask, 0] = 2 * Lx - pos[mask, 0]
        vel[mask, 0] = -vel[mask, 0]

        # Нижняя стенка
        mask = pos[:, 1] < 0
        pos[mask, 1] = -pos[mask, 1]
        vel[mask, 1] = -vel[mask, 1]

        # Верхняя стенка
        mask = pos[:, 1] > Ly
        pos[mask, 1] = 2 * Ly - pos[mask, 1]
        vel[mask, 1] = -vel[mask, 1]

    def run(self, steps: int, **kw):
        Co = self.mp.Co
        sigma = self.potential.sigma
        mass = self.mp.m

        # Начальные силы
        pairs = self._find_neighbors()
        acc = self._compute_forces(pairs) / mass

        # Массивы для записи (каждые save_every шагов)
        save_every = kw.get("save_every", 100)
        history = []

        for step in trange(steps, desc="Интегрирование"):
            v_max = np.max(np.linalg.norm(self.velocities, axis=1))
            dt = min(5e-3, Co * sigma / max(v_max, 1e-10))

            self.positions += self.velocities * dt + 0.5 * acc * dt**2
            self._apply_walls()
            self.velocities += 0.5 * acc * dt
            pairs = self._find_neighbors()
            acc_new = self._compute_forces(pairs) / mass
            self.velocities += 0.5 * acc_new * dt
            acc = acc_new

            # Сохранение снимков
            if step % save_every == 0:
                x_centers, density, velocity_x, temperature, pressure = (
                    self.compute_profiles()
                )

                history.append(
                    {
                        "step": step,
                        "time": step * dt,
                        "positions": self.positions.copy(),
                        "velocities": self.velocities.copy(),
                        "x_centers": x_centers,
                        "density": density,
                        "velocity_x": velocity_x,
                        "temperature": temperature,
                        "pressure": pressure,
                    }
                )

        return history

    def compute_profiles(self, N_bins=100):
        pos = self.positions
        vel = self.velocities

        dx_bin = self.mp.Lx / N_bins
        bin_idx = np.clip((pos[:, 0] / dx_bin).astype(int), 0, N_bins - 1)

        x_centers = (np.arange(N_bins) + 0.5) * dx_bin

        density = np.zeros(N_bins)
        velocity_x = np.zeros(N_bins)
        temperature = np.zeros(N_bins)
        pressure = np.zeros(N_bins)

        for b in range(N_bins):
            mask = bin_idx == b
            count = np.sum(mask)

            if count == 0:
                continue

            area = dx_bin * self.mp.Ly
            density[b] = count / area  # числовая плотность n

            vx = vel[mask, 0]
            vy = vel[mask, 1]

            velocity_x[b] = np.mean(vx)  # <v_x> — макро-скорость

            # Температура: T = m·<(v - <v>)²> / (d·k_B), d = 2 (2D)
            dvx = vx - velocity_x[b]
            dvy = vy - np.mean(vy)
            temperature[b] = 0.5 * np.mean(dvx**2 + dvy**2)  # T = <δv²>/2 при k_B=m=1

            # Давление идеального газа: p = n·T (в 2D при k_B = 1)
            pressure[b] = density[b] * temperature[b]

        return x_centers, density, velocity_x, temperature, pressure

    @property
    def parameters(self):
        return dict(
            n_l=self.mp.n_particles_l, n_r=self.mp.n_particles_r, **self.mp.__dict__
        )

    def save_history(self, history: list[dict], path: Path):
        """Сохраняет историю симуляции в HDF5.

        Структура файла:
            labels          [N]       bool    — метки частиц (False=левые, True=правые)
            steps           [S]       int32   — номера шагов
            times           [S]       float64 — физическое время снимка
            positions       [S, N, 2] float32 — координаты (x, y)
            velocities      [S, N, 2] float32 — скорости (vx, vy)
        """
        N = len(self.labels)
        M = len(history[0]["x_centers"])

        steps = np.array([h["step"] for h in history], dtype=np.uint32)
        times = np.array([h["time"] for h in history], dtype=np.float64)
        positions = np.stack([h["positions"] for h in history]).astype(
            np.float32
        )  # [S, N, 2]
        velocities = np.stack([h["velocities"] for h in history]).astype(
            np.float32
        )  # [S, N, 2]
        x_centers = history[0]["x_centers"].astype(np.float32)  # [S]
        pressures = np.stack([h["pressure"] for h in history]).astype(
            np.float32
        )  # [S, N]
        densities = np.stack([h["density"] for h in history]).astype(
            np.float32
        )  # [S, N]
        temperatures = np.stack([h["temperature"] for h in history]).astype(
            np.float32
        )  # [S, N]
        velocities_x = np.stack([h["velocity_x"] for h in history]).astype(
            np.float32
        )  # [S, N]

        with h5py.File(path, "w") as f:
            f.attrs.update(self.parameters)

            f.create_dataset("labels", data=self.labels)
            f.create_dataset("steps", data=steps)
            f.create_dataset("times", data=times)
            f.create_dataset(
                "positions",
                data=positions,
                chunks=(1, N, 2),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "velocities",
                data=velocities,
                chunks=(1, N, 2),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "x_centers",
                data=x_centers,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "pressures",
                data=pressures,
                chunks=(1, M),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "densities",
                data=densities,
                chunks=(1, M),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "temperatures",
                data=temperatures,
                chunks=(1, M),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "velocities_x",
                data=velocities_x,
                chunks=(1, M),
                compression="gzip",
                compression_opts=4,
            )


if __name__ == "__main__":
    model = Model(ModelParams())
    history = model.run(steps=20_000, save_every=100)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    model.save_history(history, data_dir / "history.h5")
