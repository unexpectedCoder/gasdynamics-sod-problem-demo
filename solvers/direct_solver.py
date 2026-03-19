from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.random import default_rng
from scipy.spatial import KDTree
from tqdm import trange

from solvers.direct_solver_params import SolverParams


@dataclass(frozen=True)
class Potential:
    sigma: float = 1.0
    epsilon: float = 1.0

    @property
    def r_cut(self):
        return 2 ** (1 / 6) * self.sigma


class Model:
    def __init__(self, mp: SolverParams, pot=Potential(), rng=default_rng()):
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
        sigma_v = np.sqrt(T / self.mp.m)
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
                    self.compute_profiles(pairs=pairs)
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

    def compute_profiles(self, N_bins=300, pairs: np.ndarray | None = None):
        pos = self.positions
        vel = self.velocities
        mass = self.mp.m

        dx_bin = self.mp.Lx / N_bins
        bin_idx = np.clip((pos[:, 0] / dx_bin).astype(int), 0, N_bins - 1)

        x_centers = (np.arange(N_bins) + 0.5) * dx_bin

        # 1D "линейная" числовая плотность вдоль x (интегрировано по y)
        # n_1d [частиц/длина] = count / dx
        counts = np.bincount(bin_idx, minlength=N_bins).astype(np.float64)
        density = counts / dx_bin

        # Макроскорости по бинам
        vx = vel[:, 0]
        vy = vel[:, 1]

        sum_vx = np.bincount(bin_idx, weights=vx, minlength=N_bins)
        sum_vy = np.bincount(bin_idx, weights=vy, minlength=N_bins)

        velocity_x = np.zeros(N_bins, dtype=np.float64)
        mean_vy = np.zeros(N_bins, dtype=np.float64)
        np.divide(sum_vx, counts, out=velocity_x, where=counts > 0)
        np.divide(sum_vy, counts, out=mean_vy, where=counts > 0)

        # Температура из кинетики флуктуаций (2D, k_B = 1):
        # T = (m/2) * (Var(vx) + Var(vy))
        sum_vx2 = np.bincount(bin_idx, weights=vx * vx, minlength=N_bins)
        sum_vy2 = np.bincount(bin_idx, weights=vy * vy, minlength=N_bins)

        mean_vx2 = np.zeros(N_bins, dtype=np.float64)
        mean_vy2 = np.zeros(N_bins, dtype=np.float64)
        np.divide(sum_vx2, counts, out=mean_vx2, where=counts > 0)
        np.divide(sum_vy2, counts, out=mean_vy2, where=counts > 0)

        var_vx = mean_vx2 - velocity_x * velocity_x
        var_vy = mean_vy2 - mean_vy * mean_vy

        temperature = 0.5 * mass * (var_vx + var_vy)
        temperature = np.where(counts > 0, temperature, 0.0)

        # Давление: идеальный вклад + вириальный вклад от парных взаимодействий.
        # Здесь давление трактуем как 2D-давление (усреднение по площади бина A = dx * Ly):
        # p = n_2d * T + (1/(d*A)) * sum_{pairs in bin} (r_ij · F_ij), d = 2
        # где n_2d = count / (dx*Ly) = density / Ly
        area_bin = dx_bin * self.mp.Ly
        n2d = counts / area_bin
        p_ideal = n2d * temperature

        # Вириальный вклад распределяем по бинам по x-координате середины пары (Irving–Kirkwood midpoint)
        if pairs is None:
            pairs = self._find_neighbors()
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

        p_virial = np.zeros(N_bins, dtype=np.float64)
        if pairs.shape[0] > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]

            dr = pos[i] - pos[j]
            r2 = np.sum(dr * dr, axis=1)

            # Отсечение (на всякий случай) и защита от деления на 0
            mask = r2 < self.potential.r_cut**2
            if np.any(mask):
                i = i[mask]
                j = j[mask]
                r2 = np.maximum(r2[mask], 1e-10)

                inv_r2 = (self.potential.sigma * self.potential.sigma) / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6

                # F(r)/r = 24ε/r² · [2(σ/r)^12 − (σ/r)^6]
                f_over_r = 24 * self.potential.epsilon / r2 * (2 * inv_r12 - inv_r6)

                # r_ij · F_ij = (F/r) * r^2
                r_dot_f = f_over_r * r2

                x_mid = 0.5 * (pos[i, 0] + pos[j, 0])
                bin_mid = np.clip((x_mid / dx_bin).astype(int), 0, N_bins - 1)

                virial_sum = np.bincount(bin_mid, weights=r_dot_f, minlength=N_bins)

                # d = 2 (2D)
                p_virial = virial_sum / (2.0 * area_bin)

        pressure = p_ideal + p_virial

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
    model = Model(SolverParams())
    history = model.run(steps=10_000, save_every=100)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    model.save_history(history, data_dir / "history.h5")
