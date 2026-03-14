from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    Lx: float = 1000.0
    Ly: float = 200.0
    rho_l: float = 0.8
    rho_r: float = 0.2
    T_l: float = 1.0
    T_r: float = 0.5
    m: float = 1.0
    Co: float = 0.5

    @property
    def n_particles_l(self):
        return int(self.rho_l * 0.5 * self.Lx * self.Ly)

    @property
    def n_particles_r(self):
        return int(self.rho_r * 0.5 * self.Lx * self.Ly)

    @property
    def n_particles(self):
        return self.n_particles_l + self.n_particles_r
