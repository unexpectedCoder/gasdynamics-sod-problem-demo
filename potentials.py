import locale
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")
plt.style.use("sciart.mplstyle")

U = lambda d, eps, sigma: 4 * eps * ((sigma / d) ** 12 - (sigma / d) ** 6)
WCA = lambda d, eps, sigma: np.where(
    d < 2 ** (1 / 6) * sigma, 4 * eps * ((sigma / d) ** 12 - (sigma / d) ** 6) + eps, 0
)
d = np.linspace(0.9, 3, 300)

fig, ax = plt.subplots(num="LJ")
eps, sigma = 1, 1
ax.plot(
    d,
    U(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

eps, sigma = 0.5, 1
ax.plot(
    d,
    U(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

eps, sigma = 1, 0.5
d = np.linspace(0.48, 3, 300)
ax.plot(
    d,
    U(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

ax.set(xlabel=r"$d / \mathrm{\sigma}$", ylabel=r"$U / \mathrm{\varepsilon}$")
ax.legend()
fig.savefig(Path("pics/lennard-jones"))

fig, ax = plt.subplots(num="WCA")
d = np.linspace(0.9, 3, 300)
eps, sigma = 1, 1
ax.plot(
    d,
    WCA(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

eps, sigma = 0.5, 1
ax.plot(
    d,
    WCA(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

eps, sigma = 1, 0.5
d = np.linspace(0.48, 3, 300)
ax.plot(
    d,
    WCA(d, eps, sigma),
    label=rf"$\mathrm{{\varepsilon}}={eps}$, $\mathrm{{\sigma}}={sigma}$",
)

ax.set(xlabel=r"$d / \mathrm{\sigma}$", ylabel=r"$U / \mathrm{\varepsilon}$")
ax.legend()
fig.savefig(Path("pics/WCA"))

plt.show()
