import numpy as np
import matplotlib.pyplot as plt
from shape_fns import hierarchical

p = 5  # Polynomial order to plot (change to 8 to match Fig. 3.2)
xi_vals = np.linspace(-1.0, 1.0, 300)

N_vals = np.zeros((len(xi_vals), p + 1))
dN_vals = np.zeros((len(xi_vals), p + 1))

for k, xi in enumerate(xi_vals):
    N, dN = hierarchical(xi, None, p=p)
    N_vals[k, :] = N
    dN_vals[k, :] = dN[:, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Shape functions phi_i(xi)
for i in range(p + 1):
    label = rf"$\phi_{i+1}$ (linear)" if i < 2 else rf"$\phi_{i+1} = \hat{{\phi}}_{i}$ ($p={i}$)"
    ax1.plot(xi_vals, N_vals[:, i], linewidth=1.8, label=label)

ax1.axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
ax1.set_title(rf"Hierarchical Shape Functions ($1 \leq p \leq {p}$)")
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\phi_i(\xi)$")
ax1.set_xlim(-1, 1)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="best", fontsize=9)

# Right: Derivatives d(phi_i)/d(xi)
for i in range(p + 1):
    label = rf"$\phi'_{i+1}$"
    ax2.plot(xi_vals, dN_vals[:, i], linewidth=1.8, label=label)

ax2.axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
ax2.set_title(rf"Derivatives of Shape Functions ($1 \leq p \leq {p}$)")
ax2.set_xlabel(r"$\xi$")
ax2.set_ylabel(r"$\mathrm{d}\phi_i / \mathrm{d}\xi$")
ax2.set_xlim(-1, 1)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="best", fontsize=9)

plt.tight_layout()
plt.show()