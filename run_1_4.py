import os
import re
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from shape_fns import hierarchical

# Anchor paths to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
efem_path = os.path.join(script_dir, "efem.py")
input_dir = os.path.join(script_dir, "input")
os.makedirs(input_dir, exist_ok=True)

# Problem 1, Task 4 parameters (a = 1.0, xb = 0.5)
a = 1.0
xb = 0.5
U_exact = 0.1397797556255038

p_orders = [1, 2, 3, 4, 5]
ne_p = 2
h_elements = [2, 4, 8, 16, 32]


def exact_u(x, a=1.0, xb=0.5):
    b = x - xb
    return (1.0 - x) * (np.arctan(a * b) + np.arctan(a * xb))


def parse_uh(stdout):
    """Extract FE strain energy Uh from efem.py stdout."""
    match = re.search(
        r"^[ \t]*(?:--\s*)?(?:FE\s*)?Strain\s*energy(?:\s*\(Uh\))?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return float(match.group(1))
    return None


def run_efem_case(elem_type, ne, p=1, n_gauss=10, tag="case"):
    """Generate JSON input, execute efem.py, and return (n_dofs, Uh, nodes, u_full)."""
    h = 1.0 / ne
    if elem_type == "line2":
        nnodes = ne + 1
        nodes = [[i * h] for i in range(nnodes)]
        elems = [[i, i + 1] for i in range(ne)]
        right_node = ne
        n_dofs = nnodes
    elif elem_type == "line3":
        nnodes = 2 * ne + 1
        nodes = [[i * (h / 2.0)] for i in range(nnodes)]
        elems = [[2 * i, 2 * i + 1, 2 * i + 2] for i in range(ne)]
        right_node = 2 * ne
        n_dofs = nnodes
    elif elem_type == "linep":
        nnodes = ne + 1
        nodes = [[i * h] for i in range(nnodes)]
        elems = [[i, i + 1] for i in range(ne)]
        right_node = ne
        n_dofs = nnodes + ne * (p - 1)  # p * ne + 1 total DOFs
    else:
        raise ValueError(f"Unsupported element type: {elem_type}")

    input_data = {
        "element": elem_type,
        "order": p,
        "nodes": nodes,
        "elements": elems,
        "material": [1.0, 1.0],
        "boundary": [
            [0, right_node],
            [0.0, 0.0],
        ],
        "load": [
            [],
            [],
        ],
        "gauss": n_gauss,
        "params": [a, xb],
    }

    json_path = os.path.join(input_dir, f"input_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(input_data, f, indent=4)

    result = subprocess.run(
        [sys.executable, efem_path, json_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"efem.py failed for {tag}:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

    npz_path = os.path.splitext(json_path)[0] + "_out.npz"
    if os.path.exists(npz_path):
        out = np.load(npz_path)
        Uh = float(out["Uh"])
        u_full = out["u"]
    else:
        Uh = parse_uh(result.stdout)
        u_full = None
        if Uh is None:
            print(f"Could not extract Uh for {tag}.")
            print(result.stdout)
            sys.exit(1)

    return n_dofs, Uh, np.array(nodes)[:, 0], elems, u_full


def eval_hierarchical_curve(x_nodes, elems, u_full, p, pts_per_elem=100):
    """Reconstruct u^h(x) inside each p-FEM element from nodal + bubble DOFs."""
    nnodes = len(x_nodes)
    xi_loc = np.linspace(-1.0, 1.0, pts_per_elem)
    x_curve, u_curve = [], []

    for e, (n0, n1) in enumerate(elems):
        b_start = nnodes + e * (p - 1)
        u_elem = np.concatenate([
            [u_full[n0], u_full[n1]],
            u_full[b_start : b_start + (p - 1)],
        ])
        x0, x1 = x_nodes[n0], x_nodes[n1]

        for xi in xi_loc:
            N, _ = hierarchical(xi, None, p=p)
            x_curve.append(0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1)
            u_curve.append(np.dot(N, u_elem))

    return np.array(x_curve), np.array(u_curve)


# --- 1. Run p-FEM (2 elements, p = 1..5) ---
print("--- Problem 1, Task 4: 1D Hierarchical p-FEM (Ne = 2, a = 1.0) ---")
p_dofs, p_errors, p_curves = [], [], []

for p in p_orders:
    # Ensure sufficient Gauss points for both stiffness (p) and rational body force f(x)
    n_gauss = max(10, p + 5)
    n_dofs, Uh, x_nodes, elems, u_full = run_efem_case(
        "linep", ne_p, p=p, n_gauss=n_gauss, tag=f"linep_ne{ne_p}_p{p}"
    )
    if u_full is None or len(u_full) != n_dofs:
        raise RuntimeError(
            "Full DOF vector missing from _out.npz. Make sure efem.py saves "
            "`np.savez(filepath + '_out.npz', x=nodes[:, 0], u=U, Uh=Uh, p=p)` before VTK padding."
        )

    rel_err = np.sqrt(max(0.0, U_exact - Uh) / U_exact)
    p_dofs.append(n_dofs)
    p_errors.append(rel_err)

    x_c, u_c = eval_hierarchical_curve(x_nodes, elems, u_full, p)
    p_curves.append((p, x_nodes, u_full[:len(x_nodes)], x_c, u_c))
    print(f"p = {p} | n_GP = {n_gauss:2d} | n = {n_dofs:2d} | Uh = {Uh:.15f} | e = {rel_err:.8e}")

# --- 2. Run h-FEM (line2 and line3, Ne = 2..32) for comparison ---
print("\n--- Running h-FEM Baselines (a = 1.0) for Comparison ---")
h1_dofs, h1_errors = [], []
for ne in h_elements:
    n_dofs, Uh, _, _, _ = run_efem_case("line2", ne, n_gauss=10, tag=f"line2_a1_{ne}")
    rel_err = np.sqrt(max(0.0, U_exact - Uh) / U_exact)
    h1_dofs.append(n_dofs)
    h1_errors.append(rel_err)
    print(f"line2 Ne = {ne:2d} | n = {n_dofs:2d} | e = {rel_err:.8e}")

h2_dofs, h2_errors = [], []
for ne in h_elements:
    n_dofs, Uh, _, _, _ = run_efem_case("line3", ne, n_gauss=10, tag=f"line3_a1_{ne}")
    rel_err = np.sqrt(max(0.0, U_exact - Uh) / U_exact)
    h2_dofs.append(n_dofs)
    h2_errors.append(rel_err)
    print(f"line3 Ne = {ne:2d} | n = {n_dofs:2d} | e = {rel_err:.8e}")

# --- Figure 1: Numerical solutions vs. exact solution ---
x_fine = np.linspace(0.0, 1.0, 500)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(x_fine, exact_u(x_fine, a, xb), "k-", linewidth=2.2, label="Exact $u(x)$")

for p, x_nodes, u_nodal, x_c, u_c in p_curves:
    (line,) = ax1.plot(x_c, u_c, linestyle="--", linewidth=1.6, label=f"$p$-FEM ($p={p}$)")
    if p == 1:
        ax1.plot(x_nodes, u_nodal, "o", markersize=5, color="k", zorder=5, label="Physical nodes")

ax1.set_title(r"1D Hierarchical $p$-FEM Solutions vs. Exact ($N_e=2, a=1, x_b=0.5$)")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$u(x)$")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()
fig1.tight_layout()

# --- Figure 2: e vs. n for p-FEM alone ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.loglog(p_dofs, p_errors, "g^-", linewidth=2.0, markersize=8, label=r"$p$-FEM ($N_e=2, p=1\dots 5$)")
for p, nd, err in zip(p_orders, p_dofs, p_errors):
    ax2.annotate(f" $p={p}$", xy=(nd, err), fontsize=9, color="darkgreen")
ax2.set_title(r"$p$-FEM Energy-Norm Error vs. Total DOFs $n$ ($a=1$)")
ax2.set_xlabel("Total DOFs $n$")
ax2.set_ylabel(r"Relative error in energy norm $e$")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
ax2.legend()
fig2.tight_layout()

# --- Figure 3: e vs. n (p-FEM overlaid with h-FEM) ---
slope_h1 = np.log(h1_errors[-1] / h1_errors[-2]) / np.log(h1_dofs[-1] / h1_dofs[-2])
slope_h2 = np.log(h2_errors[-1] / h2_errors[-2]) / np.log(h2_dofs[-1] / h2_dofs[-2])

fig3, ax3 = plt.subplots(figsize=(8, 5.5))
ax3.loglog(h1_dofs, h1_errors, "bo-", linewidth=1.8, markersize=6,
           label=rf"$h$-FEM Linear ($p=1$), slope $= {slope_h1:.2f}$")
ax3.loglog(h2_dofs, h2_errors, "rs-", linewidth=1.8, markersize=6,
           label=rf"$h$-FEM Quadratic ($p=2$), slope $= {slope_h2:.2f}$")
ax3.loglog(p_dofs, p_errors, "g^-", linewidth=2.0, markersize=8,
           label=r"$p$-FEM ($N_e=2, p=1\dots 5$)")

for p, nd, err in zip(p_orders, p_dofs, p_errors):
    ax3.annotate(f" $p={p}$", xy=(nd, err), fontsize=9, color="darkgreen")

ax3.set_title(r"Convergence Comparison: $h$-FEM vs. $p$-FEM ($a=1, x_b=0.5$)")
ax3.set_xlabel("Total DOFs $n$")
ax3.set_ylabel(r"Relative error in energy norm $e$")
ax3.grid(True, which="both", linestyle="--", alpha=0.6)
ax3.legend()
fig3.tight_layout()

plt.show()