import os
import re
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Anchor paths to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
efem_path = os.path.join(script_dir, "efem.py")
input_dir = os.path.join(script_dir, "input")
os.makedirs(input_dir, exist_ok=True)

# Problem 1, Task 1 parameters (quadratic elements, a = 1.0, xb = 0.5)
a = 1.0
xb = 0.5
U_exact = 0.1397797556255038
elements_list = [2, 4, 8, 16, 32]
n_gauss = 10


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

    match_uh = re.search(
        r"^[ \t]*(?:--\s*)?Uh\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    if match_uh:
        return float(match_uh.group(1))

    return None


def parse_vtk(vtk_path):
    """Read 1D x-coordinates and u(x) displacements from efem.py's legacy ASCII VTK output."""
    with open(vtk_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    x_vals, u_vals = [], []
    nnodes = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("POINTS "):
            nnodes = int(lines[i].split()[1])
            for k in range(1, nnodes + 1):
                x_vals.append(float(lines[i + k].split()[0]))
            i += nnodes + 1
            continue
        if lines[i].startswith("VECTORS displacement"):
            for k in range(1, nnodes + 1):
                u_vals.append(float(lines[i + k].split()[0]))
            break
        i += 1

    return np.array(x_vals), np.array(u_vals)


def eval_line3_curve(x_fe, u_fe, elems, pts_per_elem=21):
    """Evaluate the piecewise quadratic FE curve inside each line3 element."""
    xi_loc = np.linspace(-1.0, 1.0, pts_per_elem)
    N0 = 0.5 * xi_loc * (xi_loc - 1.0)
    N1 = 1.0 - xi_loc**2
    N2 = 0.5 * xi_loc * (xi_loc + 1.0)

    x_curve, u_curve = [], []
    for n0, n1, n2 in elems:
        x_curve.extend(N0 * x_fe[n0] + N1 * x_fe[n1] + N2 * x_fe[n2])
        u_curve.extend(N0 * u_fe[n0] + N1 * u_fe[n1] + N2 * u_fe[n2])
    return np.array(x_curve), np.array(u_curve)


h_vals = []
dof_vals = []
errors = []
solutions = []

print("--- Problem 1, Task 1: 1D Quadratic Elements (line3, a = 1.0, 10 Gauss pts) ---")

for ne in elements_list:
    h = 1.0 / ne
    nnodes = 2 * ne + 1
    nodes = [[i * (h / 2.0)] for i in range(nnodes)]
    elems = [[2 * i, 2 * i + 1, 2 * i + 2] for i in range(ne)]

    input_data = {
        "element": "line3",
        "nodes": nodes,
        "elements": elems,
        "material": [1.0, 1.0],  # E = 1.0, A = 1.0 -> EA = 1.0
        "boundary": [
            [0, 2 * ne],         # u(0) = 0, u(1) = 0
            [0.0, 0.0],
        ],
        "load": [
            [],
            [],
        ],
        "gauss": n_gauss,
    }

    json_path = os.path.join(input_dir, f"input_line3_{ne}.json")
    with open(json_path, "w") as f:
        json.dump(input_data, f, indent=4)

    result = subprocess.run(
        [sys.executable, efem_path, json_path],
        capture_output=True,
        text=True,
    )

    stem = os.path.splitext(json_path)[0]
    npz_path = stem + "_out.npz"
    vtk_path = stem + ".vtk"

    if result.returncode != 0:
        print(f"efem.py failed for Ne = {ne}:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

    if os.path.exists(npz_path):
        out = np.load(npz_path)
        x_fe = out["x"]
        u_fe = out["u"]
        Uh = float(out["Uh"])
    else:
        Uh = parse_uh(result.stdout)
        if Uh is None or not os.path.exists(vtk_path):
            print(f"Could not extract Uh or VTK output for Ne = {ne}.")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            sys.exit(1)
        x_fe, u_fe = parse_vtk(vtk_path)

    rel_err = np.sqrt((U_exact - Uh) / U_exact)

    h_vals.append(h)
    dof_vals.append(nnodes)  # total DOFs n = nnodes * dpn (dpn = 1)
    errors.append(rel_err)
    solutions.append((ne, x_fe, u_fe, elems))

    print(f"Ne = {ne:2d} | h = {h:.5f} | n = {nnodes:2d} | Uh = {Uh:.12f} | e = {rel_err:.8e}")

# Terminal convergence rates from the last two points (Ne = 16 and 32)
beta_h = np.log(errors[-2] / errors[-1]) / np.log(h_vals[-2] / h_vals[-1])
beta_n = np.log(errors[-1] / errors[-2]) / np.log(dof_vals[-1] / dof_vals[-2])

print("\n--- Terminal Convergence Rates ---")
print(f"Rate vs h (beta_h): {beta_h:.4f}")
print(f"Slope vs n        : {beta_n:.4f}")

# 1. Numerical solutions vs. exact solution
x_fine = np.linspace(0.0, 1.0, 500)
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(x_fine, exact_u(x_fine, a, xb), "k-", linewidth=2.0, label="Exact $u(x)$")

for ne, x_fe, u_fe, elems in solutions:
    x_curve, u_curve = eval_line3_curve(x_fe, u_fe, elems)
    (line,) = ax1.plot(x_curve, u_curve, linestyle="--", label=f"Quadratic $N_e={ne}$")
    ax1.plot(x_fe, u_fe, marker="s", markersize=4, linestyle="None", color=line.get_color())

ax1.set_title(r"1D Quadratic $h$-FEM Solutions vs. Exact ($a=1, x_b=0.5$)")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$u(x)$")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()
fig1.tight_layout()

# 2. Relative error in energy norm vs. mesh size h
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.loglog(
    h_vals, errors, "rs-", linewidth=1.8, markersize=7,
    label=rf"Quadratic ($p=2$), $\beta_h = {beta_h:.3f}$",
)
ax2.annotate(
    rf"$\beta_h \approx {beta_h:.3f}$",
    xy=(h_vals[-1], errors[-1]),
    xytext=(h_vals[-1] * 1.4, errors[-1] * 1.3),
    fontsize=11,
    color="red",
)
ax2.set_title(r"Energy-Norm Error vs. Mesh Size $h$ ($a=1$)")
ax2.set_xlabel("Mesh size $h$")
ax2.set_ylabel(r"Relative error in energy norm $e$")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
ax2.legend()
fig2.tight_layout()

# 3. Relative error in energy norm vs. total DOFs n
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.loglog(
    dof_vals, errors, "rs-", linewidth=1.8, markersize=7,
    label=rf"Quadratic ($p=2$), slope $= {beta_n:.3f}$",
)
ax3.annotate(
    rf"slope $\approx {beta_n:.3f}$",
    xy=(dof_vals[-1], errors[-1]),
    xytext=(dof_vals[-2] * 0.9, errors[-1] * 1.5),
    fontsize=11,
    color="red",
)
ax3.set_title(r"Energy-Norm Error vs. Total DOFs $n$ ($a=1$)")
ax3.set_xlabel("Total DOFs $n$")
ax3.set_ylabel(r"Relative error in energy norm $e$")
ax3.grid(True, which="both", linestyle="--", alpha=0.6)
ax3.legend()
fig3.tight_layout()

plt.show()