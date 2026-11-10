import os
import re
import sys
import json
import math
import subprocess
import numpy as np
from scipy.optimize import brentq

# Anchor paths to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
efem_path = os.path.join(script_dir, "efem.py")
input_dir = os.path.join(script_dir, "input")
os.makedirs(input_dir, exist_ok=True)


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


def build_1d_mesh(elem_type, ne):
    """Build 1D uniform nodes, connectivity, right boundary node, and total DOFs (dpn = 1)."""
    h = 1.0 / ne
    if elem_type == "line2":
        nnodes = ne + 1
        nodes = [[i * h] for i in range(nnodes)]
        elems = [[i, i + 1] for i in range(ne)]
        right_node = ne
    elif elem_type == "line3":
        nnodes = 2 * ne + 1
        nodes = [[i * (h / 2.0)] for i in range(nnodes)]
        elems = [[2 * i, 2 * i + 1, 2 * i + 2] for i in range(ne)]
        right_node = 2 * ne
    else:
        raise ValueError(f"Unsupported 1D element type: {elem_type}")
    return nodes, elems, right_node, nnodes


def solve_mesh(elem_type, ne, a, xb, n_gauss):
    """Generate JSON input, run efem.py, and return (total DOFs n_D, FE strain energy Uh)."""
    nodes, elems, right_node, ndofs = build_1d_mesh(elem_type, ne)

    input_data = {
        "element": elem_type,
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

    json_path = os.path.join(input_dir, f"input_{elem_type}_a{int(a)}_{ne}.json")
    with open(json_path, "w") as f:
        json.dump(input_data, f, indent=4)

    result = subprocess.run(
        [sys.executable, efem_path, json_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"efem.py failed for {elem_type}, Ne = {ne}:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

    npz_path = os.path.splitext(json_path)[0] + "_out.npz"
    if os.path.exists(npz_path):
        Uh = float(np.load(npz_path)["Uh"])
    else:
        Uh = parse_uh(result.stdout)
        if Uh is None:
            print(f"Could not extract Uh for {elem_type}, Ne = {ne}.")
            print("--- STDOUT ---\n", result.stdout)
            print("--- STDERR ---\n", result.stderr)
            sys.exit(1)

    return ndofs, Uh


def a_posteriori_estimate(n0, n1, n2, u0, u1, u2):
    """Solve Eq. (2.56) for the extrapolated strain energy U using DOF-based Q."""
    if not (n0 < n1 < n2):
        raise ValueError("DOFs must satisfy n0 < n1 < n2.")
    if not (u0 < u1 < u2):
        raise ValueError("Strain energies must increase monotonically: U0 < U1 < U2.")

    # Exponent Q in terms of total DOFs n_D
    Q = math.log(n1 / n0) / math.log(n2 / n1)

    # Large-U asymptotic check: fn(U) > 0 as U -> inf only if (u1 - u0) > Q * (u2 - u1)
    if (u1 - u0) <= Q * (u2 - u1):
        raise RuntimeError(
            "No valid root above U2: energy increments have not shrunk enough "
            "(pre-asymptotic regime or roundoff plateau)."
        )

    def fn(U):
        return (U - u0) / (U - u1) - ((U - u1) / (U - u2)) ** Q

    # Bracket strictly above the vertical asymptote at U = U2
    step = u2 - u1
    low = u2 + max(1e-13 * abs(u2), 1e-9 * step)
    high = u2 + 10.0 * step

    for _ in range(80):
        if fn(high) > 0.0:
            break
        high = u2 + 2.0 * (high - u2)
    else:
        raise RuntimeError("Could not bracket root above U2.")

    u_approx, root_info = brentq(fn, low, high, full_output=True)
    return Q, u_approx, root_info


def run_aposteriori_case(label, elem_type, elements_list, a, xb, U_exact, n_gauss=10):
    print(f"\n=== {label} ===")
    n_list = []
    Uh_list = []

    for ne in elements_list:
        ndofs, Uh = solve_mesh(elem_type, ne, a, xb, n_gauss)
        n_list.append(ndofs)
        Uh_list.append(Uh)
        print(f"Ne = {ne:3d} | n_D = {ndofs:4d} | Uh = {Uh:.15f}")

    print("\n--- Sliding-Triplet A Posteriori Diagnostics (Blind Check + Comparison) ---")
    header = (
        f"{'Meshes (Ne)':<16} | {'DOFs (n0,n1,n2)':<17} | {'beta_app':>8} | "
        f"{'Q':>7} | {'Extrapolated U*':>18} | {'Drift |dU*|/U*':>14} | {'Rel Err vs U (%)':>16}"
    )
    print(header)
    print("-" * len(header))

    prev_u_star = None
    last_valid = None

    for k in range(2, len(elements_list)):
        ne0, ne1, ne2 = elements_list[k - 2], elements_list[k - 1], elements_list[k]
        n0, n1, n2 = n_list[k - 2], n_list[k - 1], n_list[k]
        u0, u1, u2 = Uh_list[k - 2], Uh_list[k - 1], Uh_list[k]

        # Blind apparent convergence rate from consecutive energy increments (no U_exact needed)
        du1 = u1 - u0
        du2 = u2 - u1
        if du1 > 0.0 and du2 > 0.0 and abs((ne1 / ne0) - (ne2 / ne1)) < 1e-12:
            beta_app = math.log(du1 / du2) / (2.0 * math.log(ne1 / ne0))
        else:
            beta_app = float("nan")

        mesh_str = f"({ne0},{ne1},{ne2})"
        dof_str = f"({n0},{n1},{n2})"

        try:
            Q, u_star, root_info = a_posteriori_estimate(n0, n1, n2, u0, u1, u2)
            drift_str = (
                f"{abs(u_star - prev_u_star) / u_star:.3e}"
                if prev_u_star is not None
                else "       ---    "
            )
            rel_err_pct = abs(U_exact - u_star) / U_exact * 100.0
            print(
                f"{mesh_str:<16} | {dof_str:<17} | {beta_app:8.4f} | "
                f"{Q:7.4f} | {u_star:18.14f} | {drift_str:>14} | {rel_err_pct:15.6f}%"
            )
            prev_u_star = u_star
            last_valid = (mesh_str, dof_str, Q, u_star, rel_err_pct, root_info)
        except (ValueError, RuntimeError):
            Q_val = math.log(n1 / n0) / math.log(n2 / n1)
            print(
                f"{mesh_str:<16} | {dof_str:<17} | {beta_app:8.4f} | "
                f"{Q_val:7.4f} | {'NO ROOT (pre-asymp)':>18} | {'---':>14} | {'---':>16}"
            )
            prev_u_star = None

    if last_valid is not None:
        mesh_str, dof_str, Q, u_star, rel_err_pct, root_info = last_valid
        print(f"\nFinal valid estimate on {mesh_str} (DOFs {dof_str}):")
        print(f"  Extrapolated U* : {u_star:.15f}")
        print(f"  Exact U         : {U_exact:.15f}")
        print(f"  Relative error  : {rel_err_pct:.6f}% (brentq converged: {root_info.converged})")


if __name__ == "__main__":
    cases = [
        {
            "label": "Problem 1, Task 3a: 1D Linear Elements (line2, a = 1.0)",
            "elem_type": "line2",
            "elements_list": [2, 4, 8, 16, 32],
            "a": 1.0,
            "xb": 0.5,
            "U_exact": 0.1397797556255038,
            "n_gauss": 10,
        },
        {
            "label": "Problem 1, Task 3b: 1D Quadratic Elements (line3, a = 1.0)",
            "elem_type": "line3",
            "elements_list": [2, 4, 8, 16, 32],
            "a": 1.0,
            "xb": 0.5,
            "U_exact": 0.1397797556255038,
            "n_gauss": 10,
        },
        {
            "label": "Problem 1, Task 3c: 1D Quadratic Elements (line3, a = 100.0)",
            "elem_type": "line3",
            "elements_list": [5, 10, 20, 40, 80, 160, 320, 640],
            "a": 100.0,
            "xb": 0.5,
            "U_exact": 19.6425414852039,
            "n_gauss": 10,
        },
    ]

    for case in cases:
        run_aposteriori_case(**case)