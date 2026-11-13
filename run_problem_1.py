import os
import re
import sys
import json
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from shape_fns import hierarchical

# Anchor paths to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
efem_path = os.path.join(script_dir, "efem.py")
input_dir = os.path.join(script_dir, "input")
os.makedirs(input_dir, exist_ok=True)

# Exact problem parameters (Problem 1)
xb = 0.5
U_EXACT_1 = 0.1397797556255038
U_EXACT_100 = 19.6425414852039


def exact_u(x, a, xb=0.5):
    b = x - xb
    return (1.0 - x) * (np.arctan(a * b) + np.arctan(a * xb))


def parse_stdout_metrics(stdout):
    """Fallback parser if _out.npz is missing fields."""
    uh_match = re.search(
        r"^[ \t]*(?:--\s*)?(?:FE\s*)?Strain\s*energy(?:\s*\(Uh\))?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    cond_match = re.search(
        r"Condition\s*number\s*cond\(K\)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        stdout,
        re.IGNORECASE | re.MULTILINE,
    )
    Uh = float(uh_match.group(1)) if uh_match else None
    cond_K = float(cond_match.group(1)) if cond_match else None
    return Uh, cond_K


def run_efem(elem_type, ne, a, p=1, n_gauss=10, tag="run"):
    """Create JSON input, execute efem.py, and return a dictionary of results."""
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
    Uh_stdout, cond_stdout = parse_stdout_metrics(result.stdout)

    if os.path.exists(npz_path):
        out = np.load(npz_path)
        Uh = float(out["Uh"])
        u_full = out["u"]
        cond_K = float(out["cond_K"]) if "cond_K" in out else cond_stdout
    else:
        Uh = Uh_stdout
        u_full = None
        cond_K = cond_stdout

    if Uh is None:
        print(f"Could not extract Uh for {tag}.")
        sys.exit(1)

    U_ref = U_EXACT_1 if np.isclose(a, 1.0) else U_EXACT_100
    rel_err = np.sqrt(max(0.0, U_ref - Uh) / U_ref)

    return {
        "ne": ne,
        "h": h,
        "p": p,
        "n_dofs": n_dofs,
        "Uh": Uh,
        "err": rel_err,
        "cond_K": cond_K,
        "x_nodes": np.array(nodes)[:, 0],
        "elems": elems,
        "u_full": u_full,
    }


def eval_line3_curve(x_nodes, u_full, elems, pts_per_elem=31):
    """Evaluate piecewise quadratic Lagrange curve inside each line3 element."""
    xi = np.linspace(-1.0, 1.0, pts_per_elem)
    N0 = 0.5 * xi * (xi - 1.0)
    N1 = 1.0 - xi**2
    N2 = 0.5 * xi * (xi + 1.0)
    xc, uc = [], []
    for n0, n1, n2 in elems:
        xc.extend(N0 * x_nodes[n0] + N1 * x_nodes[n1] + N2 * x_nodes[n2])
        uc.extend(N0 * u_full[n0] + N1 * u_full[n1] + N2 * u_full[n2])
    return np.array(xc), np.array(uc)


def eval_linep_curve(x_nodes, u_full, elems, p, pts_per_elem=120):
    """Reconstruct piecewise degree-p hierarchical curve inside each linep element."""
    nnodes = len(x_nodes)
    xi_loc = np.linspace(-1.0, 1.0, pts_per_elem)
    xc, uc = [], []
    for e, (n0, n1) in enumerate(elems):
        b_start = nnodes + e * (p - 1)
        u_elem = np.concatenate([
            [u_full[n0], u_full[n1]],
            u_full[b_start : b_start + (p - 1)],
        ])
        x0, x1 = x_nodes[n0], x_nodes[n1]
        for xi in xi_loc:
            N, _ = hierarchical(xi, None, p=p)
            xc.append(0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1)
            uc.append(np.dot(N, u_elem))
    return np.array(xc), np.array(uc)


def a_posteriori_extrapolation(runs, U_ref, label):
    """Task 3: Extrapolate exact strain energy U from finest 3 meshes using DOF-based Q."""
    r0, r1, r2 = runs[-3], runs[-2], runs[-1]
    n0, n1, n2 = r0["n_dofs"], r1["n_dofs"], r2["n_dofs"]
    u0, u1, u2 = r0["Uh"], r1["Uh"], r2["Uh"]

    Q = math.log(n1 / n0) / math.log(n2 / n1)

    def fn(U):
        return (U - u0) / (U - u1) - ((U - u1) / (U - u2)) ** Q

    step = u2 - u1
    if (u1 - u0) <= Q * step or step <= 0.0:
        print(f"{label:<36} | Pre-asymptotic / no valid root above U2")
        return None

    low = u2 + max(1e-13 * abs(u2), 1e-9 * step)
    high = u2 + 10.0 * step
    for _ in range(80):
        if fn(high) > 0.0:
            break
        high = u2 + 2.0 * (high - u2)

    u_star = brentq(fn, low, high)
    rel_pct = abs(U_ref - u_star) / U_ref * 100.0
    print(
        f"{label:<36} | DOFs=({n0},{n1},{n2}) | Q={Q:.5f} | "
        f"U*={u_star:.12f} | U={U_ref:.12f} | err={rel_pct:.4f}%"
    )
    return u_star


# ==============================================================================
# 1. RUN ALL CASES
# ==============================================================================

# Task 1: a = 1, linear (line2) & quadratic (line3) Lagrange, Ne = 2..32
print("=== Running Task 1: h-FEM Linear & Quadratic (a = 1.0) ===")
h_meshes_a1 = [2, 4, 8, 16, 32]
runs_h1_a1 = [run_efem("line2", ne, a=1.0, n_gauss=10, tag=f"t1_line2_{ne}") for ne in h_meshes_a1]
runs_h2_a1 = [run_efem("line3", ne, a=1.0, n_gauss=10, tag=f"t1_line3_{ne}") for ne in h_meshes_a1]

for r in runs_h1_a1:
    print(f"[line2 a=1] Ne={r['ne']:2d} | n={r['n_dofs']:2d} | Uh={r['Uh']:.12f} | e={r['err']:.6e} | cond={r['cond_K']:.4e}")
for r in runs_h2_a1:
    print(f"[line3 a=1] Ne={r['ne']:2d} | n={r['n_dofs']:2d} | Uh={r['Uh']:.12f} | e={r['err']:.6e} | cond={r['cond_K']:.4e}")

# Task 2: a = 100, quadratic Lagrange (line3), Ne = 5..40
print("\n=== Running Task 2: h-FEM Quadratic (a = 100.0) ===")
h_meshes_a100 = [5, 10, 20, 40]
runs_h2_a100 = [run_efem("line3", ne, a=100.0, n_gauss=20, tag=f"t2_line3_a100_{ne}") for ne in h_meshes_a100]
for r in runs_h2_a100:
    print(f"[line3 a=100] Ne={r['ne']:2d} | n={r['n_dofs']:2d} | Uh={r['Uh']:.12f} | e={r['err']:.6e}")

# Task 3: A posteriori strain energy extrapolation
print("\n=== Task 3: A Posteriori Strain Energy Extrapolation (Eq. 2.56) ===")
a_posteriori_extrapolation(runs_h1_a1, U_EXACT_1, "Linear h-FEM (a=1, Ne=8,16,32)")
a_posteriori_extrapolation(runs_h2_a1, U_EXACT_1, "Quadratic h-FEM (a=1, Ne=8,16,32)")
a_posteriori_extrapolation(runs_h2_a100, U_EXACT_100, "Quadratic h-FEM (a=100, Ne=10,20,40)")

# Task 4: a = 1, p-FEM (Ne = 2, p = 1..5)
print("\n=== Running Task 4: Hierarchical p-FEM (Ne = 2, a = 1.0, p = 1..5) ===")
p_orders = [1, 2, 3, 4, 5]
runs_p_a1 = [
    run_efem("linep", ne=2, a=1.0, p=p, n_gauss=max(10, p + 5), tag=f"t4_linep_a1_p{p}")
    for p in p_orders
]
for r in runs_p_a1:
    print(f"[p-FEM a=1, Ne=2] p={r['p']} | n={r['n_dofs']:2d} | Uh={r['Uh']:.12f} | e={r['err']:.6e} | cond={r['cond_K']:.4e}")

# Task 5: a = 100, p-FEM (Ne = 5, p = 1..5)
print("\n=== Running Task 5: Hierarchical p-FEM (Ne = 5, a = 100.0, p = 1..5) ===")
runs_p_a100 = [
    run_efem("linep", ne=5, a=100.0, p=p, n_gauss=max(30, p + 10), tag=f"t5_linep_a100_p{p}")
    for p in p_orders
]
for r in runs_p_a100:
    print(f"[p-FEM a=100, Ne=5] p={r['p']} | n={r['n_dofs']:2d} | Uh={r['Uh']:.12f} | e={r['err']:.6e} | cond={r['cond_K']:.4e}")


# ==============================================================================
# 2. COMPUTE CONVERGENCE & CONDITION-NUMBER RATES
# ==============================================================================
def log_slope(x, y):
    return np.log(y[-1] / y[-2]) / np.log(x[-1] / x[-2])


h_a1 = [r["h"] for r in runs_h1_a1]
n_h1_a1 = [r["n_dofs"] for r in runs_h1_a1]
e_h1_a1 = [r["err"] for r in runs_h1_a1]

n_h2_a1 = [r["n_dofs"] for r in runs_h2_a1]
e_h2_a1 = [r["err"] for r in runs_h2_a1]

beta_h1_a1 = np.log(e_h1_a1[-2] / e_h1_a1[-1]) / np.log(h_a1[-2] / h_a1[-1])
beta_h2_a1 = np.log(e_h2_a1[-2] / e_h2_a1[-1]) / np.log(h_a1[-2] / h_a1[-1])
slope_n_h1_a1 = log_slope(n_h1_a1, e_h1_a1)
slope_n_h2_a1 = log_slope(n_h2_a1, e_h2_a1)

h_a100 = [r["h"] for r in runs_h2_a100]
n_h2_a100 = [r["n_dofs"] for r in runs_h2_a100]
e_h2_a100 = [r["err"] for r in runs_h2_a100]
beta_h2_a100 = np.log(e_h2_a100[-2] / e_h2_a100[-1]) / np.log(h_a100[-2] / h_a100[-1])
slope_n_h2_a100 = log_slope(n_h2_a100, e_h2_a100)

n_p_a1 = [r["n_dofs"] for r in runs_p_a1]
e_p_a1 = [r["err"] for r in runs_p_a1]

n_p_a100 = [r["n_dofs"] for r in runs_p_a100]
e_p_a100 = [r["err"] for r in runs_p_a100]


# ==============================================================================
# 3. PLOTS FOR TASKS 1, 2, 4, 5, AND 6
# ==============================================================================

# --- Figure 1 (Task 1): e vs h and e vs n for a = 1 (Linear & Quadratic h-FEM) ---
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))

ax1a.loglog(h_a1, e_h1_a1, "bo-", lw=1.8, ms=7, label=rf"Linear ($p=1$), $\beta_h = {beta_h1_a1:.2f}$")
ax1a.loglog(h_a1, e_h2_a1, "rs-", lw=1.8, ms=7, label=rf"Quadratic ($p=2$), $\beta_h = {beta_h2_a1:.2f}$")
ax1a.set_title(r"Task 1: Relative Error vs. Mesh Size $h$ ($a=1$)")
ax1a.set_xlabel("Mesh size $h$")
ax1a.set_ylabel(r"Relative error in energy norm $e$")
ax1a.grid(True, which="both", ls="--", alpha=0.6)
ax1a.legend()

ax1b.loglog(n_h1_a1, e_h1_a1, "bo-", lw=1.8, ms=7, label=rf"Linear ($p=1$), slope $= {slope_n_h1_a1:.2f}$")
ax1b.loglog(n_h2_a1, e_h2_a1, "rs-", lw=1.8, ms=7, label=rf"Quadratic ($p=2$), slope $= {slope_n_h2_a1:.2f}$")
ax1b.set_title(r"Task 1: Relative Error vs. Total DOFs $n$ ($a=1$)")
ax1b.set_xlabel("Total DOFs $n$")
ax1b.set_ylabel(r"Relative error in energy norm $e$")
ax1b.grid(True, which="both", ls="--", alpha=0.6)
ax1b.legend()
fig1.tight_layout()

# --- Figure 2 (Task 2): Quadratic h-FEM Solutions & e vs h for a = 100 ---
x_fine = np.linspace(0.0, 1.0, 1000)
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))

ax2a.plot(x_fine, exact_u(x_fine, 100.0, xb), "k-", lw=2.0, label="Exact $u(x)$ ($a=100$)")
for r in runs_h2_a100:
    xc, uc = eval_line3_curve(r["x_nodes"], r["u_full"], r["elems"])
    ax2a.plot(xc, uc, "--", lw=1.5, label=f"Quadratic $N_e={r['ne']}$")
ax2a.set_title(r"Task 2: Quadratic $h$-FEM Solutions ($a=100$)")
ax2a.set_xlabel("$x$")
ax2a.set_ylabel("$u(x)$")
ax2a.grid(True, ls="--", alpha=0.6)
ax2a.legend()

ax2b.loglog(h_a100, e_h2_a100, "rs-", lw=1.8, ms=7,
            label=rf"Quadratic ($p=2$), terminal $\beta_h = {beta_h2_a100:.2f}$")
ax2b.set_title(r"Task 2: Relative Error vs. Mesh Size $h$ ($a=100$)")
ax2b.set_xlabel("Mesh size $h$")
ax2b.set_ylabel(r"Relative error in energy norm $e$")
ax2b.grid(True, which="both", ls="--", alpha=0.6)
ax2b.legend()
fig2.tight_layout()

# --- Figure 3 (Task 4): p-FEM vs h-FEM for a = 1 ---
fig3, ax3 = plt.subplots(figsize=(8, 5.5))
ax3.loglog(n_h1_a1, e_h1_a1, "bo-", lw=1.8, ms=6, label=rf"$h$-FEM Linear ($p=1$), slope $= {slope_n_h1_a1:.2f}$")
ax3.loglog(n_h2_a1, e_h2_a1, "rs-", lw=1.8, ms=6, label=rf"$h$-FEM Quadratic ($p=2$), slope $= {slope_n_h2_a1:.2f}$")
ax3.loglog(n_p_a1, e_p_a1, "g^-", lw=2.0, ms=8, label=r"$p$-FEM ($N_e=2, p=1\dots 5$)")
for r in runs_p_a1:
    ax3.annotate(f" $p={r['p']}$", xy=(r["n_dofs"], r["err"]), fontsize=9, color="darkgreen")
ax3.set_title(r"Task 4: $p$-FEM vs. $h$-FEM Convergence ($a=1, x_b=0.5$)")
ax3.set_xlabel("Total DOFs $n$")
ax3.set_ylabel(r"Relative error in energy norm $e$")
ax3.grid(True, which="both", ls="--", alpha=0.6)
ax3.legend()
fig3.tight_layout()

# --- Figure 4 (Task 5): p-FEM Solutions & Convergence vs h-FEM for a = 100 ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5.2))

ax4a.plot(x_fine, exact_u(x_fine, 100.0, xb), "k-", lw=2.0, label="Exact $u(x)$ ($a=100$)")
for r in runs_p_a100:
    xc, uc = eval_linep_curve(r["x_nodes"], r["u_full"], r["elems"], r["p"])
    ax4a.plot(xc, uc, "--", lw=1.5, label=f"$p$-FEM ($N_e=5, p={r['p']}$)")
ax4a.set_title(r"Task 5: $p$-FEM Solutions ($N_e=5, a=100$)")
ax4a.set_xlabel("$x$")
ax4a.set_ylabel("$u(x)$")
ax4a.grid(True, ls="--", alpha=0.6)
ax4a.legend()

ax4b.loglog(n_h2_a100, e_h2_a100, "rs-", lw=1.8, ms=6,
            label=rf"$h$-FEM Quadratic ($p=2$), slope $= {slope_n_h2_a100:.2f}$")
ax4b.loglog(n_p_a100, e_p_a100, "g^-", lw=2.0, ms=8, label=r"$p$-FEM ($N_e=5, p=1\dots 5$)")
for r in runs_p_a100:
    ax4b.annotate(f" $p={r['p']}$", xy=(r["n_dofs"], r["err"]), fontsize=9, color="darkgreen")
ax4b.set_title(r"Task 5: $p$-FEM vs. $h$-FEM Convergence ($a=100$)")
ax4b.set_xlabel("Total DOFs $n$")
ax4b.set_ylabel(r"Relative error in energy norm $e$")
ax4b.grid(True, which="both", ls="--", alpha=0.6)
ax4b.legend()
fig4.tight_layout()

# --- Figure 5 (Task 6): Condition Number cond(K_ff) vs Total DOFs n ---
cond_h1 = [r["cond_K"] for r in runs_h1_a1]
cond_h2 = [r["cond_K"] for r in runs_h2_a1]
cond_p2 = [r["cond_K"] for r in runs_p_a1]
cond_p5 = [r["cond_K"] for r in runs_p_a100]

if all(c is not None for c in cond_h1 + cond_h2 + cond_p2 + cond_p5):
    gamma_h1 = log_slope(n_h1_a1, cond_h1)
    gamma_h2 = log_slope(n_h2_a1, cond_h2)
    gamma_p2 = log_slope(n_p_a1, cond_p2)
    gamma_p5 = log_slope(n_p_a100, cond_p5)

    print("\n=== Task 6: Condition Number Growth Rates (cond(K) ~ n^gamma) ===")
    print(f"h-FEM Linear (line2)          : gamma = {gamma_h1:.4f} (theoretical: 2.0)")
    print(f"h-FEM Quadratic (line3)       : gamma = {gamma_h2:.4f} (theoretical: 2.0)")
    print(f"p-FEM (Ne=2, p=1..5)          : gamma = {gamma_p2:.4f}")
    print(f"p-FEM (Ne=5, p=1..5)          : gamma = {gamma_p5:.4f}")

    fig5, ax5 = plt.subplots(figsize=(8, 5.5))
    ax5.loglog(n_h1_a1, cond_h1, "bo-", lw=1.8, ms=6, label=rf"$h$-FEM Linear, slope $= {gamma_h1:.2f}$")
    ax5.loglog(n_h2_a1, cond_h2, "rs-", lw=1.8, ms=6, label=rf"$h$-FEM Quadratic, slope $= {gamma_h2:.2f}$")
    ax5.loglog(n_p_a1, cond_p2, "g^-", lw=2.0, ms=8, label=rf"$p$-FEM ($N_e=2$), slope $= {gamma_p2:.2f}$")
    ax5.loglog(n_p_a100, cond_p5, "mv-", lw=2.0, ms=8, label=rf"$p$-FEM ($N_e=5$), slope $= {gamma_p5:.2f}$")
    ax5.set_title(r"Task 6: Stiffness Matrix Condition Number $\mathrm{cond}(\mathbf{K}_{ff})$ vs. DOFs $n$")
    ax5.set_xlabel("Total DOFs $n$")
    ax5.set_ylabel(r"Condition number $\mathrm{cond}(\mathbf{K}_{ff}) = \lambda_{\max}/\lambda_{\min}$")
    ax5.grid(True, which="both", ls="--", alpha=0.6)
    ax5.legend()
    fig5.tight_layout()
else:
    print("\n[Warning] cond_K missing in output. Ensure efem.py exports `cond_K` in _out.npz.")

plt.show()