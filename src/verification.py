"""
verification.py - Comprehensive verification suite covering all 10 compulsory Section 11 checks.
Implements:
1. Endpoint verification: Start and end match A and B exactly for both routes.
2. Independent recomputation of L, S, J matching solver outputs within float tolerance.
3. Toy Test 1 (Easy): Small grid with obstacle wall and known single gap. Dijkstra vs analytical hand calculation.
4. Toy Test 2 (Hard): Flat zero-risk terrain. SLSQP vs analytical straight line (kappa=0, L=|AB|).
5. Toy Test 3 (Hard): Centered Gaussian risk bump. SLSQP symmetric detour verification.
6. NetworkX formal graph shortest path cross-check against Dijkstra.
7. Dense constraint validation and density study (200, 1000, 5000 points).
8. Bicubic interpolator overshoot and masked-cell validation.
9. SLSQP multi-start spread and convergence rate analysis.
10. Metric grid vs WGS84 geodesic distance validation with pyproj.Geod.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize
import networkx as nx

from src.data import validate_distance
from src.easy import build_feasibility_mask, solve_dijkstra, cross_check_networkx
from src.hard import build_interpolators, verify_interpolator_overshoot, ContinuousTunnelOptimizer
from src.metrics import run_density_study


def run_toy_test_1():
    """
    Toy Test 1 (Easy): 11x11 grid with an impassable wall from row 1 to 8 at col 5,
    leaving a single gap at row 9, 10.
    Start at (5, 1), Goal at (5, 9).
    Direct horizontal path is blocked. Route must detour down to the gap at row 9 and back up.
    Verifies Dijkstra against exact hand calculation.
    """
    grid_shape = (11, 11)
    feasibility = np.ones(grid_shape, dtype=bool)

    # Wall at column 5, rows 0 to 8
    feasibility[0:9, 5] = False

    start_rc = (5, 1)
    goal_rc = (5, 9)

    dx, dy = 10.0, 10.0 # 10m grid
    dijk_res = solve_dijkstra(feasibility, start_rc, goal_rc, dx=dx, dy=dy)

    # Cross check with NetworkX
    nx_path, nx_len = cross_check_networkx(feasibility, start_rc, goal_rc, dx=dx, dy=dy)

    # Analytical hand distance:
    # From (5,1) to (9,5): delta_r = 4, delta_c = 4 -> 4 diagonal moves = 4 * 10 * sqrt(2) ~ 56.5685 m
    # From (9,5) to (5,9): delta_r = 4, delta_c = 4 -> 4 diagonal moves = 4 * 10 * sqrt(2) ~ 56.5685 m
    # Total analytical distance = 8 * 10 * sqrt(2) ~ 113.13708 m
    hand_dist = 8.0 * 10.0 * np.sqrt(2.0)

    error = abs(dijk_res["length_m"] - hand_dist)
    passed = error < 1e-6 and abs(dijk_res["length_m"] - nx_len) < 1e-9

    return {
        "name": "Toy Test 1 (Easy Problem - Grid Wall Detour)",
        "start": start_rc,
        "goal": goal_rc,
        "dijkstra_length_m": dijk_res["length_m"],
        "hand_calculated_m": hand_dist,
        "networkx_length_m": nx_len,
        "absolute_error_m": error,
        "passed": bool(passed),
        "path_rc": dijk_res["path_rc"],
        "wall_mask": (~feasibility).tolist()
    }


def run_toy_test_2():
    """
    Toy Test 2 (Hard): Flat terrain with zero risk everywhere.
    Start at (0, 0), Goal at (1000, 1000).
    The analytical global optimum is the straight line with L = 1000*sqrt(2) ~ 1414.21356 m,
    zero curvature kappa = 0, and zero terrain risk S = 0.
    Verifies that continuous SLSQP returns the straight line.
    """
    xA, yA = 0.0, 0.0
    xB, yB = 1000.0, 1000.0
    K = 5
    M = 100
    t_knots = np.linspace(0.0, 1.0, K + 2)
    t_samples = np.linspace(0.0, 1.0, M)

    # Initial guess: perturbed path away from straight line
    rng = np.random.default_rng(123)
    p0 = np.concatenate([
        xA + t_knots[1:-1] * (xB - xA) + rng.uniform(-100, 100, size=K),
        yA + t_knots[1:-1] * (yB - yA) + rng.uniform(-100, 100, size=K)
    ])

    def eval_spline(p):
        xk = np.concatenate([[xA], p[:K], [xB]])
        yk = np.concatenate([[yA], p[K:], [yB]])
        return CubicSpline(t_knots, xk, bc_type="natural"), CubicSpline(t_knots, yk, bc_type="natural")

    def objective(p):
        cs_x, cs_y = eval_spline(p)
        dx = cs_x(t_samples, 1)
        dy = cs_y(t_samples, 1)
        v = np.hypot(dx, dy)
        return float(np.trapezoid(v, t_samples))

    res = minimize(objective, p0, method="SLSQP", options={"maxiter": 200, "ftol": 1e-8})

    cs_x, cs_y = eval_spline(res.x)
    dx = cs_x(t_samples, 1)
    dy = cs_y(t_samples, 1)
    ddx = cs_x(t_samples, 2)
    ddy = cs_y(t_samples, 2)
    v2 = dx**2 + dy**2
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(v2**1.5, 1e-12)

    L_opt = float(np.trapezoid(np.hypot(dx, dy), t_samples))
    L_analytical = float(np.hypot(xB - xA, yB - yA))
    max_kappa = float(np.max(kappa))

    err_L = abs(L_opt - L_analytical)
    passed = err_L < 0.1 and max_kappa < 1e-4

    return {
        "name": "Toy Test 2 (Hard Problem - Flat Zero-Risk Terrain)",
        "L_numerical_m": L_opt,
        "L_analytical_m": L_analytical,
        "length_error_m": err_L,
        "max_curvature_m_inv": max_kappa,
        "passed": bool(passed),
        "path_x": cs_x(t_samples).tolist(),
        "path_y": cs_y(t_samples).tolist()
    }


def run_toy_test_3():
    """
    Toy Test 3 (Hard): Symmetric risk bump centered between A=(-500, 0) and B=(500, 0).
    Gaussian risk hill: R(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2)).
    The optimal route must detour around the bump, and due to symmetry across x=0 and y=0,
    the detour must be structurally symmetric (|y(t)| = |y(1-t)|).
    """
    xA, yA = -500.0, 0.0
    xB, yB = 500.0, 0.0
    K = 7
    M = 150
    t_knots = np.linspace(0.0, 1.0, K + 2)
    t_samples = np.linspace(0.0, 1.0, M)
    sigma = 100.0

    # Initial guess: small upward perturbation to break symmetry
    p0 = np.concatenate([
        xA + t_knots[1:-1] * (xB - xA),
        yA + t_knots[1:-1] * (yB - yA) + 80.0 * np.sin(np.pi * t_knots[1:-1])
    ])

    def eval_spline(p):
        xk = np.concatenate([[xA], p[:K], [xB]])
        yk = np.concatenate([[yA], p[K:], [yB]])
        return CubicSpline(t_knots, xk, bc_type="natural"), CubicSpline(t_knots, yk, bc_type="natural")

    def objective(p):
        cs_x, cs_y = eval_spline(p)
        xs = cs_x(t_samples)
        ys = cs_y(t_samples)
        dx = cs_x(t_samples, 1)
        dy = cs_y(t_samples, 1)
        v = np.hypot(dx, dy)
        R_vals = np.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))
        L = np.trapezoid(v, t_samples)
        S = np.trapezoid(R_vals * v, t_samples)
        return L + 3000.0 * S

    res = minimize(objective, p0, method="SLSQP", options={"maxiter": 200, "ftol": 1e-7})

    cs_x, cs_y = eval_spline(res.x)
    xs = cs_x(t_samples)
    ys = cs_y(t_samples)

    # Check detour: maximum deflection from straight line y=0 must be significant
    max_deflection = float(np.max(np.abs(ys)))

    # Check symmetry across midpoint t=0.5:
    # xs(t) should be anti-symmetric: xs(t) approx -xs(1-t)
    # ys(t) should be symmetric: ys(t) approx ys(1-t)
    xs_rev = -xs[::-1]
    ys_rev = ys[::-1]
    asym_x = float(np.max(np.abs(xs - xs_rev)))
    asym_y = float(np.max(np.abs(ys - ys_rev)))

    passed = (max_deflection > 50.0) and (asym_y < 2.0) and (asym_x < 2.0)

    return {
        "name": "Toy Test 3 (Hard Problem - Centered Risk Bump Symmetric Detour)",
        "max_deflection_m": max_deflection,
        "symmetry_error_x_m": asym_x,
        "symmetry_error_y_m": asym_y,
        "passed": bool(passed),
        "path_x": xs.tolist(),
        "path_y": ys.tolist()
    }


def plot_toy_tests(toy1, toy2, toy3, out_path="figures/fig10_verification_toy_tests.png"):
    """
    Visualizes the three verification toy problems.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Toy 1: Grid obstacle wall
    ax1 = axes[0]
    wall_grid = np.array(toy1["wall_mask"])
    ax1.imshow(wall_grid, cmap="gray_r", origin="upper")
    path_rc = np.array(toy1["path_rc"])
    ax1.plot(path_rc[:, 1], path_rc[:, 0], "r.-", linewidth=2.5, markersize=8, label="Dijkstra Route")
    ax1.scatter([toy1["start"][1]], [toy1["start"][0]], color="lime", s=100, zorder=5, label="Start A")
    ax1.scatter([toy1["goal"][1]], [toy1["goal"][0]], color="cyan", s=100, zorder=5, label="Goal B")
    ax1.set_title(f"Toy 1: Grid Obstacle Detour\nHand Calc: {toy1['hand_calculated_m']:.2f} m | Dijkstra: {toy1['dijkstra_length_m']:.2f} m", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Toy 2: Flat zero risk
    ax2 = axes[1]
    ax2.plot(toy2["path_x"], toy2["path_y"], "b-", linewidth=2.5, label="SLSQP Spline")
    ax2.plot([0, 1000], [0, 1000], "r--", linewidth=1.5, label="Analytical Line")
    ax2.scatter([0, 1000], [0, 1000], color=["lime", "cyan"], s=90, zorder=5)
    ax2.set_title(f"Toy 2: Flat Zero-Risk Terrain\nAnalytical: {toy2['L_analytical_m']:.2f} m | SLSQP: {toy2['L_numerical_m']:.2f} m", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Toy 3: Centered risk bump
    ax3 = axes[2]
    X_grid, Y_grid = np.meshgrid(np.linspace(-500, 500, 100), np.linspace(-250, 250, 100))
    R_bump = np.exp(-(X_grid**2 + Y_grid**2) / (2.0 * 100.0**2))
    ax3.contourf(X_grid, Y_grid, R_bump, levels=20, cmap="YlOrRd")
    ax3.plot(toy3["path_x"], toy3["path_y"], "b-", linewidth=2.5, label="Optimized Detour")
    ax3.plot([-500, 500], [0, 0], "k--", alpha=0.6, label="Straight Line")
    ax3.scatter([-500, 500], [0, 0], color=["lime", "cyan"], s=90, zorder=5)
    ax3.set_title(f"Toy 3: Centered Risk Bump\nMax Detour: {toy3['max_deflection_m']:.1f} m | Symmetry Error: {toy3['symmetry_error_y_m']:.2e} m", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Toy test verification plot saved to {out_path}.")


def run_full_verification(data_dict, features_dict, cfg, easy_res, hard_res, out_dir="results"):
    """
    Runs all 10 compulsory verification checks and records formal audit log.
    """
    os.makedirs(out_dir, exist_ok=True)
    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]

    print("--- Running Verification Suite (Section 11) ---")
    results = {}

    # Check 1: Start and end match A and B exactly
    xA, yA = pt_A["utm_x"], pt_A["utm_y"]
    xB, yB = pt_B["utm_x"], pt_B["utm_y"]
    easy_start_match = (abs(easy_res["path_x"][0] - xA) < 1.0 and abs(easy_res["path_y"][0] - yA) < 1.0)
    easy_end_match = (abs(easy_res["path_x"][-1] - xB) < 1.0 and abs(easy_res["path_y"][-1] - yB) < 1.0)
    hard_start_match = (abs(hard_res["path_x"][0] - xA) < 1e-3 and abs(hard_res["path_y"][0] - yA) < 1e-3)
    hard_end_match = (abs(hard_res["path_x"][-1] - xB) < 1e-3 and abs(hard_res["path_y"][-1] - yB) < 1e-3)

    chk1_pass = easy_start_match and easy_end_match and hard_start_match and hard_end_match
    results["check_1_endpoints"] = {
        "description": "Both routes start exactly at A and end exactly at B",
        "easy_start_error_m": float(np.hypot(easy_res["path_x"][0] - xA, easy_res["path_y"][0] - yA)),
        "easy_end_error_m": float(np.hypot(easy_res["path_x"][-1] - xB, easy_res["path_y"][-1] - yB)),
        "hard_start_error_m": float(np.hypot(hard_res["path_x"][0] - xA, hard_res["path_y"][0] - yA)),
        "hard_end_error_m": float(np.hypot(hard_res["path_x"][-1] - xB, hard_res["path_y"][-1] - yB)),
        "passed": bool(chk1_pass)
    }

    # Check 2: Independent recomputation of L, S, J with separate code
    # Independent polyline integrator
    easy_dx = np.diff(easy_res["path_x"])
    easy_dy = np.diff(easy_res["path_y"])
    easy_segs = np.hypot(easy_dx, easy_dy)
    L_easy_recalc = float(np.sum(easy_segs))

    R_grid = features_dict["R"]
    r_nodes = [R_grid[r, c] for r, c in easy_res["path_rc"]]
    r_mid = 0.5 * (np.array(r_nodes[:-1]) + np.array(r_nodes[1:]))
    S_easy_recalc = float(np.sum(r_mid * easy_segs))
    J_easy_recalc = float(cfg["hard_solver"]["alpha"] * L_easy_recalc + cfg["hard_solver"]["beta"] * S_easy_recalc)

    err_L_easy = abs(L_easy_recalc - easy_res["L_E"])
    err_S_easy = abs(S_easy_recalc - easy_res["S_E"])
    err_J_easy = abs(J_easy_recalc - easy_res["J_E"])

    chk2_pass = err_L_easy < 1e-6 and err_S_easy < 1e-6 and err_J_easy < 1e-4
    results["check_2_independent_recomputation"] = {
        "description": "Recompute L, S, J with independent code matching solver outputs",
        "easy_L_error": err_L_easy,
        "easy_S_error": err_S_easy,
        "easy_J_error": err_J_easy,
        "passed": bool(chk2_pass)
    }

    # Check 3: Toy Test 1 (Easy)
    toy1 = run_toy_test_1()
    results["check_3_toy_test_1"] = toy1

    # Check 4: Toy Test 2 (Hard)
    toy2 = run_toy_test_2()
    results["check_4_toy_test_2"] = toy2

    # Check 5: Toy Test 3 (Hard)
    toy3 = run_toy_test_3()
    results["check_5_toy_test_3"] = toy3

    # Plot toy tests
    plot_toy_tests(toy1, toy2, toy3)

    # Check 6: Dijkstra vs NetworkX cross-check
    f_mask = build_feasibility_mask(
        depth_arr=features_dict["d"],
        slope_arr=features_dict["s"],
        mask_land=data_dict["mask_land"],
        d_min=cfg["constraints"]["d_min"],
        d_max=cfg["constraints"]["d_max"],
        s_max=cfg["constraints"]["s_max"]
    )
    nx_path, nx_len = cross_check_networkx(
        f_mask, (pt_A["grid_row"], pt_A["grid_col"]), (pt_B["grid_row"], pt_B["grid_col"])
    )
    diff_nx = abs(easy_res["L_E"] - nx_len)
    results["check_6_networkx_crosscheck"] = {
        "description": "Cross-check Dijkstra against NetworkX shortest_path on the same graph",
        "dijkstra_length_m": easy_res["L_E"],
        "networkx_length_m": nx_len,
        "difference_m": diff_nx,
        "passed": bool(diff_nx < 1e-9)
    }

    # Check 7: Dense validation and density study (200, 1000, 5000 points)
    interpolators = build_interpolators(data_dict, features_dict)
    opt = ContinuousTunnelOptimizer(
        interpolators=interpolators,
        pt_A=pt_A,
        pt_B=pt_B,
        cfg=cfg,
        K=cfg["hard_solver"]["num_control_points"],
        M=cfg["hard_solver"]["num_samples"],
        alpha=cfg["hard_solver"]["alpha"],
        beta=cfg["hard_solver"]["beta"]
    )
    p_opt = np.concatenate([hard_res["control_points_x"], hard_res["control_points_y"]])
    density_study = run_density_study(opt, p_opt, sample_counts=[200, 1000, 5000])
    results["check_7_density_study"] = {
        "description": "Dense validation at 200, 1000, and 5000 sample points",
        "study": density_study,
        "passed": all(d["max_viol_depth_m"] < 1e-3 and d["max_viol_slope"] < 1e-4 for d in density_study)
    }

    # Check 8: Interpolator overshoot and domain validity
    overshoot_rep = verify_interpolator_overshoot(interpolators, features_dict)
    results["check_8_interpolator_checks"] = {
        "description": "Bicubic interpolator overshoot and domain validity checks",
        "overshoot": overshoot_rep,
        "domain_valid": hard_res["domain_valid"],
        "passed": bool(hard_res["domain_valid"] and all(v["acceptable"] for v in overshoot_rep.values()))
    }

    # Check 9: Multi-start spread for SLSQP
    m_stats = hard_res["multi_start_stats"]
    results["check_9_multi_start_spread"] = {
        "description": "Multi-start SLSQP spread and convergence rate",
        "total_starts": m_stats["total_starts"],
        "converged_starts": m_stats["converged_starts"],
        "convergence_rate": m_stats["convergence_rate"],
        "j_spread": m_stats["j_spread"],
        "passed": bool(m_stats["converged_starts"] >= 1)
    }

    # Check 10: Projected grid vs Geodesic distance validation
    dist_val = validate_distance(pt_A, pt_B, crs_projected=cfg["metadata"]["crs_projected"])
    results["check_10_geodesic_distance"] = {
        "description": "Compare A-B distance in UTM grid with true WGS84 geodesic distance (pyproj.Geod)",
        "distance_grid_m": dist_val["distance_grid_m"],
        "distance_geodesic_m": dist_val["distance_geodesic_m"],
        "difference_m": dist_val["difference_m"],
        "difference_percent": dist_val["difference_percent"],
        "passed": bool(dist_val["passed"])
    }

    # Overall Audit Status
    all_passed = all(chk["passed"] for chk in results.values())
    results["all_checks_passed"] = all_passed

    report_path = os.path.join(out_dir, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    summary_path = os.path.join(out_dir, "verification_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("UNDERWATER TUNNEL ROUTE OPTIMIZATION - VERIFICATION AUDIT (SECTION 11)\n")
        f.write("="*70 + "\n\n")
        for k, v in results.items():
            if k == "all_checks_passed":
                continue
            status = "PASS [✓]" if v["passed"] else "FAIL [✗]"
            f.write(f"[{status}] {k.upper()}: {v.get('description', v.get('name', ''))}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"OVERALL VERIFICATION STATUS: {'ALL CHECKS PASSED [SUCCESS]' if all_passed else 'SOME CHECKS FAILED'}\n")
        f.write("="*70 + "\n")

    print(f"Verification audit completed. Overall Passed: {all_passed}")
    print(f"Report written to {report_path} and {summary_path}.")
    return results


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    from src.features import derive_features
    from src.easy import solve_easy_problem
    from src.hard import solve_hard_problem

    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)

    easy_res, _ = solve_easy_problem(data, feat, cfg)
    hard_res = solve_hard_problem(data, feat, cfg, easy_res=easy_res)

    verif = run_full_verification(data, feat, cfg, easy_res, hard_res)
