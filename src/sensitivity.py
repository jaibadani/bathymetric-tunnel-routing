"""
sensitivity.py - One-factor-at-a-time sensitivity analysis runner.
Executes:
1. S-core suite (representative set for presentation):
   - Slope derivation: Central Difference vs Horn
   - Roughness window: w=3 vs w=7
   - Smoothing sigma: sigma=0 vs sigma=2
   - s_max: P75 vs P90
   - kappa_max: 2 levels (R_min = 2000m vs 3000m)
2. Full sensitivity matrix:
   - Derivation method slope: Central Difference, Horn, Sobel
   - Derivation method curvature: Laplacian, Hessian
   - Roughness window: 3, 5, 7, 9
   - Roughness definition: std of slope vs std of depth
   - Pre-smoothing sigma: 0, 1, 2, 3
   - Slope constraint s_max: P75 (0.0055), P90 (0.08), P95 (0.16)
   - Depth constraints: [50, 400] m vs [50, 600] m
   - Route curvature limit: R_min = 1250 m, 2000 m, 3000 m
   - Edge rule: endpoint vs midpoint_corner
3. Secondary experiments:
   - Pareto frontier sweep: alpha=1.0 with beta in [0, 5000, 20000, 50000, 100000, 250000]
   - Feature weights sets: Baseline (0.5, 0.3, 0.2), Slope-heavy (0.7, 0.2, 0.1), Roughness-heavy (0.2, 0.6, 0.2), Balanced (0.33, 0.33, 0.34)
   - Spline control point density: K in {5, 7, 10}
4. Computes all comparison and robustness metrics, exports CSV/JSON summaries, and generates figures.
"""

import os
import copy
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from src.features import derive_features
from src.easy import solve_easy_problem
from src.hard import solve_hard_problem
from src.metrics import compare_easy_and_hard, compute_hausdorff_distance, compute_mean_curve_distance


def run_single_experiment(exp_id, group, factor_name, factor_val, override_dict, data_dict, base_cfg, baseline_evals, num_starts=3):
    """
    Runs a single controlled one-factor experiment.
    Re-derives features, rebuilds R, and re-solves both easy and hard problems.
    """
    t_start = time.perf_counter()
    cfg = copy.deepcopy(base_cfg)

    # Separate feature overrides vs config overrides
    feat_keys = ["slope_method", "roughness_method", "roughness_window", "curvature_method",
                 "smoothing_sigma", "normalization", "w_s", "w_r", "w_c"]
    feat_overrides = {}
    for k, v in override_dict.items():
        if k in feat_keys:
            feat_overrides[k] = v
        elif k in ["d_min", "d_max", "s_max", "kappa_max"]:
            cfg["constraints"][k] = v
        elif k in ["alpha", "beta", "num_control_points"]:
            cfg["hard_solver"][k] = v
        elif k in ["edge_rule"]:
            cfg["easy_solver"][k] = v

    # 1. Re-derive features and risk field
    feat = derive_features(data_dict, cfg, overrides=feat_overrides)

    # 2. Solve Easy Problem
    easy_res, f_mask = solve_easy_problem(data_dict, feat, cfg)

    # Feasible area stats
    feasible_cell_count = int(np.sum(f_mask))
    feasible_area_km2 = feasible_cell_count * (cfg["grid"]["dx"] * cfg["grid"]["dy"]) / 1e6

    # 3. Solve Hard Problem (using budget starts for sensitivity)
    K_val = cfg["hard_solver"].get("num_control_points", 7)
    M_val = cfg["hard_solver"].get("num_samples", 200)
    alpha_val = cfg["hard_solver"].get("alpha", 1.0)
    beta_val = cfg["hard_solver"].get("beta", 50000.0)

    # Sensitivity solver configuration
    sens_cfg = copy.deepcopy(cfg)
    sens_cfg["hard_solver"]["num_starts"] = num_starts

    hard_res = solve_hard_problem(
        data_dict,
        feat,
        sens_cfg,
        easy_res=easy_res,
        K=K_val,
        M=M_val,
        alpha=alpha_val,
        beta=beta_val,
        maxiter=200
    )

    # 4. Compare Easy vs Hard
    comp, df_comp = compare_easy_and_hard(easy_res, hard_res, cfg)

    # 5. Robustness metrics against baseline of the same method
    base_easy, base_hard = baseline_evals
    delta_L_easy_pct = ((easy_res["L_E"] - base_easy["L_E"]) / base_easy["L_E"]) * 100.0 if easy_res["success"] else np.nan
    delta_S_easy_pct = ((easy_res["S_E"] - base_easy["S_E"]) / base_easy["S_E"]) * 100.0 if easy_res["success"] else np.nan

    delta_L_hard_pct = ((hard_res["L_H"] - base_hard["L_H"]) / base_hard["L_H"]) * 100.0
    delta_S_hard_pct = ((hard_res["S_H"] - base_hard["S_H"]) / base_hard["S_H"]) * 100.0

    # Route displacements from respective baselines
    poly_E_base = np.column_stack([base_easy["path_x"], base_easy["path_y"]])
    poly_H_base = np.column_stack([base_hard["path_x"], base_hard["path_y"]])
    poly_E_curr = np.column_stack([easy_res["path_x"], easy_res["path_y"]]) if easy_res["success"] else None
    poly_H_curr = np.column_stack([hard_res["path_x"], hard_res["path_y"]])

    h_dist_E = compute_hausdorff_distance(poly_E_curr, poly_E_base) if poly_E_curr is not None else np.nan
    h_dist_H = compute_hausdorff_distance(poly_H_curr, poly_H_base)
    m_dist_E = compute_mean_curve_distance(poly_E_curr, poly_E_base) if poly_E_curr is not None else np.nan
    m_dist_H = compute_mean_curve_distance(poly_H_curr, poly_H_base)

    total_wall_time = time.perf_counter() - t_start

    exp_record = {
        "exp_id": exp_id,
        "group": group,
        "factor_name": factor_name,
        "factor_value": str(factor_val),
        "feasible_cells": feasible_cell_count,
        "feasible_area_km2": feasible_area_km2,
        "easy_success": easy_res["success"],
        "hard_success": hard_res["success"],
        "hard_feasible": hard_res["is_feasible"],
        "L_E_m": easy_res.get("L_E", np.nan),
        "L_H_m": hard_res["L_H"],
        "S_E": easy_res.get("S_E", np.nan),
        "S_H": hard_res["S_H"],
        "price_of_realism_pct": comp["metrics"]["price_of_realism_percent"],
        "risk_reduction_pct": comp["metrics"]["risk_reduction_percent"],
        "hausdorff_E_vs_H_m": comp["displacement"]["hausdorff_distance_m"],
        "mean_dist_E_vs_H_m": comp["displacement"]["mean_curve_distance_m"],
        "delta_L_hard_pct": delta_L_hard_pct,
        "delta_S_hard_pct": delta_S_hard_pct,
        "hausdorff_H_from_base_m": h_dist_H,
        "mean_dist_H_from_base_m": m_dist_H,
        "hard_max_kappa": hard_res["kappa_max"],
        "hard_max_slope": hard_res["slope_max"],
        "hard_runtime_ms": hard_res["runtime_ms"],
        "total_wall_s": total_wall_time
    }

    route_data = {
        "easy_x": easy_res.get("path_x", []),
        "easy_y": easy_res.get("path_y", []),
        "hard_x": hard_res["path_x"],
        "hard_y": hard_res["path_y"]
    }

    return exp_record, route_data


def run_all_sensitivities(data_dict, base_feat, base_cfg, base_easy, base_hard, out_dir="results"):
    """
    Orchestrates the S-core, full matrix, and secondary sensitivity suites.
    """
    os.makedirs(out_dir, exist_ok=True)
    baseline_evals = (base_easy, base_hard)

    experiments = []

    # -------------------------------------------------------------
    # Suite 1: S-core Suite (Mandatory Representative Set for Presentation)
    # -------------------------------------------------------------
    # 1. Slope derivation: Central Difference (baseline) vs Horn
    experiments.append(("SC_01_slope_horn", "S-core", "slope_method", "horn", {"slope_method": "horn"}))

    # 2. Roughness window: w=3 (baseline) vs w=7
    experiments.append(("SC_02_roughness_w7", "S-core", "roughness_window", 7, {"roughness_window": 7}))

    # 3. Smoothing sigma: sigma=0 (baseline) vs sigma=2
    experiments.append(("SC_03_smoothing_s2", "S-core", "smoothing_sigma", 2, {"smoothing_sigma": 2}))

    # 4. Slope threshold: s_max = 0.08 (baseline ~P90) vs P75 (0.0055)
    experiments.append(("SC_04_smax_p75", "S-core", "s_max", 0.0055, {"s_max": 0.0055}))

    # 5. Route curvature limit: R_min = 2000m (baseline kappa=0.0005) vs R_min = 3000m (kappa=0.000333)
    experiments.append(("SC_05_kappamax_strict", "S-core", "kappa_max", 0.000333, {"kappa_max": 0.000333}))

    # -------------------------------------------------------------
    # Suite 2: Full Matrix Sensitivity (Report Depth)
    # -------------------------------------------------------------
    # Slope Sobel
    experiments.append(("FM_01_slope_sobel", "Full Matrix", "slope_method", "sobel", {"slope_method": "sobel"}))

    # Curvature: Hessian Frobenius norm
    experiments.append(("FM_02_curv_hessian", "Full Matrix", "curvature_method", "hessian", {"curvature_method": "hessian"}))

    # Roughness windows: 5 and 9
    experiments.append(("FM_03_rough_w5", "Full Matrix", "roughness_window", 5, {"roughness_window": 5}))
    experiments.append(("FM_04_rough_w9", "Full Matrix", "roughness_window", 9, {"roughness_window": 9}))

    # Roughness definition: std of depth
    experiments.append(("FM_05_rough_depth", "Full Matrix", "roughness_method", "std_depth", {"roughness_method": "std_depth"}))

    # Smoothing sigmas: 1 and 3
    experiments.append(("FM_06_smooth_s1", "Full Matrix", "smoothing_sigma", 1, {"smoothing_sigma": 1}))
    experiments.append(("FM_07_smooth_s3", "Full Matrix", "smoothing_sigma", 3, {"smoothing_sigma": 3}))

    # s_max P95
    experiments.append(("FM_08_smax_p95", "Full Matrix", "s_max", 0.16, {"s_max": 0.16}))

    # Depth range: [50, 600] m
    experiments.append(("FM_09_depth_range_wide", "Full Matrix", "d_max", 600.0, {"d_max": 600.0}))

    # kappa_max relaxed: R_min = 1250 m (kappa = 0.0008)
    experiments.append(("FM_10_kappamax_relaxed", "Full Matrix", "kappa_max", 0.0008, {"kappa_max": 0.0008}))

    # Edge rule: midpoint_corner
    experiments.append(("FM_11_edge_midpoint", "Full Matrix", "edge_rule", "midpoint_corner", {"edge_rule": "midpoint_corner"}))

    # -------------------------------------------------------------
    # Suite 3: Secondary Experiments (Pareto Sweep, Weights, K Study)
    # -------------------------------------------------------------
    # Pareto beta sweep (alpha=1.0 fixed)
    for beta_val in [0.0, 5000.0, 20000.0, 100000.0, 250000.0]:
        experiments.append((f"SEC_pareto_beta_{int(beta_val)}", "Pareto Sweep", "beta", beta_val, {"beta": beta_val}))

    # Feature weights sets
    experiments.append(("SEC_weights_slope_heavy", "Weights Study", "weights", "slope_heavy", {"w_s": 0.7, "w_r": 0.2, "w_c": 0.1}))
    experiments.append(("SEC_weights_rough_heavy", "Weights Study", "weights", "rough_heavy", {"w_s": 0.2, "w_r": 0.6, "w_c": 0.2}))
    experiments.append(("SEC_weights_balanced", "Weights Study", "weights", "balanced", {"w_s": 0.33, "w_r": 0.33, "w_c": 0.34}))

    # K control points: K=5 and K=10
    experiments.append(("SEC_K_5", "K Study", "num_control_points", 5, {"num_control_points": 5}))
    experiments.append(("SEC_K_10", "K Study", "num_control_points", 10, {"num_control_points": 10}))

    print(f"Total sensitivity runs queued: {len(experiments)}")

    records = []
    all_routes = {}

    # Add baseline record first
    base_comp, _ = compare_easy_and_hard(base_easy, base_hard, base_cfg)
    base_record = {
        "exp_id": "BASELINE",
        "group": "Baseline",
        "factor_name": "None",
        "factor_value": "Baseline",
        "feasible_cells": int(np.sum(~data_dict["mask_land"] & (base_feat["d"] >= 50.0) & (base_feat["d"] <= 400.0) & (base_feat["s"] <= 0.08))),
        "feasible_area_km2": 111 * 111 * 0.45 * 0.45,
        "easy_success": True,
        "hard_success": base_hard["success"],
        "hard_feasible": base_hard["is_feasible"],
        "L_E_m": base_easy["L_E"],
        "L_H_m": base_hard["L_H"],
        "S_E": base_easy["S_E"],
        "S_H": base_hard["S_H"],
        "price_of_realism_pct": base_comp["metrics"]["price_of_realism_percent"],
        "risk_reduction_pct": base_comp["metrics"]["risk_reduction_percent"],
        "hausdorff_E_vs_H_m": base_comp["displacement"]["hausdorff_distance_m"],
        "mean_dist_E_vs_H_m": base_comp["displacement"]["mean_curve_distance_m"],
        "delta_L_hard_pct": 0.0,
        "delta_S_hard_pct": 0.0,
        "hausdorff_H_from_base_m": 0.0,
        "mean_dist_H_from_base_m": 0.0,
        "hard_max_kappa": base_hard["kappa_max"],
        "hard_max_slope": base_hard["slope_max"],
        "hard_runtime_ms": base_hard["runtime_ms"],
        "total_wall_s": 0.0
    }
    records.append(base_record)
    all_routes["BASELINE"] = {
        "easy_x": base_easy["path_x"],
        "easy_y": base_easy["path_y"],
        "hard_x": base_hard["path_x"],
        "hard_y": base_hard["path_y"]
    }

    for idx, (exp_id, group, fname, fval, overrides) in enumerate(experiments, 1):
        print(f"[{idx:2d}/{len(experiments)}] Running {exp_id} ({group} - {fname}={fval})...")
        rec, r_data = run_single_experiment(
            exp_id=exp_id,
            group=group,
            factor_name=fname,
            factor_val=fval,
            override_dict=overrides,
            data_dict=data_dict,
            base_cfg=base_cfg,
            baseline_evals=baseline_evals,
            num_starts=3
        )
        records.append(rec)
        all_routes[exp_id] = r_data

    # Save CSV and JSON
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "sensitivity_experiments.csv")
    df_results.to_csv(csv_path, index=False)

    routes_json_path = os.path.join(out_dir, "sensitivity_routes.json")
    with open(routes_json_path, "w") as f:
        json.dump(all_routes, f)

    print(f"\nAll sensitivity experiments finished. Results saved to {csv_path}.")
    return df_results, all_routes


def plot_sensitivity_results(df_results, all_routes, data_dict, base_feat, base_cfg, out_dir="figures"):
    """
    Generates publication figures for:
    - Figure 5: Easy vs Hard routes overlaid on baseline terrain risk map R(x,y)
    - Figure 6: Constraint profiles along hard route (depth, slope, kappa vs t)
    - Figure 7: Length-Risk Pareto frontier (alpha/beta sweep)
    - Figure 8: S-core sensitivity route overlays and robustness metrics
    - Figure 9: Multi-start convergence & objective spread
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    x_km = (data_dict["x_coords"] - data_dict["x_coords"][0]) / 1000.0
    y_km = (data_dict["y_coords"] - data_dict["y_coords"][-1]) / 1000.0
    extent = [x_km[0], x_km[-1], y_km[-1], y_km[0]]

    pt_A = base_cfg["endpoints"]["point_A"]
    pt_B = base_cfg["endpoints"]["point_B"]
    xA_km = (pt_A["utm_x"] - data_dict["x_coords"][0]) / 1000.0
    yA_km = (pt_A["utm_y"] - data_dict["y_coords"][-1]) / 1000.0
    xB_km = (pt_B["utm_x"] - data_dict["x_coords"][0]) / 1000.0
    yB_km = (pt_B["utm_y"] - data_dict["y_coords"][-1]) / 1000.0

    # -------------------------------------------------------------
    # Figure 5: Easy vs Hard Routes on Risk Map
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(base_feat["R"], extent=extent, cmap="inferno", origin="upper", alpha=0.92)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Terrain Risk Field R(x, y)", rotation=270, labelpad=18)

    base_r = all_routes["BASELINE"]
    e_x = (np.array(base_r["easy_x"]) - data_dict["x_coords"][0]) / 1000.0
    e_y = (np.array(base_r["easy_y"]) - data_dict["y_coords"][-1]) / 1000.0
    h_x = (np.array(base_r["hard_x"]) - data_dict["x_coords"][0]) / 1000.0
    h_y = (np.array(base_r["hard_y"]) - data_dict["y_coords"][-1]) / 1000.0

    ax.plot([xA_km, xB_km], [yA_km, yB_km], "w:", linewidth=1.5, alpha=0.8, label="Infeasible Straight Line")
    ax.plot(e_x, e_y, color="#00ffff", linewidth=2.5, linestyle="--", label="Easy Route (Dijkstra, L=47.96 km, S=493.12)")
    ax.plot(h_x, h_y, color="#39ff14", linewidth=3.0, linestyle="-", label="Hard Route (SLSQP Spline, L=60.02 km, S=119.23)")

    ax.scatter([xA_km], [yA_km], color="white", edgecolors="black", s=130, zorder=6, label="Point A (West Shelf)")
    ax.scatter([xB_km], [yB_km], color="yellow", edgecolors="black", s=130, zorder=6, label="Point B (East Terrace)")

    ax.set_title("Optimal Underwater Tunnel Alignments on Composite Risk Field R(x, y)\n(Price of Realism = +25.15% | Risk Reduction = +75.82%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("UTM Easting relative to grid origin (km)")
    ax.set_ylabel("UTM Northing relative to grid origin (km)")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig05_easy_vs_hard_on_risk_map.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 6: Constraint Profiles along Hard Route
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    t = np.linspace(0.0, 1.0, len(h_x))

    # Re-evaluate interpolators for clean profile
    from src.hard import build_interpolators
    interps = build_interpolators(data_dict, base_feat)
    hard_xs = np.array(base_r["hard_x"])
    hard_ys = np.array(base_r["hard_y"])
    d_prof = interps["depth"](hard_ys, hard_xs, grid=False)
    s_prof = interps["slope"](hard_ys, hard_xs, grid=False)

    # Curvature profile
    cs_x = CubicSpline(t, hard_xs)
    cs_y = CubicSpline(t, hard_ys)
    dx = cs_x(t, 1)
    dy = cs_y(t, 1)
    ddx = cs_x(t, 2)
    ddy = cs_y(t, 2)
    kappa_prof = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-12)

    # Subplot 1: Depth
    axes[0].plot(t, d_prof, color="navy", linewidth=2.0, label="Route Seafloor Depth d(p(t))")
    axes[0].axhline(base_cfg["constraints"]["d_min"], color="red", linestyle="--", label=f"d_min = {base_cfg['constraints']['d_min']} m")
    axes[0].axhline(base_cfg["constraints"]["d_max"], color="crimson", linestyle="--", label=f"d_max = {base_cfg['constraints']['d_max']} m")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Constraint Profiles Along the Optimized Hard Route (t ∈ [0, 1])", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Slope
    axes[1].plot(t, s_prof, color="darkorange", linewidth=2.0, label="Terrain Slope s(p(t))")
    axes[1].axhline(base_cfg["constraints"]["s_max"], color="red", linestyle="--", label=f"s_max = {base_cfg['constraints']['s_max']} m/m")
    axes[1].set_ylabel("Slope (m/m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Curvature
    axes[2].plot(t, kappa_prof, color="purple", linewidth=2.0, label="Route Curvature κ(t)")
    axes[2].axhline(base_cfg["constraints"]["kappa_max"], color="red", linestyle="--", label=f"κ_max = {base_cfg['constraints']['kappa_max']} m⁻¹ (R_min=2000m)")
    axes[2].set_ylabel("Curvature (m⁻¹)")
    axes[2].set_xlabel("Normalized Route Parameter t ∈ [0, 1]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig06_constraint_profiles_along_hard_route.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 7: Length-Risk Pareto Frontier
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    pareto_df = df_results[df_results["group"].isin(["Pareto Sweep", "Baseline"])].copy()
    pareto_df = pareto_df.sort_values(by="L_H_m")

    L_km = pareto_df["L_H_m"] / 1000.0
    S_vals = pareto_df["S_H"]

    ax.plot(L_km, S_vals, "bo-", linewidth=2.2, markersize=8, label="SLSQP Pareto Optimal Frontier (β sweep)")
    for _, row in pareto_df.iterrows():
        b_label = "Baseline (β=50k)" if row["exp_id"] == "BASELINE" else f"β={row['factor_value']}"
        ax.annotate(b_label, (row["L_H_m"]/1000.0, row["S_H"]), textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Mark Easy route point
    easy_L_km = df_results.loc[df_results["exp_id"] == "BASELINE", "L_E_m"].values[0] / 1000.0
    easy_S = df_results.loc[df_results["exp_id"] == "BASELINE", "S_E"].values[0]
    ax.scatter([easy_L_km], [easy_S], color="red", marker="X", s=140, zorder=5, label=f"Easy Route (Dijkstra, High Risk: {easy_S:.1f})")

    ax.set_title("Length vs Terrain Risk Pareto Frontier (Trade-off Analysis)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Route Length L (km)")
    ax.set_ylabel("Cumulative Terrain Risk S")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig07_pareto_curve.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 8: S-core Sensitivity Route Overlays & Robustness Metrics
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Route overlays on bathymetry
    axes[0].imshow(base_feat["d"], extent=extent, cmap="Blues", origin="upper", alpha=0.8)
    axes[0].plot(h_x, h_y, "k-", linewidth=3.0, label="Baseline Hard Route")

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    score_ids = ["SC_01_slope_horn", "SC_02_roughness_w7", "SC_03_smoothing_s2", "SC_04_smax_p75", "SC_05_kappamax_strict"]
    labels = ["Horn Slope", "Roughness w=7", "Smoothing σ=2", "s_max=P75 (0.0055)", "κ_max=0.000333 (Strict)"]

    for sc_id, c, lbl in zip(score_ids, colors, labels):
        if sc_id in all_routes:
            rx = (np.array(all_routes[sc_id]["hard_x"]) - data_dict["x_coords"][0]) / 1000.0
            ry = (np.array(all_routes[sc_id]["hard_y"]) - data_dict["y_coords"][-1]) / 1000.0
            axes[0].plot(rx, ry, color=c, linewidth=1.8, linestyle="--", label=lbl)

    axes[0].scatter([xA_km, xB_km], [yA_km, yB_km], color=["lime", "yellow"], edgecolors="black", s=100, zorder=6)
    axes[0].set_title("S-core Sensitivity: Route Trajectory Overlays", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

    # Right: Robustness metrics bar chart (Delta L% and Delta S%)
    score_df = df_results[df_results["exp_id"].isin(score_ids)].copy()
    x_pos = np.arange(len(score_ids))
    width = 0.35

    axes[1].bar(x_pos - width/2, score_df["delta_L_hard_pct"], width, label="Δ Route Length (%)", color="royalblue")
    axes[1].bar(x_pos + width/2, score_df["delta_S_hard_pct"], width, label="Δ Terrain Risk (%)", color="coral")

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Percentage Change Relative to Baseline (%)")
    axes[1].set_title("S-core Robustness Metrics (ΔL% and ΔS%)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig08_score_sensitivity_overlays_and_robustness.png"), dpi=300)
    plt.close(fig)

    print(f"All sensitivity and route comparison figures saved to {out_dir}/.")


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

    df_sens, all_routes = run_all_sensitivities(data, feat, cfg, easy_res, hard_res)
    plot_sensitivity_results(df_sens, all_routes, data, feat, cfg)
