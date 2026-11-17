"""
run_all.py - Master Pipeline Execution Script.
Executes the end-to-end Underwater Tunnel Route Optimization pipeline:
1. Data ingestion, projected CRS resampling, masking, and geodesic distance validation.
2. Feature derivation (slope, roughness, curvature) and risk field construction.
3. Deep data analysis: descriptive statistics, correlations, Moran's I, PCA, and Figures 1-4.
4. Easy problem optimization: 8-neighbor Dijkstra shortest path with NetworkX verification.
5. Hard problem optimization: Multi-start continuous cubic spline with SciPy SLSQP.
6. Unified comparison, Price of Realism, and robustness metrics calculation.
7. Full verification audit (Section 11) with Toy Tests 1, 2, 3, density study, and Figure 10.
8. Sensitivity analysis: S-core, full matrix, Pareto frontier sweep, and Figures 5-9.
"""

import sys
import os
import json
import time

from src.data import load_config, fetch_and_reproject_gebco, validate_distance
from src.features import derive_features
from src.analysis import run_full_data_analysis
from src.easy import solve_easy_problem
from src.hard import solve_hard_problem
from src.metrics import compare_easy_and_hard
from src.verification import run_full_verification
from src.sensitivity import run_all_sensitivities, plot_sensitivity_results


def main():
    t_start = time.perf_counter()
    print("=" * 75)
    print(" UNDERWATER TUNNEL ROUTE OPTIMIZATION - MASTER BUILD EXECUTION")
    print("=" * 75)

    # 1. Load config
    print("\n[Step 1/6] Loading Configuration and Validating Parameters...")
    cfg = load_config("config.yaml")
    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]
    print(f"  Region: {cfg['metadata']['region_name']}")
    print(f"  CRS Projected: {cfg['metadata']['crs_projected']} | Geographic: {cfg['metadata']['crs_geographic']}")
    print(f"  Grid: {cfg['grid']['nx']}x{cfg['grid']['ny']} cells at {cfg['grid']['dx']}m spacing")
    print(f"  Point A: {pt_A['description']} ({pt_A['lon']:.4f}, {pt_A['lat']:.4f})")
    print(f"  Point B: {pt_B['description']} ({pt_B['lon']:.4f}, {pt_B['lat']:.4f})")

    # Distance validation
    dist_val = validate_distance(pt_A, pt_B, crs_projected=cfg["metadata"]["crs_projected"])
    print(f"  Distance: Grid={dist_val['distance_grid_m']:.2f} m | Geodesic={dist_val['distance_geodesic_m']:.2f} m | Diff={dist_val['difference_m']:.2f} m ({dist_val['difference_percent']:.4f}%)")

    # 2. Ingest and derive features
    print("\n[Step 2/6] Data Ingestion and Feature Derivation...")
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)
    print(f"  Depth range: [{feat['d'].min():.1f}, {feat['d'].max():.1f}] m")
    print(f"  Slope range: [{feat['s'].min():.4f}, {feat['s'].max():.4f}] m/m")
    print(f"  Roughness range: [{feat['r'].min():.4f}, {feat['r'].max():.4f}] m/m")
    print(f"  Curvature range: [{feat['c'].min():.6f}, {feat['c'].max():.6f}] m^-1")
    print(f"  Risk field R range: [{feat['R'].min():.4f}, {feat['R'].max():.4f}]")

    # 3. Deep Data Analysis
    print("\n[Step 3/6] Running Deep Data Analysis (Section 8)...")
    analysis_res = run_full_data_analysis(data, feat, cfg)
    print("  Descriptive statistics, correlation heatmaps, Moran's I, PCA, and Figures 1-4 generated.")

    # 4. Solve Easy and Hard Problems
    print("\n[Step 4/6] Solving Baseline Easy (Dijkstra) and Hard (SLSQP Spline) Problems...")
    easy_res, f_mask = solve_easy_problem(data, feat, cfg)
    print(f"  Easy Route solved in {easy_res['runtime_ms']:.2f} ms: L_E = {easy_res['L_E']:.2f} m, S_E = {easy_res['S_E']:.4f}, Max Curvature = {easy_res['max_kappa_fit']:.6f}")

    hard_res = solve_hard_problem(data, feat, cfg, easy_res=easy_res)
    print(f"  Hard Route solved: L_H = {hard_res['L_H']:.2f} m, S_H = {hard_res['S_H']:.4f}, Max Curvature = {hard_res['kappa_max']:.6f}, Feasible = {hard_res['is_feasible']}")

    comp, df_comp = compare_easy_and_hard(easy_res, hard_res, cfg)
    print("\n--- Baseline Comparison ---")
    print(df_comp.to_string(index=False))
    print(f"\n  Price of Realism: {comp['metrics']['price_of_realism_percent']:+.2f}%")
    print(f"  Risk Reduction:   {comp['metrics']['risk_reduction_percent']:+.2f}%")
    print(f"  Hausdorff Dist:   {comp['displacement']['hausdorff_distance_m']:.2f} m")

    # 5. Full Verification Audit
    print("\n[Step 5/6] Executing Section 11 Verification Suite...")
    verif = run_full_verification(data, feat, cfg, easy_res, hard_res)
    print(f"  All 10 Verification Checks Passed: {verif['all_checks_passed']}")

    # 6. Sensitivity Analysis
    print("\n[Step 6/6] Executing S-core, Full Matrix, and Pareto Sensitivity Suite (Section 9)...")
    df_sens, all_routes = run_all_sensitivities(data, feat, cfg, easy_res, hard_res)
    plot_sensitivity_results(df_sens, all_routes, data, feat, cfg)
    print("  Sensitivity analysis completed and Figures 5-9 generated.")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 75)
    print(f" MASTER BUILD COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 75)


if __name__ == "__main__":
    main()
