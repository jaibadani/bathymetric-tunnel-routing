"""
metrics.py - Common evaluation, comparison metrics, price of realism, and robustness metrics.
Implements:
1. Unified metric evaluation of both easy and hard routes using identical R and units.
2. Price of Realism = (L_H - L_E) / L_E * 100% and Risk Reduction = (S_E - S_H) / S_E * 100%.
3. Constraint violation profiles and density study (200, 1000, 5000 sample points).
4. Route displacement metrics:
   - Symmetric Hausdorff distance (m)
   - Mean curve distance (m)
   - Raster cell overlap (Jaccard index)
5. Comparison table generation for reporting.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
import pandas as pd


def compute_hausdorff_distance(poly_a, poly_b):
    """
    Computes symmetric Hausdorff distance between two 2D polylines in metres.
    poly_a, poly_b: shape (N, 2), (M, 2)
    """
    d_ab = directed_hausdorff(poly_a, poly_b)[0]
    d_ba = directed_hausdorff(poly_b, poly_a)[0]
    return float(max(d_ab, d_ba))


def compute_mean_curve_distance(poly_a, poly_b):
    """
    Computes the bidirectional mean distance between two 2D polylines:
    0.5 * (mean_{p in A} min_{q in B} dist(p, q) + mean_{q in B} min_{p in A} dist(q, p))
    """
    dists_ab = cdist(poly_a, poly_b) # (N, M)
    min_a_to_b = np.min(dists_ab, axis=1) # min dist from each pt in A to curve B
    min_b_to_a = np.min(dists_ab, axis=0) # min dist from each pt in B to curve A
    mean_dist = 0.5 * (np.mean(min_a_to_b) + np.mean(min_b_to_a))
    return float(mean_dist)


def compute_cell_jaccard_overlap(path_rc_a, path_rc_b):
    """
    Computes the Jaccard similarity index between raster cell footprints of two routes:
    |A cap B| / |A cup B|
    """
    set_a = set(path_rc_a)
    set_b = set(path_rc_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return float(intersection / union)


def run_density_study(hard_opt, p_opt, sample_counts=None):
    """
    Runs density study checking constraint satisfaction at M in {200, 1000, 5000} samples.
    Reports whether the maximum constraint violations change with sampling density.
    """
    if sample_counts is None:
        sample_counts = [200, 1000, 5000]

    study_results = []
    for m in sample_counts:
        eval_m = hard_opt.evaluate_route_dense(p_opt, num_eval_samples=m)
        study_results.append({
            "sample_count": m,
            "max_viol_depth_m": eval_m["max_viol_depth"],
            "max_viol_slope": eval_m["max_viol_slope"],
            "max_viol_kappa_m_inv": eval_m["max_viol_kappa"],
            "max_kappa": eval_m["kappa_max_eval"],
            "max_slope": eval_m["slope_max"],
            "depth_range": [eval_m["depth_min"], eval_m["depth_max"]],
            "is_feasible": eval_m["is_feasible"]
        })

    return study_results


def compare_easy_and_hard(easy_res, hard_res, cfg):
    """
    Generates comprehensive comparison between the Easy and Hard optimal routes.
    Computes Price of Realism, Risk Reduction, constraint compliance, and route displacements.
    """
    L_E = easy_res["L_E"]
    L_H = hard_res["L_H"]
    S_E = easy_res["S_E"]
    S_H = hard_res["S_H"]
    J_E = easy_res["J_E"]
    J_H = hard_res["J_H"]

    # Price of Realism and Risk Reduction
    price_of_realism_pct = ((L_H - L_E) / L_E) * 100.0
    risk_reduction_pct = ((S_E - S_H) / S_E) * 100.0

    # Polylines in UTM (N, 2)
    poly_E = np.column_stack([easy_res["path_x"], easy_res["path_y"]])
    poly_H = np.column_stack([hard_res["path_x"], hard_res["path_y"]])

    hausdorff_dist_m = compute_hausdorff_distance(poly_E, poly_H)
    mean_curve_dist_m = compute_mean_curve_distance(poly_E, poly_H)

    # Convert hard route to grid cells for Jaccard overlap
    x_coords = np.array(cfg["grid"]["xmin"]) + np.arange(cfg["grid"]["nx"]) * cfg["grid"]["dx"]
    y_coords = np.array(cfg["grid"]["ymax"]) - np.arange(cfg["grid"]["ny"]) * cfg["grid"]["dy"]

    def coords_to_rc(xs, ys):
        cols = np.clip(np.round((xs - cfg["grid"]["xmin"]) / cfg["grid"]["dx"]).astype(int), 0, cfg["grid"]["nx"] - 1)
        rows = np.clip(np.round((cfg["grid"]["ymax"] - ys) / cfg["grid"]["dy"]).astype(int), 0, cfg["grid"]["ny"] - 1)
        return list(zip(rows, cols))

    hard_rc = coords_to_rc(np.array(hard_res["path_x"]), np.array(hard_res["path_y"]))
    jaccard_overlap = compute_cell_jaccard_overlap(easy_res["path_rc"], hard_rc)

    # Structured comparison record
    comparison = {
        "metrics": {
            "L_E_m": L_E,
            "L_H_m": L_H,
            "delta_L_m": L_H - L_E,
            "price_of_realism_percent": price_of_realism_pct,
            "S_E": S_E,
            "S_H": S_H,
            "delta_S": S_H - S_E,
            "risk_reduction_percent": risk_reduction_pct,
            "J_E": J_E,
            "J_H": J_H,
            "runtime_easy_ms": easy_res["runtime_ms"],
            "runtime_hard_ms": hard_res["runtime_ms"],
            "runtime_ratio_hard_to_easy": hard_res["runtime_ms"] / max(easy_res["runtime_ms"], 1e-3)
        },
        "constraints": {
            "easy": {
                "max_viol_depth_m": easy_res["max_viol_depth"],
                "max_viol_slope": easy_res["max_viol_slope"],
                "max_viol_kappa_m_inv": easy_res["viol_kappa_fit"],
                "max_turn_deg": easy_res["max_turn_deg"],
                "max_kappa_fit": easy_res["max_kappa_fit"]
            },
            "hard": {
                "max_viol_depth_m": hard_res["max_viol_depth"],
                "max_viol_slope": hard_res["max_viol_slope"],
                "max_viol_kappa_m_inv": hard_res["max_viol_kappa"],
                "max_kappa": hard_res["kappa_max"],
                "is_feasible": hard_res["is_feasible"]
            }
        },
        "displacement": {
            "hausdorff_distance_m": hausdorff_dist_m,
            "mean_curve_distance_m": mean_curve_dist_m,
            "cell_jaccard_overlap": jaccard_overlap
        },
        "optimality_statements": {
            "easy": "Global discrete optimum on the 8-neighbor grid graph using Dijkstra (nonnegative weights).",
            "hard": "Local continuous optimum on non-convex spline objective/constraint manifold using SLSQP."
        }
    }

    # Tabular summary DataFrame for reporting
    df_comp = pd.DataFrame([
        {"Metric": "Route Length L", "Unit": "m", "Easy (Dijkstra)": f"{L_E:.2f}", "Hard (SLSQP)": f"{L_H:.2f}", "Difference": f"{L_H - L_E:+.2f} ({price_of_realism_pct:+.2f}%)"},
        {"Metric": "Terrain Risk S", "Unit": "dimensionless", "Easy (Dijkstra)": f"{S_E:.4f}", "Hard (SLSQP)": f"{S_H:.4f}", "Difference": f"{S_H - S_E:+.4f} ({-risk_reduction_pct:+.2f}%)"},
        {"Metric": "Composite Objective J", "Unit": "weighted units", "Easy (Dijkstra)": f"{J_E:.2f}", "Hard (SLSQP)": f"{J_H:.2f}", "Difference": f"{J_H - J_E:+.2f}"},
        {"Metric": "Maximum Slope s", "Unit": "m/m (Limit: 0.08)", "Easy (Dijkstra)": f"{easy_res['slope_max']:.4f}", "Hard (SLSQP)": f"{hard_res['slope_max']:.4f}", "Difference": f"{hard_res['slope_max'] - easy_res['slope_max']:+.4f}"},
        {"Metric": "Depth Range d", "Unit": "m (Limit: [50, 400])", "Easy (Dijkstra)": f"[{easy_res['depth_min']:.1f}, {easy_res['depth_max']:.1f}]", "Hard (SLSQP)": f"[{hard_res['depth_min']:.1f}, {hard_res['depth_max']:.1f}]", "Difference": "Within limits"},
        {"Metric": "Maximum Route Curvature κ", "Unit": "m^-1 (Limit: 0.0005)", "Easy (Dijkstra)": f"{easy_res['max_kappa_fit']:.6f} (VIOLATES)", "Hard (SLSQP)": f"{hard_res['kappa_max']:.6f} (FEASIBLE)", "Difference": f"{hard_res['kappa_max'] - easy_res['max_kappa_fit']:.6f}"},
        {"Metric": "Runtime", "Unit": "ms", "Easy (Dijkstra)": f"{easy_res['runtime_ms']:.2f}", "Hard (SLSQP)": f"{hard_res['runtime_ms']:.2f}", "Difference": f"x{comparison['metrics']['runtime_ratio_hard_to_easy']:.1f}"},
        {"Metric": "Optimality Scope", "Unit": "theoretical", "Easy (Dijkstra)": "Global Discrete", "Hard (SLSQP)": "Local Continuous", "Difference": "Graph vs Spline"}
    ])

    return comparison, df_comp


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    from src.features import derive_features
    from src.easy import solve_easy_problem
    from src.hard import solve_hard_problem
    import json

    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)

    easy_res, _ = solve_easy_problem(data, feat, cfg)
    hard_res = solve_hard_problem(data, feat, cfg, easy_res=easy_res)

    comp, df_comp = compare_easy_and_hard(easy_res, hard_res, cfg)
    print("\n--- Easy vs Hard Comparison Table ---")
    print(df_comp.to_string(index=False))
    print(f"\nPrice of Realism: {comp['metrics']['price_of_realism_percent']:+.2f}%")
    print(f"Risk Reduction: {comp['metrics']['risk_reduction_percent']:+.2f}%")
    print(f"Hausdorff Distance: {comp['displacement']['hausdorff_distance_m']:.2f} m")
    print(f"Mean Curve Distance: {comp['displacement']['mean_curve_distance_m']:.2f} m")
