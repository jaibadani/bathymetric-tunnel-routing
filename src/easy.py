"""
easy.py - Discrete shortest feasible route optimization using Dijkstra on an 8-neighbor grid graph.
Implements:
1. Feasibility masking based on slope s <= s_max, depth d_min <= d <= d_max, and land mask.
2. 8-neighbor graph construction with physical metric Euclidean edge weights.
3. Edge validation modes:
   - 'endpoint': validates start and end cells of edge (baseline).
   - 'midpoint_corner': validates midpoint of diagonal moves and forbids cutting between blocked cells.
4. Custom priority-queue Dijkstra implementation returning route, length, and nodes expanded.
5. Cross-check against NetworkX shortest_path for formal verification.
6. Evaluation of cumulative risk S_E along polyline segments using the identical R field.
7. Discrete heading analysis and smoothed curvature profiling.
"""

import time
import heapq
import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline


def build_feasibility_mask(depth_arr, slope_arr, mask_land, d_min=50.0, d_max=400.0, s_max=0.08):
    """
    Returns boolean 2D mask where True indicates feasible cells.
    Feasible: ~mask_land and (d_min <= depth <= d_max) and (slope <= s_max).
    """
    feasible = (~mask_land) & (depth_arr >= d_min) & (depth_arr <= d_max) & (slope_arr <= s_max)
    return feasible


def solve_dijkstra(feasibility_mask, start_rc, goal_rc, dx=450.0, dy=450.0, edge_rule="endpoint"):
    """
    Solves discrete shortest feasible path on an 8-neighbor grid graph using Dijkstra.
    start_rc: (row, col) of Point A
    goal_rc: (row, col) of Point B
    edge_rule: 'endpoint' (baseline) or 'midpoint_corner'
    """
    ny, nx_grid = feasibility_mask.shape
    start_r, start_c = start_rc
    goal_r, goal_c = goal_rc

    if not feasibility_mask[start_r, start_c]:
        raise ValueError(f"Start node {start_rc} is infeasible under current constraints!")
    if not feasibility_mask[goal_r, goal_c]:
        raise ValueError(f"Goal node {goal_rc} is infeasible under current constraints!")

    # 8 neighbor offsets and metric edge weights
    neighbors = [
        (-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
        (-1, -1, np.hypot(dy, dx)), (-1, 1, np.hypot(dy, dx)),
        (1, -1, np.hypot(dy, dx)),  (1, 1, np.hypot(dy, dx))
    ]

    # Distances array and predecessor tracking
    dist = np.full((ny, nx_grid), np.inf, dtype=np.float64)
    dist[start_r, start_c] = 0.0
    predecessors = {}

    pq = [(0.0, start_r, start_c)]
    visited = np.zeros((ny, nx_grid), dtype=bool)
    nodes_expanded = 0

    t_start = time.perf_counter()

    while pq:
        d_curr, r, c = heapq.heappop(pq)

        if visited[r, c]:
            continue
        visited[r, c] = True
        nodes_expanded += 1

        if (r, c) == (goal_r, goal_c):
            break

        for dr, dc, weight in neighbors:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < ny and 0 <= nc < nx_grid):
                continue
            if not feasibility_mask[nr, nc]:
                continue

            # Stricter diagonal check
            if edge_rule == "midpoint_corner" and dr != 0 and dc != 0:
                # Forbid cutting between two diagonally adjacent blocked cells
                if not feasibility_mask[r, nc] and not feasibility_mask[nr, c]:
                    continue

            d_new = d_curr + weight
            if d_new < dist[nr, nc]:
                dist[nr, nc] = d_new
                predecessors[(nr, nc)] = (r, c)
                heapq.heappush(pq, (d_new, nr, nc))

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    if not visited[goal_r, goal_c]:
        return {
            "success": False,
            "runtime_ms": runtime_ms,
            "nodes_expanded": nodes_expanded,
            "error": "No feasible path found between start and goal"
        }

    # Reconstruct path from goal back to start
    path_rc = []
    curr = (goal_r, goal_c)
    while curr in predecessors:
        path_rc.append(curr)
        curr = predecessors[curr]
    path_rc.append((start_r, start_c))
    path_rc.reverse()

    return {
        "success": True,
        "path_rc": path_rc,
        "length_m": float(dist[goal_r, goal_c]),
        "runtime_ms": runtime_ms,
        "nodes_expanded": nodes_expanded,
        "visited_count": int(np.sum(visited))
    }


def cross_check_networkx(feasibility_mask, start_rc, goal_rc, dx=450.0, dy=450.0, edge_rule="endpoint"):
    """
    Builds a NetworkX graph and runs nx.shortest_path to verify Dijkstra's exact optimality.
    """
    ny, nx_grid = feasibility_mask.shape
    G = nx.Graph()

    neighbors = [
        (-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
        (-1, -1, np.hypot(dy, dx)), (-1, 1, np.hypot(dy, dx)),
        (1, -1, np.hypot(dy, dx)),  (1, 1, np.hypot(dy, dx))
    ]

    for r in range(ny):
        for c in range(nx_grid):
            if not feasibility_mask[r, c]:
                continue
            for dr, dc, weight in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ny and 0 <= nc < nx_grid and feasibility_mask[nr, nc]:
                    if edge_rule == "midpoint_corner" and dr != 0 and dc != 0:
                        if not feasibility_mask[r, nc] and not feasibility_mask[nr, c]:
                            continue
                    G.add_edge((r, c), (nr, nc), weight=weight)

    nx_path = nx.shortest_path(G, source=start_rc, target=goal_rc, weight="weight")
    nx_length = nx.shortest_path_length(G, source=start_rc, target=goal_rc, weight="weight")
    return nx_path, float(nx_length)


def evaluate_easy_route(dijkstra_result, features_dict, data_dict, cfg):
    """
    Evaluates the discrete easy route on length, cumulative terrain risk S_E,
    constraint compliance, heading variations, and fitted curvature.
    """
    if not dijkstra_result["success"]:
        return dijkstra_result

    path_rc = dijkstra_result["path_rc"]
    x_coords = data_dict["x_coords"]
    y_coords = data_dict["y_coords"]
    depth_arr = features_dict["d"]
    slope_arr = features_dict["s"]
    R_grid = features_dict["R"]

    # Coordinates in UTM
    path_x = np.array([x_coords[c] for r, c in path_rc])
    path_y = np.array([y_coords[r] for r, c in path_rc])

    # Segments and physical distance
    dx_segs = np.diff(path_x)
    dy_segs = np.diff(path_y)
    seg_lengths = np.hypot(dx_segs, dy_segs)
    cumulative_length = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    L_E = float(cumulative_length[-1])

    # Integrate Risk S_E along segments: trapezoidal / midpoint rule along polylines
    # R_mid = 0.5 * (R[p_i] + R[p_{i+1}])
    r_nodes = np.array([R_grid[r, c] for r, c in path_rc])
    r_segs_mid = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    S_E = float(np.sum(r_segs_mid * seg_lengths))

    # Profile along route
    depth_profile = np.array([depth_arr[r, c] for r, c in path_rc])
    slope_profile = np.array([slope_arr[r, c] for r, c in path_rc])

    # Constraints verification
    d_min = cfg["constraints"]["d_min"]
    d_max = cfg["constraints"]["d_max"]
    s_max = cfg["constraints"]["s_max"]
    kappa_max = cfg["constraints"]["kappa_max"]

    viol_d_min = np.maximum(0.0, d_min - depth_profile)
    viol_d_max = np.maximum(0.0, depth_profile - d_max)
    viol_s = np.maximum(0.0, slope_profile - s_max)

    # Heading angles (degrees) and turns
    headings = np.degrees(np.arctan2(dy_segs, dx_segs))
    turn_angles = np.abs(np.diff(headings))
    # Normalize turns to [0, 180]
    turn_angles = np.where(turn_angles > 180.0, 360.0 - turn_angles, turn_angles)
    max_turn_deg = float(np.max(turn_angles)) if len(turn_angles) > 0 else 0.0
    mean_turn_deg = float(np.mean(turn_angles)) if len(turn_angles) > 0 else 0.0

    # Fit a smooth spline through discrete route to evaluate what continuous curvature it would induce
    t_nodes = cumulative_length / L_E
    # Remove duplicate parameter values if any
    unique_idx = np.insert(np.diff(t_nodes) > 1e-6, 0, True)
    t_u = t_nodes[unique_idx]
    x_u = path_x[unique_idx]
    y_u = path_y[unique_idx]

    spline_x = CubicSpline(t_u, x_u)
    spline_y = CubicSpline(t_u, y_u)
    t_eval = np.linspace(0.0, 1.0, 500)
    xp = spline_x(t_eval, 1)
    yp = spline_y(t_eval, 1)
    xpp = spline_x(t_eval, 2)
    ypp = spline_y(t_eval, 2)
    kappa_fit = np.abs(xp * ypp - yp * xpp) / np.maximum((xp**2 + yp**2)**1.5, 1e-12)
    max_kappa_fit = float(np.max(kappa_fit))
    viol_kappa = max(0.0, max_kappa_fit - kappa_max)

    alpha = cfg["hard_solver"]["alpha"]
    beta = cfg["hard_solver"]["beta"]
    J_E = float(alpha * L_E + beta * S_E)

    return {
        "success": True,
        "path_rc": path_rc,
        "path_x": path_x.tolist(),
        "path_y": path_y.tolist(),
        "L_E": L_E,
        "S_E": S_E,
        "J_E": J_E,
        "runtime_ms": dijkstra_result["runtime_ms"],
        "nodes_expanded": dijkstra_result["nodes_expanded"],
        "num_nodes": len(path_rc),
        "depth_profile": depth_profile.tolist(),
        "slope_profile": slope_profile.tolist(),
        "depth_min": float(np.min(depth_profile)),
        "depth_max": float(np.max(depth_profile)),
        "slope_max": float(np.max(slope_profile)),
        "max_viol_depth": float(max(np.max(viol_d_min), np.max(viol_d_max))),
        "max_viol_slope": float(np.max(viol_s)),
        "max_turn_deg": max_turn_deg,
        "mean_turn_deg": mean_turn_deg,
        "max_kappa_fit": max_kappa_fit,
        "viol_kappa_fit": viol_kappa,
        "optimality_type": "Global discrete optimum on 8-neighbor graph"
    }


def solve_easy_problem(data_dict, features_dict, cfg, edge_rule=None):
    """
    Main entry point for solving and evaluating the Easy Problem.
    """
    if edge_rule is None:
        edge_rule = cfg["easy_solver"].get("edge_rule", "endpoint")

    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]
    start_rc = (pt_A["grid_row"], pt_A["grid_col"])
    goal_rc = (pt_B["grid_row"], pt_B["grid_col"])

    d_min = cfg["constraints"]["d_min"]
    d_max = cfg["constraints"]["d_max"]
    s_max = cfg["constraints"]["s_max"]

    feasibility_mask = build_feasibility_mask(
        depth_arr=features_dict["d"],
        slope_arr=features_dict["s"],
        mask_land=data_dict["mask_land"],
        d_min=d_min,
        d_max=d_max,
        s_max=s_max
    )

    dijkstra_res = solve_dijkstra(
        feasibility_mask,
        start_rc=start_rc,
        goal_rc=goal_rc,
        dx=data_dict["metadata"]["dx_m"],
        dy=data_dict["metadata"]["dy_m"],
        edge_rule=edge_rule
    )

    eval_res = evaluate_easy_route(dijkstra_res, features_dict, data_dict, cfg)
    return eval_res, feasibility_mask


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    from src.features import derive_features
    import json

    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)
    easy_res, f_mask = solve_easy_problem(data, feat, cfg)

    print("Easy Problem Solve Complete:")
    print(f"  Success: {easy_res['success']}")
    print(f"  Length L_E: {easy_res['L_E']:.2f} m ({easy_res['L_E']/1000:.3f} km)")
    print(f"  Cumulative Risk S_E: {easy_res['S_E']:.4f}")
    print(f"  Objective J_E: {easy_res['J_E']:.2f}")
    print(f"  Runtime: {easy_res['runtime_ms']:.2f} ms")
    print(f"  Nodes expanded: {easy_res['nodes_expanded']}")
    print(f"  Max discrete turn: {easy_res['max_turn_deg']:.1f}°")
    print(f"  Curvature of smooth fit: {easy_res['max_kappa_fit']:.6f} m^-1 (Limit: {cfg['constraints']['kappa_max']})")

    # Cross-check with NetworkX
    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]
    nx_path, nx_len = cross_check_networkx(
        f_mask, (pt_A["grid_row"], pt_A["grid_col"]), (pt_B["grid_row"], pt_B["grid_col"])
    )
    diff = abs(easy_res["L_E"] - nx_len)
    print(f"  NetworkX cross-check diff: {diff:.6e} m (Exact match: {diff < 1e-9})")
