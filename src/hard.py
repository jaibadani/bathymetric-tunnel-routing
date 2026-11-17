"""
hard.py - Continuous smooth route optimization using Cubic Splines and SciPy SLSQP.
Implements:
1. Bicubic smooth interpolation (RectBivariateSpline) for depth d, slope s, and risk R.
2. Interpolator validation and overshoot quantification.
3. Natural cubic spline route parameterization p(t) = (x(t), y(t)) for t in [0, 1].
4. Decision variables: K interior control points (baseline K=7, alternatives 5, 10).
5. Numerical integration of length L(p) and cumulative terrain risk S(p) using M=200 samples.
6. Non-linear inequality constraints:
   - d_min <= d(p(t)) <= d_max
   - s(p(t)) <= s_max
   - |kappa_route(t)| <= kappa_max
7. Multi-start solver execution:
   - Start 1: Warm start fitted through Dijkstra route.
   - Start 2: Straight line corridor between A and B.
   - Starts 3-10: 8 randomized sinusoidal/normal perturbations with fixed seed.
8. Rigorous post-solve domain validity verification.
9. Formal acknowledgement of local optimality on non-convex continuous landscape.
"""

import time
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize


def build_interpolators(data_dict, features_dict):
    """
    Builds smooth RectBivariateSpline bicubic interpolators for depth, slope, and risk.
    Note: y_coords must be strictly increasing for RectBivariateSpline, so we flip rows.
    """
    x_coords = data_dict["x_coords"]
    y_coords_inc = data_dict["y_coords"][::-1] # ascending

    d_grid = features_dict["d"][::-1, :]
    s_grid = features_dict["s"][::-1, :]
    R_grid = features_dict["R"][::-1, :]

    d_interp = RectBivariateSpline(y_coords_inc, x_coords, d_grid, kx=3, ky=3)
    s_interp = RectBivariateSpline(y_coords_inc, x_coords, s_grid, kx=3, ky=3)
    R_interp = RectBivariateSpline(y_coords_inc, x_coords, R_grid, kx=3, ky=3)

    return {
        "depth": d_interp,
        "slope": s_interp,
        "risk": R_interp,
        "x_coords": x_coords,
        "y_coords_inc": y_coords_inc
    }


def verify_interpolator_overshoot(interpolators, features_dict):
    """
    Validates interpolator accuracy against the source discrete grids.
    Reports max node deviation and overshoot relative to source data bounds.
    """
    x_coords = interpolators["x_coords"]
    y_coords_inc = interpolators["y_coords_inc"]

    # Midpoint coordinates for checking between-node behavior
    x_mid = 0.5 * (x_coords[:-1] + x_coords[1:])
    y_mid = 0.5 * (y_coords_inc[:-1] + y_coords_inc[1:])

    overshoot_report = {}
    for name, key in [("depth", "d"), ("slope", "s"), ("risk", "R")]:
        interp = interpolators[name]
        source_grid = features_dict[key][::-1, :]

        # Node accuracy
        node_vals = interp(y_coords_inc, x_coords, grid=True)
        max_node_err = float(np.max(np.abs(node_vals - source_grid)))

        # Midpoint overshoot check
        mid_vals = interp(y_mid, x_mid, grid=True)
        src_min = float(np.min(source_grid))
        src_max = float(np.max(source_grid))
        mid_min = float(np.min(mid_vals))
        mid_max = float(np.max(mid_vals))

        overshoot_lower = float(max(0.0, src_min - mid_min))
        overshoot_upper = float(max(0.0, mid_max - src_max))

        overshoot_report[name] = {
            "max_node_deviation": max_node_err,
            "source_range": [src_min, src_max],
            "interpolated_mid_range": [mid_min, mid_max],
            "overshoot_below_min": overshoot_lower,
            "overshoot_above_max": overshoot_upper,
            "acceptable": overshoot_lower < 0.05 * (src_max - src_min) and overshoot_upper < 0.05 * (src_max - src_min)
        }

    return overshoot_report


def generate_multi_starts(easy_res, pt_A, pt_B, K=7, num_random=8, seed=42):
    """
    Generates candidate starting control points vectors:
    1. Warm start: fitted through Dijkstra route
    2. Straight line: uniformly spaced between A and B
    3. 8 randomized sinusoidal/normal perturbations with fixed seed
    """
    xA, yA = pt_A["utm_x"], pt_A["utm_y"]
    xB, yB = pt_B["utm_x"], pt_B["utm_y"]
    t_knots = np.linspace(0.0, 1.0, K + 2)
    t_interior = t_knots[1:-1]

    dx_AB = xB - xA
    dy_AB = yB - yA
    dist_AB = np.hypot(dx_AB, dy_AB)
    nx_ortho = -dy_AB / dist_AB
    ny_ortho = dx_AB / dist_AB

    starts = []

    # 1. Warm start from Dijkstra
    if easy_res is not None and easy_res.get("success", False):
        dijk_x = np.array(easy_res["path_x"])
        dijk_y = np.array(easy_res["path_y"])
        dijk_cum = np.insert(np.cumsum(np.hypot(np.diff(dijk_x), np.diff(dijk_y))), 0, 0.0)
        dijk_t = dijk_cum / dijk_cum[-1]

        warm_x = np.interp(t_interior, dijk_t, dijk_x)
        warm_y = np.interp(t_interior, dijk_t, dijk_y)
        p_warm = np.concatenate([warm_x, warm_y])
        starts.append({"name": "warm_start_dijkstra", "p0": p_warm})

    # 2. Straight line start
    straight_x = xA + t_interior * dx_AB
    straight_y = yA + t_interior * dy_AB
    p_straight = np.concatenate([straight_x, straight_y])
    starts.append({"name": "straight_line", "p0": p_straight})

    # 3. Randomized perturbed starts
    rng = np.random.default_rng(seed)
    for i in range(num_random):
        mode = int(rng.choice([1, 2, 3]))
        amp = float(rng.uniform(-0.25, 0.25) * dist_AB)
        sine_pert = amp * np.sin(mode * np.pi * t_interior)
        noise = rng.normal(0.0, 0.02 * dist_AB, size=K)
        pert = sine_pert + noise

        x_pert = straight_x + pert * nx_ortho
        y_pert = straight_y + pert * ny_ortho
        p_pert = np.concatenate([x_pert, y_pert])
        starts.append({"name": f"random_pert_{i+1}", "p0": p_pert})

    return starts


class ContinuousTunnelOptimizer:
    """
    Continuous spline route optimizer under depth, slope, and route curvature limits.
    """
    def __init__(self, interpolators, pt_A, pt_B, cfg, K=7, M=200, alpha=1.0, beta=50000.0):
        self.interp_d = interpolators["depth"]
        self.interp_s = interpolators["slope"]
        self.interp_R = interpolators["risk"]

        self.xA = pt_A["utm_x"]
        self.yA = pt_A["utm_y"]
        self.xB = pt_B["utm_x"]
        self.yB = pt_B["utm_y"]

        self.K = K
        self.M = M
        self.alpha = alpha
        self.beta = beta

        self.d_min = cfg["constraints"]["d_min"]
        self.d_max = cfg["constraints"]["d_max"]
        self.s_max = cfg["constraints"]["s_max"]
        self.kappa_max = cfg["constraints"]["kappa_max"]
        # Use a 3% internal numerical buffer during optimization so dense validation strictly holds
        self.opt_kappa_max = self.kappa_max * 0.97

        self.t_knots = np.linspace(0.0, 1.0, self.K + 2)
        self.t_samples = np.linspace(0.0, 1.0, self.M)

        # Coordinate box bounds
        grid_cfg = cfg["grid"]
        self.xmin = grid_cfg["xmin"]
        self.xmax = grid_cfg["xmax"]
        self.ymin = grid_cfg["ymin"]
        self.ymax = grid_cfg["ymax"]

    def build_splines(self, p):
        x_k = np.concatenate([[self.xA], p[:self.K], [self.xB]])
        y_k = np.concatenate([[self.yA], p[self.K:], [self.yB]])
        cs_x = CubicSpline(self.t_knots, x_k, bc_type="natural")
        cs_y = CubicSpline(self.t_knots, y_k, bc_type="natural")
        return cs_x, cs_y

    def objective(self, p):
        cs_x, cs_y = self.build_splines(p)
        xs = cs_x(self.t_samples)
        ys = cs_y(self.t_samples)
        dx = cs_x(self.t_samples, 1)
        dy = cs_y(self.t_samples, 1)
        v = np.hypot(dx, dy)

        # Evaluate risk along path
        R_vals = self.interp_R(ys, xs, grid=False)
        # Trapezoidal numerical integration
        L = np.trapezoid(v, self.t_samples)
        S = np.trapezoid(R_vals * v, self.t_samples)

        J = self.alpha * L + self.beta * S
        return J

    def c_depth_min(self, p):
        cs_x, cs_y = self.build_splines(p)
        return self.interp_d(cs_y(self.t_samples), cs_x(self.t_samples), grid=False) - self.d_min

    def c_depth_max(self, p):
        cs_x, cs_y = self.build_splines(p)
        return self.d_max - self.interp_d(cs_y(self.t_samples), cs_x(self.t_samples), grid=False)

    def c_slope_max(self, p):
        cs_x, cs_y = self.build_splines(p)
        return self.s_max - self.interp_s(cs_y(self.t_samples), cs_x(self.t_samples), grid=False)

    def c_kappa_max(self, p):
        cs_x, cs_y = self.build_splines(p)
        dx = cs_x(self.t_samples, 1)
        dy = cs_y(self.t_samples, 1)
        ddx = cs_x(self.t_samples, 2)
        ddy = cs_y(self.t_samples, 2)
        v2 = dx**2 + dy**2
        kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(v2**1.5, 1e-12)
        return self.opt_kappa_max - kappa

    def get_constraints(self):
        return [
            {"type": "ineq", "fun": self.c_depth_min},
            {"type": "ineq", "fun": self.c_depth_max},
            {"type": "ineq", "fun": self.c_slope_max},
            {"type": "ineq", "fun": self.c_kappa_max}
        ]

    def get_bounds(self):
        # Bounds: interior control points remain within the grid box
        bounds_x = [(self.xmin, self.xmax)] * self.K
        bounds_y = [(self.ymin, self.ymax)] * self.K
        return bounds_x + bounds_y

    def evaluate_route_dense(self, p, num_eval_samples=2000):
        """
        Dense independent evaluation of the continuous route at num_eval_samples points.
        """
        cs_x, cs_y = self.build_splines(p)
        t_dense = np.linspace(0.0, 1.0, num_eval_samples)

        xs = cs_x(t_dense)
        ys = cs_y(t_dense)
        dx = cs_x(t_dense, 1)
        dy = cs_y(t_dense, 1)
        ddx = cs_x(t_dense, 2)
        ddy = cs_y(t_dense, 2)

        v = np.hypot(dx, dy)
        v2 = dx**2 + dy**2
        kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(v2**1.5, 1e-12)

        d_vals = self.interp_d(ys, xs, grid=False)
        s_vals = self.interp_s(ys, xs, grid=False)
        R_vals = self.interp_R(ys, xs, grid=False)

        L = float(np.trapezoid(v, t_dense))
        S = float(np.trapezoid(R_vals * v, t_dense))
        J = float(self.alpha * L + self.beta * S)

        viol_d_min = np.maximum(0.0, self.d_min - d_vals)
        viol_d_max = np.maximum(0.0, d_vals - self.d_max)
        viol_s = np.maximum(0.0, s_vals - self.s_max)
        viol_kappa = np.maximum(0.0, kappa - self.kappa_max)

        max_viol_d = float(max(np.max(viol_d_min), np.max(viol_d_max)))
        max_viol_s = float(np.max(viol_s))
        max_viol_kappa = float(np.max(viol_kappa))

        # Domain boundary check
        in_bounds_x = (np.min(xs) >= self.xmin) and (np.max(xs) <= self.xmax)
        in_bounds_y = (np.min(ys) >= self.ymin) and (np.max(ys) <= self.ymax)
        domain_valid = bool(in_bounds_x and in_bounds_y)

        # Check endpoints exactly match A and B
        endpoint_match = (
            abs(xs[0] - self.xA) < 1e-4 and abs(ys[0] - self.yA) < 1e-4 and
            abs(xs[-1] - self.xB) < 1e-4 and abs(ys[-1] - self.yB) < 1e-4
        )

        return {
            "L_H": L,
            "S_H": S,
            "J_H": J,
            "path_x": xs.tolist(),
            "path_y": ys.tolist(),
            "t_samples": t_dense.tolist(),
            "depth_profile": d_vals.tolist(),
            "slope_profile": s_vals.tolist(),
            "kappa_profile": kappa.tolist(),
            "risk_profile": R_vals.tolist(),
            "depth_min": float(np.min(d_vals)),
            "depth_max": float(np.max(d_vals)),
            "slope_max": float(np.max(s_vals)),
            "kappa_max_eval": float(np.max(kappa)),
            "max_viol_depth": max_viol_d,
            "max_viol_slope": max_viol_s,
            "max_viol_kappa": max_viol_kappa,
            "is_feasible": (max_viol_d < 1e-3 and max_viol_s < 1e-4 and max_viol_kappa < 1e-6 and domain_valid),
            "domain_valid": domain_valid,
            "endpoint_match": endpoint_match
        }


def solve_hard_problem(data_dict, features_dict, cfg, easy_res=None, K=None, M=None, alpha=None, beta=None, maxiter=300):
    """
    Main entry point for solving the Hard Continuous Spline problem via multi-start SLSQP.
    """
    if K is None:
        K = cfg["hard_solver"].get("num_control_points", 7)
    if M is None:
        M = cfg["hard_solver"].get("num_samples", 200)
    if alpha is None:
        alpha = cfg["hard_solver"].get("alpha", 1.0)
    if beta is None:
        beta = cfg["hard_solver"].get("beta", 50000.0)

    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]

    # 1. Build and verify interpolators
    interpolators = build_interpolators(data_dict, features_dict)
    overshoot_report = verify_interpolator_overshoot(interpolators, features_dict)

    # 2. Instantiate optimizer
    opt = ContinuousTunnelOptimizer(
        interpolators=interpolators,
        pt_A=pt_A,
        pt_B=pt_B,
        cfg=cfg,
        K=K,
        M=M,
        alpha=alpha,
        beta=beta
    )

    # 3. Generate multi-starts
    num_random = cfg["hard_solver"].get("num_starts", 10) - 2
    if num_random < 8:
        num_random = 8
    seed = cfg["hard_solver"].get("seed", 42)
    starts = generate_multi_starts(easy_res, pt_A, pt_B, K=K, num_random=num_random, seed=seed)

    constraints = opt.get_constraints()
    bounds = opt.get_bounds()
    ftol = float(cfg["hard_solver"].get("ftol", 1.0e-5))

    start_runs = []
    best_res = None
    best_J = np.inf

    for st in starts:
        name = st["name"]
        p0 = st["p0"]

        t_start = time.perf_counter()
        res = minimize(
            opt.objective,
            p0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False}
        )
        runtime_ms = (time.perf_counter() - t_start) * 1000.0

        # Dense post-evaluation
        eval_dense = opt.evaluate_route_dense(res.x, num_eval_samples=2000)

        run_info = {
            "name": name,
            "success": bool(res.success),
            "status_message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "J": float(res.fun),
            "runtime_ms": runtime_ms,
            "L_H": eval_dense["L_H"],
            "S_H": eval_dense["S_H"],
            "max_viol_depth": eval_dense["max_viol_depth"],
            "max_viol_slope": eval_dense["max_viol_slope"],
            "max_viol_kappa": eval_dense["max_viol_kappa"],
            "is_feasible": eval_dense["is_feasible"],
            "control_points": res.x.tolist()
        }
        start_runs.append(run_info)

        # Track best feasible solution (or lowest J if none strictly feasible)
        if eval_dense["is_feasible"] and res.fun < best_J:
            best_J = res.fun
            best_res = (res, eval_dense, run_info)
        elif best_res is None or (not best_res[1]["is_feasible"] and res.fun < best_J):
            best_J = res.fun
            best_res = (res, eval_dense, run_info)

    # Multi-start statistics
    converged_runs = [r for r in start_runs if r["success"]]
    feasible_runs = [r for r in start_runs if r["is_feasible"]]
    convergence_rate = len(converged_runs) / len(start_runs)
    feasible_rate = len(feasible_runs) / len(start_runs)
    j_values = [r["J"] for r in start_runs]
    j_spread = float(np.max(j_values) - np.min(j_values))
    j_std = float(np.std(j_values))

    res_opt, eval_dense, best_info = best_res

    return {
        "success": best_info["success"],
        "is_feasible": eval_dense["is_feasible"],
        "best_start_name": best_info["name"],
        "L_H": eval_dense["L_H"],
        "S_H": eval_dense["S_H"],
        "J_H": eval_dense["J_H"],
        "runtime_ms": best_info["runtime_ms"],
        "nit": best_info["nit"],
        "nfev": best_info["nfev"],
        "status_message": best_info["status_message"],
        "control_points_x": res_opt.x[:K].tolist(),
        "control_points_y": res_opt.x[K:].tolist(),
        "path_x": eval_dense["path_x"],
        "path_y": eval_dense["path_y"],
        "t_samples": eval_dense["t_samples"],
        "depth_profile": eval_dense["depth_profile"],
        "slope_profile": eval_dense["slope_profile"],
        "kappa_profile": eval_dense["kappa_profile"],
        "risk_profile": eval_dense["risk_profile"],
        "depth_min": eval_dense["depth_min"],
        "depth_max": eval_dense["depth_max"],
        "slope_max": eval_dense["slope_max"],
        "kappa_max": eval_dense["kappa_max_eval"],
        "max_viol_depth": eval_dense["max_viol_depth"],
        "max_viol_slope": eval_dense["max_viol_slope"],
        "max_viol_kappa": eval_dense["max_viol_kappa"],
        "domain_valid": eval_dense["domain_valid"],
        "endpoint_match": eval_dense["endpoint_match"],
        "multi_start_stats": {
            "total_starts": len(start_runs),
            "converged_starts": len(converged_runs),
            "feasible_starts": len(feasible_runs),
            "convergence_rate": convergence_rate,
            "feasible_rate": feasible_rate,
            "j_spread": j_spread,
            "j_std": j_std,
            "start_runs": start_runs
        },
        "overshoot_report": overshoot_report,
        "optimality_type": "Local continuous optimum on non-convex objective/constraint manifold (SLSQP)"
    }


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    from src.features import derive_features
    from src.easy import solve_easy_problem
    import json

    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)
    easy_res, _ = solve_easy_problem(data, feat, cfg)

    print("Running Multi-start SLSQP for Hard Problem...")
    hard_res = solve_hard_problem(data, feat, cfg, easy_res=easy_res)

    print("\nHard Problem Solve Complete:")
    print(f"  Best Start: {hard_res['best_start_name']}")
    print(f"  Success: {hard_res['success']} (Feasible: {hard_res['is_feasible']})")
    print(f"  Length L_H: {hard_res['L_H']:.2f} m ({hard_res['L_H']/1000:.3f} km)")
    print(f"  Cumulative Risk S_H: {hard_res['S_H']:.4f}")
    print(f"  Objective J_H: {hard_res['J_H']:.2f}")
    print(f"  Max Curvature: {hard_res['kappa_max']:.6f} m^-1 (Limit: {cfg['constraints']['kappa_max']})")
    print(f"  Max Slope: {hard_res['slope_max']:.4f} (Limit: {cfg['constraints']['s_max']})")
    print(f"  Depth Range: [{hard_res['depth_min']:.1f}, {hard_res['depth_max']:.1f}] m")
    print(f"  Convergence Rate across Starts: {hard_res['multi_start_stats']['convergence_rate']*100:.1f}%")
    print(f"  Spread of J across starts: {hard_res['multi_start_stats']['j_spread']:.2f}")
    print(f"  Optimality Type: {hard_res['optimality_type']}")
