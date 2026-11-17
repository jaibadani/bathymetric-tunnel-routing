"""
features.py - Terrain feature derivation, pre-smoothing, normalization, and risk field construction.
Implements:
- Slope: Central differences (baseline), Horn (3x3), Sobel.
- Roughness: Moving window standard deviation of slope (baseline) or depth, w in {3, 5, 7, 9}.
- Terrain Curvature: Laplacian 5-point stencil (baseline), Hessian Frobenius norm.
- Pre-smoothing: Gaussian filter with sigma in {0, 1, 2, 3}.
- Normalization: Min-Max or 99th percentile clipping + Min-Max.
- Terrain Risk Field R(x, y) = w_s * s_tilde^2 + w_r * r_tilde + w_c * c_tilde.
"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def compute_slope(z, dx=450.0, dy=450.0, method="central_difference"):
    """
    Computes terrain slope s = |grad z| in metres/metre.
    z has shape (ny, nx), where row index increases southward (y decreases).
    """
    ny, nx = z.shape
    if method == "central_difference":
        # dz/dx: horizontal (column) gradient
        dz_dx = np.zeros_like(z)
        dz_dx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * dx)
        dz_dx[:, 0] = (z[:, 1] - z[:, 0]) / dx
        dz_dx[:, -1] = (z[:, -1] - z[:, -2]) / dx

        # dz/dy: vertical (row) gradient (y decreases as row increases)
        dz_dy = np.zeros_like(z)
        dz_dy[1:-1, :] = (z[:-2, :] - z[2:, :]) / (2.0 * dy)
        dz_dy[0, :] = (z[0, :] - z[1, :]) / dy
        dz_dy[-1, :] = (z[-2, :] - z[-1, :]) / dy

    elif method in ("horn", "sobel"):
        # 3x3 weighted kernel (Horn / Sobel)
        # Horn weighting: [1, 2, 1]
        dz_dx = np.zeros_like(z)
        dz_dy = np.zeros_like(z)

        # Interior 3x3
        top_left = z[:-2, :-2]
        top_mid = z[:-2, 1:-1]
        top_right = z[:-2, 2:]

        mid_left = z[1:-1, :-2]
        mid_right = z[1:-1, 2:]

        bot_left = z[2:, :-2]
        bot_mid = z[2:, 1:-1]
        bot_right = z[2:, 2:]

        # dx: right - left
        dz_dx[1:-1, 1:-1] = ((top_right + 2.0 * mid_right + bot_right) -
                             (top_left + 2.0 * mid_left + bot_left)) / (8.0 * dx)
        # dy: top - bottom
        dz_dy[1:-1, 1:-1] = ((top_left + 2.0 * top_mid + top_right) -
                             (bot_left + 2.0 * bot_mid + bot_right)) / (8.0 * dy)

        # Boundaries filled with central difference fallback
        dz_dx[0, 1:-1] = (z[0, 2:] - z[0, :-2]) / (2.0 * dx)
        dz_dx[-1, 1:-1] = (z[-1, 2:] - z[-1, :-2]) / (2.0 * dx)
        dz_dx[:, 0] = dz_dx[:, 1]
        dz_dx[:, -1] = dz_dx[:, -2]

        dz_dy[1:-1, 0] = (z[:-2, 0] - z[2:, 0]) / (2.0 * dy)
        dz_dy[1:-1, -1] = (z[:-2, -1] - z[2:, -1]) / (2.0 * dy)
        dz_dy[0, :] = dz_dy[1, :]
        dz_dy[-1, :] = dz_dy[-2, :]

    else:
        raise ValueError(f"Unknown slope method: {method}")

    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    return slope


def compute_roughness(arr, window=3, method="std_slope", slope_arr=None, depth_arr=None):
    """
    Computes terrain roughness r as local standard deviation in a w x w window.
    method can be 'std_slope' or 'std_depth'.
    """
    target = slope_arr if method == "std_slope" else depth_arr
    if target is None:
        target = arr

    # Local variance: Var(X) = E[X^2] - (E[X])^2
    mean_target = uniform_filter(target, size=window, mode="reflect")
    mean_sq = uniform_filter(target**2, size=window, mode="reflect")
    var = np.maximum(mean_sq - mean_target**2, 0.0)
    roughness = np.sqrt(var)
    return roughness


def compute_curvature(z, dx=450.0, dy=450.0, method="laplacian"):
    """
    Computes terrain curvature c.
    Baseline: |Laplacian(z)| using a 5-point stencil.
    Alternative: Frobenius norm of the Hessian matrix.
    """
    ny, nx = z.shape

    # Second derivatives
    z_xx = np.zeros_like(z)
    z_xx[:, 1:-1] = (z[:, 2:] - 2.0 * z[:, 1:-1] + z[:, :-2]) / (dx**2)
    z_xx[:, 0] = z_xx[:, 1]
    z_xx[:, -1] = z_xx[:, -2]

    z_yy = np.zeros_like(z)
    z_yy[1:-1, :] = (z[:-2, :] - 2.0 * z[1:-1, :] + z[2:, :]) / (dy**2)
    z_yy[0, :] = z_yy[1, :]
    z_yy[-1, :] = z_yy[-2, :]

    if method == "laplacian":
        laplacian = z_xx + z_yy
        curvature = np.abs(laplacian)
    elif method == "hessian":
        # Cross derivative z_xy
        z_xy = np.zeros_like(z)
        z_xy[1:-1, 1:-1] = ((z[:-2, 2:] - z[:-2, :-2]) -
                            (z[2:, 2:] - z[2:, :-2])) / (4.0 * dx * dy)
        z_xy[0, :] = z_xy[1, :]
        z_xy[-1, :] = z_xy[-2, :]
        z_xy[:, 0] = z_xy[:, 1]
        z_xy[:, -1] = z_xy[:, -2]

        # Frobenius norm of Hessian: sqrt(z_xx^2 + 2*z_xy^2 + z_yy^2)
        curvature = np.sqrt(z_xx**2 + 2.0 * z_xy**2 + z_yy**2)
    else:
        raise ValueError(f"Unknown curvature method: {method}")

    return curvature


def normalize_feature(f, method="minmax", p_clip=99.0):
    """
    Normalizes a feature map to [0, 1].
    - 'minmax': standard min-max scaling.
    - 'percentile_99': clip values at 99th percentile first, then min-max scale.
    """
    f_arr = f.copy()
    if method == "percentile_99":
        upper = np.percentile(f_arr, p_clip)
        f_arr = np.clip(f_arr, 0.0, upper)

    f_min = np.min(f_arr)
    f_max = np.max(f_arr)
    if f_max - f_min < 1e-12:
        return np.zeros_like(f_arr)
    return (f_arr - f_min) / (f_max - f_min)


def build_risk_field(s_norm, r_norm, c_norm, weights=None):
    """
    Builds the composite terrain risk field R(x, y):
    R(x, y) = w_s * s_tilde^2 + w_r * r_tilde + w_c * c_tilde
    Notice slope is squared to heavily penalize steep terrain.
    """
    if weights is None:
        weights = {"w_s": 0.5, "w_r": 0.3, "w_c": 0.2}

    w_s = weights.get("w_s", 0.5)
    w_r = weights.get("w_r", 0.3)
    w_c = weights.get("w_c", 0.2)

    risk = w_s * (s_norm**2) + w_r * r_norm + w_c * c_norm
    return risk


def derive_features(data_dict, config=None, overrides=None):
    """
    Full pipeline to derive s, r, c, normalized features, and risk field R.
    Accepts config dict and optional parameter overrides for sensitivity analysis.
    """
    z_raw = data_dict["z"]
    dx = data_dict["metadata"]["dx_m"]
    dy = data_dict["metadata"]["dy_m"]

    # Parameter extraction with overrides
    params = {
        "slope_method": "central_difference",
        "roughness_method": "std_slope",
        "roughness_window": 3,
        "curvature_method": "laplacian",
        "smoothing_sigma": 0,
        "normalization": "minmax",
        "w_s": 0.5,
        "w_r": 0.3,
        "w_c": 0.2
    }

    if config is not None and "features" in config:
        f_cfg = config["features"]
        params["slope_method"] = f_cfg.get("slope_method", params["slope_method"])
        params["roughness_method"] = f_cfg.get("roughness_method", params["roughness_method"])
        params["roughness_window"] = f_cfg.get("roughness_window", params["roughness_window"])
        params["curvature_method"] = f_cfg.get("curvature_method", params["curvature_method"])
        params["smoothing_sigma"] = f_cfg.get("smoothing_sigma", params["smoothing_sigma"])
        params["normalization"] = f_cfg.get("normalization", params["normalization"])
        rw = f_cfg.get("risk_weights", {})
        params["w_s"] = rw.get("w_s", params["w_s"])
        params["w_r"] = rw.get("w_r", params["w_r"])
        params["w_c"] = rw.get("w_c", params["w_c"])

    if overrides is not None:
        params.update(overrides)

    # 1. Pre-smoothing
    sigma = params["smoothing_sigma"]
    if sigma > 0:
        z = gaussian_filter(z_raw, sigma=sigma, mode="reflect")
    else:
        z = z_raw.copy()

    d = -z # water depth

    # 2. Derive raw features
    s = compute_slope(z, dx=dx, dy=dy, method=params["slope_method"])

    r = compute_roughness(
        arr=s if params["roughness_method"] == "std_slope" else d,
        window=params["roughness_window"],
        method=params["roughness_method"],
        slope_arr=s,
        depth_arr=d
    )

    c = compute_curvature(z, dx=dx, dy=dy, method=params["curvature_method"])

    # 3. Normalization
    norm_method = params["normalization"]
    s_norm = normalize_feature(s, method=norm_method)
    r_norm = normalize_feature(r, method=norm_method)
    c_norm = normalize_feature(c, method=norm_method)

    # 4. Composite Risk Field
    weights = {"w_s": params["w_s"], "w_r": params["w_r"], "w_c": params["w_c"]}
    R = build_risk_field(s_norm, r_norm, c_norm, weights=weights)

    return {
        "z": z,
        "d": d,
        "s": s,
        "r": r,
        "c": c,
        "s_norm": s_norm,
        "r_norm": r_norm,
        "c_norm": c_norm,
        "R": R,
        "params": params
    }


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)
    print("Features successfully derived!")
    for key in ["s", "r", "c", "R"]:
        arr = feat[key]
        print(f"  {key}: min={arr.min():.5f}, mean={arr.mean():.5f}, max={arr.max():.5f}")
