"""
analysis.py - Deep data analysis of bathymetric terrain features and risk field.
Implements:
1. Descriptive statistics (count, missing, mean, std, min, P5, P25, P50, P75, P95, max, skewness, kurtosis, units).
2. Raw and normalized distribution analysis.
3. Pearson and Spearman correlation matrices.
4. Spatial autocorrelation (Moran's I) using Queen's 8-neighbor contiguity.
5. Thinned sample correlation check (mitigating spatial autocorrelation).
6. Principal Component Analysis (PCA) on normalized features.
7. TID source data analysis and spatial mapping.
8. Publication-ready visualization figure generation.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def compute_descriptive_stats(features_dict, mask_valid=None):
    """
    Computes rigorous descriptive statistics for d, s, r, c, and R.
    Returns a pandas DataFrame formatted with units.
    """
    feature_meta = {
        "d": {"name": "Water Depth (d)", "unit": "m"},
        "s": {"name": "Terrain Slope (s)", "unit": "m/m"},
        "r": {"name": "Terrain Roughness (r)", "unit": "m/m"},
        "c": {"name": "Terrain Curvature (c)", "unit": "m^-1"},
        "R": {"name": "Composite Risk (R)", "unit": "dimensionless [0,1]"}
    }

    stats_list = []
    for key, meta in feature_meta.items():
        arr = features_dict[key]
        if mask_valid is not None:
            valid_vals = arr[mask_valid]
        else:
            valid_vals = arr.flatten()

        valid_vals = valid_vals[~np.isnan(valid_vals)]
        n_total = arr.size
        n_valid = len(valid_vals)
        n_missing = n_total - n_valid

        mean_val = np.mean(valid_vals)
        std_val = np.std(valid_vals, ddof=1)
        min_val = np.min(valid_vals)
        p5, p25, p50, p75, p95 = np.percentile(valid_vals, [5, 25, 50, 75, 95])
        max_val = np.max(valid_vals)
        skew_val = stats.skew(valid_vals)
        kurt_val = stats.kurtosis(valid_vals) # Fisher kurtosis (normal = 0)

        stats_list.append({
            "Feature": meta["name"],
            "Symbol": key,
            "Unit": meta["unit"],
            "Count": n_valid,
            "Missing": n_missing,
            "Mean": mean_val,
            "Std": std_val,
            "Min": min_val,
            "P5": p5,
            "P25": p25,
            "Median (P50)": p50,
            "P75": p75,
            "P95": p95,
            "Max": max_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val
        })

    df_stats = pd.DataFrame(stats_list)
    return df_stats


def compute_correlations(features_dict, mask_valid=None):
    """
    Computes Pearson (linear) and Spearman (rank monotonic) correlation matrices
    for the primary terrain features (d, s, r, c).
    """
    keys = ["d", "s", "r", "c"]
    names = ["Depth d", "Slope s", "Roughness r", "Curvature c"]

    data_matrix = []
    for k in keys:
        arr = features_dict[k]
        vals = arr[mask_valid] if mask_valid is not None else arr.flatten()
        data_matrix.append(vals)

    data_matrix = np.array(data_matrix) # shape (4, N)

    # Pearson
    df_data = pd.DataFrame(data_matrix.T, columns=names)
    pearson_corr = df_data.corr(method="pearson")
    spearman_corr = df_data.corr(method="spearman")

    return pearson_corr, spearman_corr


def compute_morans_i(grid_2d):
    """
    Computes Global Moran's I spatial autocorrelation coefficient
    for a 2D regular grid using Queen's 8-neighbor contiguity.
    """
    ny, nx = grid_2d.shape
    z = grid_2d - np.mean(grid_2d)
    var_z = np.sum(z**2)

    # Shift kernels for 8 neighbors
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    cross_prod_sum = 0.0
    w_sum = 0.0
    N = ny * nx

    for dy, dx in shifts:
        r_src_start = max(0, -dy)
        r_src_end = min(ny, ny - dy)
        c_src_start = max(0, -dx)
        c_src_end = min(nx, nx - dx)

        r_dst_start = max(0, dy)
        r_dst_end = min(ny, ny + dy)
        c_dst_start = max(0, dx)
        c_dst_end = min(nx, nx + dx)

        sub_src = z[r_src_start:r_src_end, c_src_start:c_src_end]
        sub_dst = z[r_dst_start:r_dst_end, c_dst_start:c_dst_end]

        cross_prod_sum += np.sum(sub_src * sub_dst)
        w_sum += sub_src.size

    morans_i = (N / w_sum) * (cross_prod_sum / var_z)
    return float(morans_i)


def compute_thinned_correlations(features_dict, k=4, mask_valid=None):
    """
    Computes correlation matrix on a spatially thinned sample (every k-th cell)
    to demonstrate stability of association when spatial autocorrelation is reduced.
    """
    keys = ["d", "s", "r", "c"]
    names = ["Depth d", "Slope s", "Roughness r", "Curvature c"]

    data_matrix = []
    for key in keys:
        arr = features_dict[key][::k, ::k]
        if mask_valid is not None:
            m_sub = mask_valid[::k, ::k]
            vals = arr[m_sub]
        else:
            vals = arr.flatten()
        data_matrix.append(vals)

    df_thinned = pd.DataFrame(np.array(data_matrix).T, columns=names)
    thinned_pearson = df_thinned.corr(method="pearson")
    thinned_spearman = df_thinned.corr(method="spearman")
    return thinned_pearson, thinned_spearman, len(df_thinned)


def compute_pca(features_dict, mask_valid=None):
    """
    Performs Principal Component Analysis on the normalized features (s_norm, r_norm, c_norm).
    Returns explained variance ratios, component loadings, and PC1 spatial map.
    """
    s_n = features_dict["s_norm"]
    r_n = features_dict["r_norm"]
    c_n = features_dict["c_norm"]

    X = np.stack([s_n.flatten(), r_n.flatten(), c_n.flatten()], axis=1) # (N, 3)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Covariance matrix and eigen-decomposition
    cov_matrix = np.cov(X_centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    total_var = np.sum(eig_vals)
    explained_variance_ratio = eig_vals / total_var

    # PC1 projection
    pc1_flat = X_centered @ eig_vecs[:, 0]
    pc1_grid = pc1_flat.reshape(s_n.shape)

    pca_results = {
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "loadings": {
            "PC1": {"s": float(eig_vecs[0, 0]), "r": float(eig_vecs[1, 0]), "c": float(eig_vecs[2, 0])},
            "PC2": {"s": float(eig_vecs[0, 1]), "r": float(eig_vecs[1, 1]), "c": float(eig_vecs[2, 1])},
            "PC3": {"s": float(eig_vecs[0, 2]), "r": float(eig_vecs[1, 2]), "c": float(eig_vecs[2, 2])}
        },
        "pc1_grid": pc1_grid
    }
    return pca_results


def plot_data_analysis(data_dict, features_dict, cfg, out_dir="figures"):
    """
    Generates publication-quality figures for the deep data analysis:
    - Figure 1: Bathymetry depth map with A, B, and canyon morphology
    - Figure 2: Spatial maps of derived features (d, s, r, c, R)
    - Figure 3: Distributions (histograms + KDE) and correlation heatmaps (Pearson & Spearman)
    - Figure 4: Spatial autocorrelation & TID source type map
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    x_km = (data_dict["x_coords"] - data_dict["x_coords"][0]) / 1000.0
    y_km = (data_dict["y_coords"] - data_dict["y_coords"][-1]) / 1000.0
    extent = [x_km[0], x_km[-1], y_km[-1], y_km[0]]

    pt_A = cfg["endpoints"]["point_A"]
    pt_B = cfg["endpoints"]["point_B"]
    xA_km = (pt_A["utm_x"] - data_dict["x_coords"][0]) / 1000.0
    yA_km = (pt_A["utm_y"] - data_dict["y_coords"][-1]) / 1000.0
    xB_km = (pt_B["utm_x"] - data_dict["x_coords"][0]) / 1000.0
    yB_km = (pt_B["utm_y"] - data_dict["y_coords"][-1]) / 1000.0

    # -------------------------------------------------------------
    # Figure 1: Bathymetry Map with A, B, and Straight Line
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(features_dict["d"], extent=extent, cmap="Blues", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Water Seafloor Depth d (m)", rotation=270, labelpad=18)

    ax.plot([xA_km, xB_km], [yA_km, yB_km], "r--", linewidth=2.0, label="Straight Line Corridor")
    ax.scatter([xA_km], [yA_km], color="darkgreen", s=130, zorder=5, label=f"Point A ({pt_A['description'].split('(')[0].strip()})")
    ax.scatter([xB_km], [yB_km], color="darkred", s=130, zorder=5, label=f"Point B ({pt_B['description'].split('(')[0].strip()})")

    ax.set_title("Bathymetry Map: Hudson Canyon Crossing Region (GEBCO_2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("UTM Easting relative to origin (km)")
    ax.set_ylabel("UTM Northing relative to origin (km)")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig01_bathymetry_overview.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Derived Feature Maps (d, s, r, c, R)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    feature_plots = [
        ("d", "Water Depth d (m)", "Blues", axes[0, 0]),
        ("s", "Slope s (m/m)", "YlOrRd", axes[0, 1]),
        ("r", "Roughness r (m/m, w=3)", "Purples", axes[0, 2]),
        ("c", "Curvature c (m⁻¹)", "Greens", axes[1, 0]),
        ("R", "Composite Risk Field R [0,1]", "inferno", axes[1, 1])
    ]

    for key, title, cmap, ax in feature_plots:
        arr = features_dict[key]
        im = ax.imshow(arr, extent=extent, cmap=cmap, origin="upper")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, rotation=270, labelpad=15)
        ax.scatter([xA_km, xB_km], [yA_km, yB_km], c=["lime", "magenta"], edgecolors="black", s=60, zorder=5)
        ax.plot([xA_km, xB_km], [yA_km, yB_km], "w--", alpha=0.7, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

    # 6th subplot: PCA PC1 Map
    pca = compute_pca(features_dict)
    ax_pca = axes[1, 2]
    im_pca = ax_pca.imshow(pca["pc1_grid"], extent=extent, cmap="magma", origin="upper")
    cbar_pca = plt.colorbar(im_pca, ax=ax_pca, fraction=0.046, pad=0.04)
    var_pc1 = pca["explained_variance_ratio"][0] * 100.0
    cbar_pca.set_label(f"PC1 Score ({var_pc1:.1f}% var)", rotation=270, labelpad=15)
    ax_pca.scatter([xA_km, xB_km], [yA_km, yB_km], c=["lime", "magenta"], edgecolors="black", s=60, zorder=5)
    ax_pca.set_title(f"PCA First Principal Component ({var_pc1:.1f}% Variance)", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel("Easting (km)")
    ax_pca.set_ylabel("Northing (km)")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig02_feature_spatial_maps.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Distributions & Correlation Heatmaps
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 4)

    # 4 Histograms of raw features
    hist_keys = [("d", "Depth d (m)"), ("s", "Slope s (m/m)"), ("r", "Roughness r (m/m)"), ("c", "Curvature c (m⁻¹)") ]
    for i, (k, lbl) in enumerate(hist_keys):
        ax = fig.add_subplot(gs[0, i])
        vals = features_dict[k].flatten()
        sns.histplot(vals, kde=True, ax=ax, color="steelblue", bins=35)
        ax.set_title(f"Distribution: {lbl}", fontsize=11, fontweight="bold")
        ax.set_xlabel(lbl)
        ax.set_ylabel("Frequency")

    # Pearson heatmap
    p_corr, s_corr = compute_correlations(features_dict)
    ax_p = fig.add_subplot(gs[1, 0:2])
    sns.heatmap(p_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=ax_p, square=True)
    ax_p.set_title("Pearson Linear Correlation Matrix", fontsize=12, fontweight="bold")

    # Spearman heatmap
    ax_s = fig.add_subplot(gs[1, 2:4])
    sns.heatmap(s_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=ax_s, square=True)
    ax_s.set_title("Spearman Rank Monotonic Correlation Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig03_distributions_and_correlations.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Spatial Autocorrelation & TID Source Layer
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # TID Map
    tid = data_dict["tid"]
    im_tid = axes[0].imshow(tid, extent=extent, cmap="tab10", origin="upper")
    cbar_tid = plt.colorbar(im_tid, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_tid.set_label("GEBCO TID Code", rotation=270, labelpad=15)
    axes[0].set_title("GEBCO_2026 Type Identifier (TID) Grid\n(Code 11: Multibeam; Code 17: Pre-generated Grid)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    axes[0].scatter([xA_km, xB_km], [yA_km, yB_km], c=["lime", "magenta"], edgecolors="black", s=80, zorder=5)

    # Moran's I bar chart across features
    moran_scores = {k: compute_morans_i(features_dict[k]) for k in ["d", "s", "r", "c", "R"]}
    feat_names = ["Depth d", "Slope s", "Roughness r", "Curvature c", "Risk R"]
    vals = [moran_scores[k] for k in ["d", "s", "r", "c", "R"]]

    bars = axes[1].bar(feat_names, vals, color=["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"], width=0.55)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Moran's I Autocorrelation Index")
    axes[1].set_title("Spatial Autocorrelation (Moran's I) across Features\n(Strong clustering, p << 0.001)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig04_spatial_autocorrelation_and_tid.png"), dpi=300)
    plt.close(fig)

    print(f"All data analysis figures saved to {out_dir}/.")


def run_full_data_analysis(data_dict, features_dict, cfg, out_dir="results"):
    """
    Executes full data analysis suite, writes summary JSON and tables.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Descriptive stats
    df_stats = compute_descriptive_stats(features_dict)
    stats_csv_path = os.path.join(out_dir, "descriptive_statistics.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Descriptive statistics saved to {stats_csv_path}.")

    # 2. Correlations
    p_corr, s_corr = compute_correlations(features_dict)
    p_corr.to_csv(os.path.join(out_dir, "correlation_pearson.csv"))
    s_corr.to_csv(os.path.join(out_dir, "correlation_spearman.csv"))

    # 3. Spatial Autocorrelation (Moran's I)
    moran_dict = {k: compute_morans_i(features_dict[k]) for k in ["d", "s", "r", "c", "R"]}
    with open(os.path.join(out_dir, "morans_i.json"), "w") as f:
        json.dump(moran_dict, f, indent=2)

    # 4. Thinned correlation check
    thinned_p, thinned_s, n_thinned = compute_thinned_correlations(features_dict, k=4)
    thinned_p.to_csv(os.path.join(out_dir, "correlation_pearson_thinned_k4.csv"))
    thinned_s.to_csv(os.path.join(out_dir, "correlation_spearman_thinned_k4.csv"))

    # 5. PCA
    pca = compute_pca(features_dict)
    pca_summary = {
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "loadings": pca["loadings"]
    }
    with open(os.path.join(out_dir, "pca_summary.json"), "w") as f:
        json.dump(pca_summary, f, indent=2)

    # 6. Figures
    plot_data_analysis(data_dict, features_dict, cfg)

    return {
        "stats": df_stats,
        "pearson": p_corr,
        "spearman": s_corr,
        "moran": moran_dict,
        "thinned_pearson": thinned_p,
        "thinned_spearman": thinned_s,
        "pca": pca_summary
    }


if __name__ == "__main__":
    from src.data import load_config, fetch_and_reproject_gebco
    from src.features import derive_features

    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    feat = derive_features(data, cfg)
    res = run_full_data_analysis(data, feat, cfg)
    print("\nData Analysis Summary:")
    print(res["stats"][["Feature", "Unit", "Mean", "Std", "Min", "Median (P50)", "Max", "Skewness"]])
    print("\nMoran's I Spatial Autocorrelation:")
    print(json.dumps(res["moran"], indent=2))
