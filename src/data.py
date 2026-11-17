"""
data.py - Data ingestion, projected CRS resampling, masking, and distance validation.
Handles GEBCO_2026 bathymetry and TID layer subsets.
"""

import os
import json
import yaml
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import pyproj


def load_config(config_path="config.yaml"):
    """Load configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_and_reproject_gebco(cfg, force_reload=False):
    """
    Downloads/extracts GEBCO_2026 bathymetry and TID grid subsets from CEDA archive,
    reprojects to local projected UTM CRS on a regular metric grid, and caches locally.
    """
    cache_file = os.path.join("data", "processed", "bathymetry_utm.npz")
    meta_file = os.path.join("data", "processed", "metadata.json")

    if not force_reload and os.path.exists(cache_file) and os.path.exists(meta_file):
        print(f"Loading cached bathymetry dataset from {cache_file}...")
        npz = np.load(cache_file)
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        return {
            "z": npz["z"],
            "d": npz["d"],
            "tid": npz["tid"],
            "x_coords": npz["x_coords"],
            "y_coords": npz["y_coords"],
            "mask_land": npz["mask_land"],
            "metadata": metadata
        }

    print("Fetching GEBCO_2026 data from CEDA Archive...")
    bathy_url = (
        "/vsicurl/https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/"
        "sub_ice_topography_bathymetry/geotiff/"
        "gebco_2026_sub_ice_n90.0_s0.0_w-90.0_e0.0_geotiff.tif"
    )
    tid_url = (
        "/vsicurl/https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/"
        "type_identifier_grid/geotiff/"
        "gebco_2026_tid_n90.0_s0.0_w-90.0_e0.0_geotiff.tif"
    )

    grid_cfg = cfg["grid"]
    nx = grid_cfg["nx"]
    ny = grid_cfg["ny"]
    dx = grid_cfg["dx"]
    dy = grid_cfg["dy"]
    xmin = grid_cfg["xmin"]
    xmax = grid_cfg["xmax"]
    ymin = grid_cfg["ymin"]
    ymax = grid_cfg["ymax"]
    dst_crs = cfg["metadata"]["crs_projected"]

    # Source bounding box with safe buffer
    w_buf, s_buf = grid_cfg["lon_bounds"][0] - 0.05, grid_cfg["lat_bounds"][0] - 0.05
    e_buf, n_buf = grid_cfg["lon_bounds"][1] + 0.05, grid_cfg["lat_bounds"][1] + 0.05

    # 1. Read Bathymetry
    with rasterio.open(bathy_url) as src_bathy:
        win_bathy = from_bounds(w_buf, s_buf, e_buf, n_buf, src_bathy.transform)
        bathy_raw = src_bathy.read(1, window=win_bathy)
        bathy_transform = src_bathy.window_transform(win_bathy)
        bathy_crs = src_bathy.crs

    # 2. Read TID Layer
    with rasterio.open(tid_url) as src_tid:
        win_tid = from_bounds(w_buf, s_buf, e_buf, n_buf, src_tid.transform)
        tid_raw = src_tid.read(1, window=win_tid)
        tid_transform = src_tid.window_transform(win_tid)
        tid_crs = src_tid.crs

    # Destination transform
    dst_transform = from_origin(xmin - dx / 2.0, ymax + dy / 2.0, dx, dy)

    # Reproject Bathymetry
    z = np.empty((ny, nx), dtype=np.float32)
    reproject(
        bathy_raw,
        z,
        src_transform=bathy_transform,
        src_crs=bathy_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    # Reproject TID (nearest neighbor to preserve discrete category IDs)
    tid = np.empty((ny, nx), dtype=np.int32)
    reproject(
        tid_raw,
        tid,
        src_transform=tid_transform,
        src_crs=tid_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    # 1D coordinate vectors
    x_coords = np.linspace(xmin, xmax, nx)
    # y_coords descends from ymax to ymin as row index increases
    y_coords = np.linspace(ymax, ymin, ny)

    # Water depth d = -z (positive down)
    d = -z
    # Land mask (z >= 0)
    mask_land = z >= 0.0

    metadata = {
        "dataset": cfg["metadata"]["dataset"],
        "version": cfg["metadata"]["version"],
        "source_url": cfg["metadata"]["source_url"],
        "citation": cfg["metadata"]["citation"],
        "crs_projected": dst_crs,
        "crs_geographic": cfg["metadata"]["crs_geographic"],
        "grid_shape": [int(ny), int(nx)],
        "dx_m": float(dx),
        "dy_m": float(dy),
        "utm_bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "geographic_bounds": grid_cfg["lon_bounds"] + grid_cfg["lat_bounds"],
        "resampling_method": grid_cfg["resampling_method"],
        "nodata_value": grid_cfg["nodata_value"],
        "water_cells": int(np.sum(~mask_land)),
        "land_cells": int(np.sum(mask_land)),
        "depth_range_m": [float(np.min(d)), float(np.max(d))],
        "elevation_range_m": [float(np.min(z)), float(np.max(z))]
    }

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    np.savez_compressed(
        cache_file,
        z=z,
        d=d,
        tid=tid,
        x_coords=x_coords,
        y_coords=y_coords,
        mask_land=mask_land
    )
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset successfully processed and cached to {cache_file}.")
    return {
        "z": z,
        "d": d,
        "tid": tid,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "mask_land": mask_land,
        "metadata": metadata
    }


def validate_distance(pt_a, pt_b, crs_projected="EPSG:32618"):
    """
    Validates physical distance calculation by comparing Euclidean distance
    in the projected UTM grid with the true WGS84 geodesic distance (pyproj.Geod).
    Returns a dict with comparison statistics.
    """
    xA, yA = pt_a["utm_x"], pt_a["utm_y"]
    xB, yB = pt_b["utm_x"], pt_b["utm_y"]

    # Euclidean distance in projected plane
    dist_grid = float(np.hypot(xB - xA, yB - yA))

    # Geodesic distance on WGS84 ellipsoid
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_geod = geod.inv(pt_a["lon"], pt_a["lat"], pt_b["lon"], pt_b["lat"])
    dist_geod = float(dist_geod)

    diff_m = dist_grid - dist_geod
    diff_pct = (diff_m / dist_geod) * 100.0

    validation_result = {
        "point_A": {"utm": [xA, yA], "geo": [pt_a["lon"], pt_a["lat"]]},
        "point_B": {"utm": [xB, yB], "geo": [pt_b["lon"], pt_b["lat"]]},
        "distance_grid_m": dist_grid,
        "distance_geodesic_m": dist_geod,
        "difference_m": diff_m,
        "difference_percent": diff_pct,
        "passed": abs(diff_pct) < 0.1 # Must be well within 0.1% for local UTM
    }
    return validation_result


if __name__ == "__main__":
    cfg = load_config()
    data = fetch_and_reproject_gebco(cfg)
    val = validate_distance(cfg["endpoints"]["point_A"], cfg["endpoints"]["point_B"])
    print("Distance validation result:")
    print(json.dumps(val, indent=2))
