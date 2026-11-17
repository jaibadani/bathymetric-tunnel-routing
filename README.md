# Underwater Tunnel Route Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Question:** *How do seafloor terrain characteristics, and the choices we make when deriving them, change the feasibility, length, terrain risk, and robustness of optimized underwater tunnel routes?*  
> **Core Takeaway:** *The shortest feasible tunnel is not always the best tunnel.* Real-world infrastructure demands continuous curvature continuity, physical clearance, and terrain risk mitigation.

---

## 1. Project Overview

This repository provides an end-to-end, reproducible implementation of the **Underwater Tunnel Route Optimization Build Specification (v2, merged)**.

We formulate, solve, verify, and benchmark two optimization paradigms on real bathymetry from the **GEBCO_2026 Grid** across a $49.5\text{ km} \times 49.5\text{ km}$ crossing in the **Hudson Canyon Continental Margin**:
1. **Easy Problem (Discrete Global Optimality):** Solved on an 8-neighbor grid graph via **Dijkstra's algorithm**, minimizing path length under depth ($d_{\min} \le d \le d_{\max}$) and slope ($s \le s_{\max}$) constraints.
2. **Hard Problem (Continuous Local Optimality):** Formulated as a $C^2$ continuous cubic spline parameterized by interior control points and solved via **Sequential Least Squares Programming (SLSQP)**, minimizing $J = \alpha L(p) + \beta S(p)$ under non-linear constraints on depth, slope, and route curvature ($|\kappa_{\text{route}}(t)| \le \kappa_{\max}$).

---

## 2. Key Results & Benchmark Comparison

| Metric | Easy Route (Dijkstra) | Hard Route (SLSQP Spline) | Difference / Impact |
|---|---|---|---|
| **Route Length $L$** | 47,955.84 m (47.96 km) | 60,017.94 m (60.02 km) | **+12,062.10 m (+25.15% Price of Realism)** |
| **Cumulative Risk $S$** | 493.1159 | 119.2297 | **-373.8862 (-75.82% Risk Reduction)** |
| **Composite Objective $J$** | 24,703,748.82 | 6,021,503.03 | **-75.63% Penalty Reduction** |
| **Max Route Curvature $\kappa$** | $0.004260\text{ m}^{-1}$ ($R=235\text{ m}$) | $0.000487\text{ m}^{-1}$ ($R=2,053\text{ m}$) | **Easy Violates (x8.5); Hard Strictly Feasible** |
| **Max Terrain Slope $s$** | $0.0739\text{ m/m}$ (4.23°) | $0.0049\text{ m/m}$ (0.28°) | Hard alignment is 15.1× flatter |
| **Runtime** | 23.24 ms | 1,701.79 ms | 73.2× computational ratio |
| **Optimality Scope** | Global Discrete (on graph) | Local Continuous (on NLP manifold) | Fundamental theoretical trade-off |

---

## 3. Directory Layout

```
optimisation/
├── README.md             # Reproduction guide, data citation, environment, and findings
├── requirements.txt      # Python dependencies
├── config.yaml           # Centralized configuration (region, A/B, parameters, weights)
├── run_all.py            # Master runner executing the entire pipeline end-to-end
├── data/
│   ├── raw/              # Raw data downloads
│   └── processed/        # Reprojected UTM arrays (bathymetry_utm.npz, metadata.json)
├── src/
│   ├── data.py           # Ingestion, UTM Zone 18N metric reprojection, geodesic validation
│   ├── features.py       # Slope, roughness, curvature, smoothing, normalization, risk R
│   ├── analysis.py       # Descriptive stats, correlations, Moran's I, PCA, Figures 1-4
│   ├── easy.py           # 8-neighbor Dijkstra shortest path with NetworkX cross-check
│   ├── hard.py           # Multi-start continuous cubic spline with SciPy SLSQP
│   ├── metrics.py        # Common L, S, J evaluation, Price of Realism, Hausdorff distance
│   ├── verification.py   # Section 11 audit (Toy Tests 1, 2, 3, density study, Figure 10)
│   └── sensitivity.py    # S-core, full matrix, Pareto sweep, Figures 5-9
├── figures/              # Publication-ready visualization figures (300 DPI)
├── results/              # Per-run CSV tables, JSON metrics, and verification logs
├── presentation/         # 12-slide presentation deck (slides.md, index.html)
├── report/               # Final comprehensive technical report (FINAL_REPORT.md)
└── notebooks/
    ├── 01_data_features_eda.ipynb
    └── 02_routes_comparison_sensitivity.ipynb
```

---

## 4. Installation & Environment Setup

Ensure Python 3.10+ is installed:
```bash
git clone <repo-url>
cd optimisation

# Install scientific dependencies
pip install -r requirements.txt
```

---

## 5. Reproduction & Execution

### Run Master Pipeline (End-to-End)
To run the full pipeline, execute all 10 verification checks, run all 26 sensitivity experiments, and regenerate all figures:
```bash
python3 run_all.py
```

### Run Individual Modules
1. **Data Ingestion & Geodesic Validation:**
   ```bash
   python3 src/data.py
   ```
2. **Feature Engineering & Risk Field:**
   ```bash
   python3 src/features.py
   ```
3. **Deep Data Analysis (Statistics, Moran's I, PCA, Figures 1–4):**
   ```bash
   python3 src/analysis.py
   ```
4. **Easy Problem (Dijkstra Shortest Path):**
   ```bash
   python3 src/easy.py
   ```
5. **Hard Problem (Continuous SLSQP Spline):**
   ```bash
   python3 src/hard.py
   ```
6. **Comparison & Price of Realism:**
   ```bash
   python3 src/metrics.py
   ```
7. **Verification Audit (Toy Tests 1, 2, 3, Figure 10):**
   ```bash
   python3 src/verification.py
   ```
8. **Sensitivity Experiments & Figures 5–9:**
   ```bash
   python3 src/sensitivity.py
   ```

---

## 6. Verification Audit Summary (Section 11)

All 10 compulsory verification checks pass with 100% compliance:
- **Check 1 (Endpoints):** Both routes start at Point A and end at Point B within $10^{-4}\text{ m}$.
- **Check 2 (Independent Recompute):** Separate independent integration verifies $L, S, J$ within $10^{-6}$.
- **Check 3 (Toy Test 1 - Easy):** Dijkstra matches hand-calculated obstacle detour ($113.14\text{ m}$).
- **Check 4 (Toy Test 2 - Hard):** SLSQP returns exact analytical straight line on flat terrain ($1,414.21\text{ m}$, $\kappa = 0$).
- **Check 5 (Toy Test 3 - Hard):** Symmetric Gaussian risk bump yields structurally symmetric detour ($|y(t)| = |y(1-t)|$).
- **Check 6 (NetworkX):** Cross-check confirms zero difference ($0.00\text{ m}$) against library shortest path.
- **Check 7 (Density Study):** Dense validation at 200, 1,000, and 5,000 samples verifies zero constraint violation growth.
- **Check 8 (Interpolator):** Bicubic overshoot is strictly bounded (<0.26%).
- **Check 9 (Multi-Start):** Evaluated across 10 distinct initializations; Dijkstra warm start avoids local traps.
- **Check 10 (Geodesic Distance):** Planar UTM grid distance matches WGS84 geodesic within **0.0199%** error.

---

## 7. Presentation & Report Deliverables

- **Interactive Presentation Deck:** Open `presentation/index.html` in any web browser, or read `presentation/slides.md`. Follows the 12-slide layout with designated visual slides.
- **Final Comprehensive Report:** Read `report/FINAL_REPORT.md` (15 sections following the course outline).

---

## 8. Dataset Citation

```bibtex
@misc{gebco_2026,
  author = {{GEBCO Bathymetric Compilation Group}},
  title  = {The GEBCO\_2026 Grid - a continuous terrain model of the global oceans and land},
  year   = {2026},
  publisher = {NERC EDS British Oceanographic Data Centre NOC},
  doi    = {10.5285/gebco_2026},
  url    = {https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/}
}
```
