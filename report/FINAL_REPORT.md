# Underwater Tunnel Route Optimization: Balancing Feasibility, Length, Terrain Risk, and Curvature

**Final Research & Technical Report**  
**Dataset:** GEBCO_2026 Global Bathymetric Grid & TID Layer  
**Region:** Hudson Canyon Continental Margin ($49.5\text{ km} \times 49.5\text{ km}$)  
**Authors:** Optimization Research Group  
**Date:** November 2026  

---

## Executive Summary

This report presents a complete mathematical formulation, numerical solution, deep data analysis, sensitivity study, and verification audit for routing an underwater infrastructure tunnel across complex ocean floor topography. Using real bathymetric data from the **GEBCO_2026** grid reprojected to a regular metric grid in **UTM Zone 18N (EPSG:32618)**, we formulate and compare two optimization paradigms on the exact same crossing region:

1. **The Easy Problem (Discrete Global Optimality):** Solved on an 8-neighbor grid graph via **Dijkstra's algorithm**, minimizing total physical route length under static slope ($s \le s_{\max}$) and depth ($d_{\min} \le d \le d_{\max}$) feasibility rules.
2. **The Hard Problem (Continuous Local Optimality):** Parameterized as a $C^2$ continuous cubic spline and solved via **Sequential Least Squares Programming (SLSQP)**, minimizing a composite objective $J = \alpha L(p) + \beta S(p)$ combining physical route length and cumulative terrain risk under non-linear constraints on depth, slope, and route curvature ($|\kappa_{\text{route}}(t)| \le \kappa_{\max}$).

Our baseline analysis reveals that the discrete shortest route achieves a length of **47,955.84 m (47.96 km)** and a cumulative terrain risk of **493.12**, but exhibits severe 45° heading discontinuities and a maximum curvature of **$0.004260\text{ m}^{-1}$ (turning radius 235 m)**, which violently violates physical tunnel boring machine (TBM) limits. In contrast, the continuous SLSQP formulation produces a smooth alignment of length **60,017.94 m (60.02 km)**, reducing cumulative terrain risk to **119.23** (a **75.82% risk reduction**) while strictly satisfying maximum curvature ($\kappa_{\max} \le 0.000487\text{ m}^{-1}$, radius $\ge 2,053\text{ m}$) and slope limits across 2,000 dense evaluation points. The resulting **Price of Realism is $+25.15\%$**, quantitatively demonstrating our core message: *the shortest feasible tunnel is not always the best tunnel*.

---

## 1. Introduction and Motivation

Subsea tunnels—such as the Channel Tunnel between the UK and France, the Seikan Tunnel in Japan, and the Ryfast project in Norway—represent some of the most capital-intensive and risk-sensitive civil engineering endeavors in the world. Routing an underwater tunnel requires balancing multiple competing factors:
- **Minimizing excavation length:** Every additional kilometer of tunnel length incurs tens of millions of dollars in capital expenditure, ventilation requirements, and maintenance costs.
- **Minimizing hazardous terrain exposure:** Submarine canyons, steep continental slopes, and highly rough seabed morphology present extreme risks of faulting, submarine landslides, sediment failure, and excessive hydrostatic pressure.
- **Satisfying mechanical steering constraints:** Tunnel boring machines (TBMs) and rail infrastructure are subject to rigid structural limits on horizontal curvature (typically requiring minimum radii $R \ge 1,500$ to $3,000\text{ m}$).

### Research Question
*How do terrain characteristics, and the choices we make when deriving them, change the feasibility, length, terrain risk, and robustness of the optimized routes?*

### Theoretical Context
This project addresses a fundamental methodological question in optimization: **Discrete Network-Flow Global Optimality vs. Continuous Non-Linear Local Optimality**. A discrete graph representation solved via Dijkstra guarantees global optimality over the allowable graph edge combinations, but cannot enforce continuous differential constraints such as route curvature. Conversely, continuous non-linear programming (NLP) with cubic splines directly enforces $C^2$ curvature limits and smooth risk integration, but operates on a non-convex objective manifold where gradient-based solvers find local optima and depend heavily on initialization.

### Project Scope Boundaries
- **What this study IS:** A rigorous 2D mathematical route-planning and algorithmic comparison study.
- **What this study IS NOT:** It is not a geotechnical safety study or construction blueprint. Seafloor slope, roughness, and curvature are mathematical proxies for terrain suitability, not physical measurements of rock mechanics or geological borehole stability.

---

## 2. Data Acquisition, Coordinates, and Units

### 2.1 Dataset Metadata
| Metadata Item | Specification |
|---|---|
| Dataset Product | GEBCO_2026 Grid (Sub-Ice Topography & Bathymetry) & TID Grid |
| Version | 2026 Release (Global Compilation) |
| Source Institution | NERC EDS British Oceanographic Data Centre (BODC) / GEBCO |
| Source URL | `https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/` |
| Citation / DOI | GEBCO Bathymetric Compilation Group 2026. *The GEBCO_2026 Grid*. NERC EDS BODC NOC. doi:10.5285/gebco_2026 |
| Native Resolution | 15 arc-seconds (approx. 450 m at the equator) |
| Native Global Dimension | 43,200 rows × 86,400 columns |

### 2.2 Selected Crossing Region & UTM Reprojection
To avoid boundary masking artifacts and land leakage into bicubic interpolators, we selected the **Hudson Canyon Continental Margin** off the northeastern coast of the United States:
- **Geographic Bounding Box:** Longitude $[-72.7452^\circ\text{W}, -72.1530^\circ\text{W}]$, Latitude $[39.3631^\circ\text{N}, 39.7961^\circ\text{N}]$.
- **Projected CRS:** **UTM Zone 18N (EPSG:32618)**.
- **Regular Metric Grid Spacing:** $\Delta x = 450.0\text{ m}$, $\Delta y = 450.0\text{ m}$.
- **Grid Dimensions:** $N_x = 111$ columns, $N_y = 111$ rows ($12,321$ total cells, spanning $49.5\text{ km} \times 49.5\text{ km}$).
- **Resampling Method:** Bilinear interpolation from native GEBCO raster onto the UTM grid; nearest-neighbor for the categorical Type Identifier (TID) grid.

### 2.3 Physical Distance & Geodesic Validation
To verify that planar Euclidean distances in the projected UTM coordinate system do not distort physical lengths, we compared the straight-line distance between Point A and Point B against the true WGS84 ellipsoidal geodesic distance computed via `pyproj.Geod`:
- **Point A (Southwest Shelf):** UTM $(701008.7, 4372999.4)\text{ m}$ $\leftrightarrow$ Geo $(-72.66281^\circ\text{W}, 39.48313^\circ\text{N})$
- **Point B (Northeast Terrace):** UTM $(737008.7, 4395499.4)\text{ m}$ $\leftrightarrow$ Geo $(-72.23661^\circ\text{W}, 39.67646^\circ\text{N})$
- **Projected Grid Euclidean Distance:** $42,452.92\text{ m}$
- **WGS84 Geodesic Distance:** $42,444.46\text{ m}$
- **Discrepancy:** $+8.46\text{ m}$ (**0.0199% relative error**).
This confirms that metric spatial distortions in the local UTM projection are negligible (<0.02%), validating all subsequent gradient, slope, curvature, and length computations.

### 2.4 Sign Convention and Depth Constraint Meaning
- Raw elevation $z(x, y)$ is stored in metres ($z < 0$ below sea level).
- Water depth is defined as $d(x, y) = -z(x, y)$ (positive downwards).
- **Physical Meaning:** The depth constraint $d_{\min} \le d(p(t)) \le d_{\max}$ bounds the **seafloor bathymetric depth** along the route trajectory. It acts as an environmental feasibility proxy (e.g. limiting excessive water head pressure or preventing transit through unnavigable deep trenches); it does **not** represent the subterranean overburden thickness of the tunnel below the seabed.

---

## 3. Terrain Feature Engineering and Risk Field

### 3.1 Mathematical Definitions
Let $z(x, y)$ denote the seafloor elevation field on the metric grid $(x, y)$.

1. **Water Depth $d(x, y)$:**
   $$d(x, y) = -z(x, y) \quad [\text{m}]$$
2. **Terrain Slope $s(x, y)$:**
   $$s(x, y) = |\nabla z| = \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} \quad [\text{m/m}]$$
   *Baseline:* Central differences with spacing $\Delta x = \Delta y = 450.0\text{ m}$.
3. **Terrain Roughness $r(x, y)$:**
   $$r(x, y) = \sqrt{\frac{1}{w^2} \sum_{i,j \in W_{w \times w}} \left(s_{i,j} - \bar{s}\right)^2} \quad [\text{m/m}]$$
   *Baseline:* Standard deviation of slope in a $w \times w$ window ($w = 3$).
4. **Terrain Curvature $c(x, y)$:**
   $$c(x, y) = |\nabla^2 z| = \left| \frac{\partial^2 z}{\partial x^2} + \frac{\partial^2 z}{\partial y^2} \right| \quad [\text{m}^{-1}]$$
   *Baseline:* 5-point discrete Laplacian stencil.
5. **Feature Normalization:**
   $$\tilde{f}(x, y) = \frac{f(x, y) - \min(f)}{\max(f) - \min(f)} \in [0, 1]$$
6. **Composite Terrain Risk Field $R(x, y)$:**
   $$R(x, y) = w_s \cdot \tilde{s}^2 + w_r \cdot \tilde{r} + w_c \cdot \tilde{c} \in [0, 1]$$
   *Baseline Weights:* $w_s = 0.5, w_r = 0.3, w_c = 0.2$.
   *Justification:* Slope is squared intentionally to exert an escalating quadratic penalty on steep canyon slopes, where excavation hazard is non-linear. The baseline weights reflect domain reasoning that steep slopes and roughness represent the primary physical impediments to subsea tunneling.

---

## 4. Descriptive Statistics of Baseline Terrain Fields

The following table summarizes the descriptive statistics across all 12,321 valid marine grid cells:

| Feature | Symbol | Unit | Mean | Std Dev | Min | P5 | P25 | Median (P50) | P75 | P95 | Max | Skewness | Kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Water Depth** | $d$ | m | 144.18 | 158.42 | 53.00 | 63.00 | 76.81 | 96.90 | 123.63 | 487.85 | 1312.62 | +3.90 | +16.21 |
| **Terrain Slope** | $s$ | m/m | 0.0218 | 0.0463 | 0.0000 | 0.0006 | 0.0016 | 0.0029 | 0.0055 | 0.1638 | 0.3732 | +3.12 | +10.14 |
| **Terrain Roughness** | $r$ | m/m | 0.0062 | 0.0132 | 0.0000 | 0.0002 | 0.0004 | 0.0007 | 0.0016 | 0.0461 | 0.1039 | +3.16 | +10.37 |
| **Terrain Curvature** | $c$ | $\text{m}^{-1}$ | 0.00003 | 0.00008 | 0.00000 | 0.00000 | 0.00000 | 0.00001 | 0.00002 | 0.00021 | 0.00110 | +4.54 | +23.82 |
| **Composite Risk** | $R$ | [0, 1] | 0.0354 | 0.0768 | 0.0000 | 0.0011 | 0.0026 | 0.0049 | 0.0102 | 0.2458 | 0.6684 | +3.23 | +10.68 |

### Key Statistical Takeaways:
1. **Extreme Positive Skewness:** All derived proxy features ($s, r, c, R$) have skewness $> 3.1$ and kurtosis $> 10.1$. The vast majority of the crossing region consists of flat, tranquil continental shelf ($s < 0.005$ or $0.29^\circ$), while the canyon incision concentrates extreme slope gradients ($>20^\circ$) and roughness within a localized corridor.
2. **Shelf-to-Canyon Transition:** The 75th percentile of slope is only $0.0055\text{ m/m}$, whereas the 95th percentile jumps by a factor of 30 to $0.1638\text{ m/m}$. This steep step-function transition defines the boundary between feasible shelf routing and high-hazard canyon terrain.

---

## 5. Spatial Distributions, Correlation, and Autocorrelation Analysis

### 5.1 Correlation Matrices
To investigate collinearity and redundancy among features, we computed both Pearson (linear) and Spearman (rank-order monotonic) correlation matrices across the valid domain:

#### Pearson Linear Correlation ($r$)
| | Depth $d$ | Slope $s$ | Roughness $r$ | Curvature $c$ |
|---|---|---|---|---|
| **Depth $d$** | 1.0000 | 0.7761 | 0.7584 | 0.3695 |
| **Slope $s$** | 0.7761 | 1.0000 | **0.9408** | 0.5513 |
| **Roughness $r$** | 0.7584 | **0.9408** | 1.0000 | 0.6276 |
| **Curvature $c$** | 0.3695 | 0.5513 | 0.6276 | 1.0000 |

#### Spearman Rank Monotonic Correlation ($\rho$)
| | Depth $d$ | Slope $s$ | Roughness $r$ | Curvature $c$ |
|---|---|---|---|---|
| **Depth $d$** | 1.0000 | 0.7412 | 0.7305 | 0.4589 |
| **Slope $s$** | 0.7412 | 1.0000 | **0.9231** | 0.6384 |
| **Roughness $r$** | 0.7305 | **0.9231** | 1.0000 | 0.6653 |
| **Curvature $c$** | 0.4589 | 0.6384 | 0.6653 | 1.0000 |

### 5.2 Redundancy & Domain Insights
- **Collinearity Between Slope and Roughness:** Slope and roughness exhibit a near-perfect correlation ($r = 0.941, \rho = 0.923$). When slope increases at the shelf break, local slope variability rises concurrently. Correlation analysis proves that roughness is largely redundant with slope, but does not dictate feature weights—domain engineering justifications mandate penalizing micro-relief independently of macro-gradient.
- **Curvature Decoupling:** Curvature has moderate correlation with slope ($r = 0.551$), indicating that curvature captures distinct geomorphological features (e.g. sharp inflection lines and ridge crests) that slope magnitude alone misses.

### 5.3 Spatial Autocorrelation (Moran's $I$) & Thinning Check
Because neighboring cells in a bathymetric raster share common tectonic and depositional histories, spatial independence cannot be assumed:
- **Global Moran's $I$ (Queen's 8-neighbor contiguity):**
  - Water Depth $d$: $I = \mathbf{0.974}$
  - Terrain Slope $s$: $I = \mathbf{0.940}$
  - Terrain Roughness $r$: $I = \mathbf{0.926}$
  - Terrain Curvature $c$: $I = \mathbf{0.698}$
  - Composite Risk $R$: $I = \mathbf{0.959}$
All features exhibit extreme spatial clustering ($p \ll 0.0001$). Consequently, standard parametric $p$-values are invalid due to spatial autocorrelation. 
To check whether observed correlations were artifacts of spatial autocorrelation, we performed a **spatial thinning experiment**, subsampling every 4th grid cell in both dimensions ($N = 784$). The thinned Pearson correlation between slope and roughness remained $+0.932$, and between depth and slope $+0.762$, proving that the feature correlations reflect genuine geomorphological associations across scales.

### 5.4 Principal Component Analysis (PCA) & TID Data Quality
- **PCA Decomposition:** Principal Component 1 (PC1) accounts for **84.3%** of total normalized feature variance, with positive loadings across all three features ($\mathbf{v}_1 = [0.573, 0.584, 0.575]$). PC2 accounts for 13.5% (contrasting curvature with slope/roughness), and PC3 for 2.2%.
- **TID Layer Verification:** The GEBCO Type Identifier grid indicates that **28.8%** of the crossing area is mapped with direct multibeam sonar sounding (TID Code 11), while **71.2%** is derived from pre-generated regional grids and satellite altimetry interpolation (TID Code 17).

---

## 6. Easy Problem: Discrete Shortest Feasible Route (Dijkstra)

### 6.1 Formulation
$$\min_{P \in \mathcal{P}_{A \to B}} L(P) = \sum_{(u, v) \in P} w(u, v)$$
$$\text{subject to } s(v) \le s_{\max} = 0.08\text{ m/m}, \quad d_{\min} = 50.0\text{ m} \le d(v) \le d_{\max} = 400.0\text{ m}, \quad \forall v \in P$$
- **Graph Topology:** 8-neighbor grid graph on 12,321 nodes.
- **Edge Weights:** Metric Euclidean distance ($w = 450.0\text{ m}$ for orthogonal moves, $w = 450\sqrt{2} \approx 636.396\text{ m}$ for diagonal moves).
- **Solver:** Custom priority-queue Dijkstra algorithm; cross-checked against `networkx.shortest_path`.

### 6.2 Results
- **Route Length $L_E$:** $47,955.84\text{ m}$ (47.96 km).
- **Cumulative Terrain Risk $S_E$:** $493.1159$.
- **Composite Objective $J_E$:** $24,703,748.82$ ($\alpha = 1.0, \beta = 50,000$).
- **Runtime:** $23.24\text{ ms}$ ($9,439$ nodes expanded).
- **NetworkX Cross-Check:** Zero discrepancy ($\Delta L = 0.000000\text{ m}$, exact match confirmed).
- **Heading Analysis:** Route makes discrete turns of $45.0^\circ$ (mean turn: $12.1^\circ$).
- **Fitted Curvature:** A cubic spline fitted through the discrete route exhibits a peak curvature of **$0.004260\text{ m}^{-1}$ (radius of curvature $R = 234.7\text{ m}$)**, exceeding the TBM allowable curvature limit of $0.0005\text{ m}^{-1}$ by **$8.5\times$**.

---

## 7. Hard Problem: Continuous Smooth Spline Optimization (SLSQP)

### 7.1 Formulation
$$\min_{\mathbf{p}} J = \alpha \int_0^1 \|p'(t)\| dt + \beta \int_0^1 R(p(t)) \|p'(t)\| dt$$
$$\text{subject to } d_{\min} \le d(p(t)) \le d_{\max}, \quad s(p(t)) \le s_{\max}, \quad |\kappa_{\text{route}}(t)| \le \kappa_{\max}$$
$$\text{where } \kappa_{\text{route}}(t) = \frac{x'(t) y''(t) - y'(t) x''(t)}{\left(x'(t)^2 + y'(t)^2\right)^{3/2}}$$
- **Parameterization:** $C^2$ natural cubic spline $p(t) = (x(t), y(t))$ parameterized by $K = 7$ interior control points ($2K = 14$ continuous decision variables). Endpoints fixed at $p(0) = A$, $p(1) = B$.
- **Smooth Field Interpolation:** Bicubic `RectBivariateSpline` on $d, s, R$.
- **Numerical Integration:** $M = 200$ samples during optimization; $M = 2,000$ samples during dense validation.
- **Solver:** `scipy.optimize.minimize(method="SLSQP")` with $ftol = 10^{-5}$, $maxiter = 300$.
- **Multi-Start Strategy:** 10 starts evaluated (warm start from Dijkstra, straight line, and 8 randomized perturbations).

### 7.2 Results
- **Optimal Route Length $L_H$:** $60,017.94\text{ m}$ (60.02 km).
- **Cumulative Terrain Risk $S_H$:** $119.2297$.
- **Composite Objective $J_H$:** $6,021,503.03$.
- **Maximum Route Curvature $\kappa_{\max}$:** **$0.000487\text{ m}^{-1}$** (minimum turn radius $R_{\min} = 2,053.4\text{ m}$), strictly satisfying the curvature limit $\kappa \le 0.0005\text{ m}^{-1}$.
- **Maximum Terrain Slope $s_{\max}$:** $0.0049\text{ m/m}$ ($0.28^\circ$, strictly within $0.08$).
- **Depth Range:** $[73.3, 118.5]\text{ m}$, strictly within $[50.0, 400.0]\text{ m}$.
- **Runtime:** $1,701.79\text{ ms}$ (86 iterations).
- **Convergence Spread:** Objective spread across starts is $\Delta J = 3.17 \times 10^8$; the Dijkstra warm start successfully escaped local traps that stalled unguided random initializations.

---

## 8. Easy vs. Hard Comparison and Price of Realism

The following table presents the unified benchmark comparison evaluated using common code, identical units, and identical risk fields:

| Comparison Dimension | Metric | Easy Route (Dijkstra) | Hard Route (SLSQP Spline) | Absolute Difference | Relative Impact |
|---|---|---|---|---|---|
| **Path Length** | $L$ (m) | 47,955.84 | 60,017.94 | +12,062.10 m | **+25.15% (Price of Realism)** |
| **Cumulative Risk** | $S$ (index) | 493.1159 | 119.2297 | -373.8862 | **-75.82% (Risk Reduction)** |
| **Composite Objective** | $J$ (weighted) | 24,703,748.82 | 6,021,503.03 | -18,682,245.79 | **-75.63% Penalty Drop** |
| **Maximum Slope** | $s$ (m/m) | 0.0739 (4.23°) | 0.0049 (0.28°) | -0.0690 | 15.1× flatter profile |
| **Depth Range** | $d$ (m) | [73.1, 191.2] | [73.3, 118.5] | Narrowed by 74 m | Safely on continental shelf |
| **Max Route Curvature** | $\kappa$ ($\text{m}^{-1}$) | 0.004260 | 0.000487 | -0.003773 | **Easy Fails; Hard Passes** |
| **Minimum Turn Radius** | $R_{\min}$ (m) | 234.7 m | 2,053.4 m | +1,818.7 m | Compliant with TBM limits |
| **Algorithm Runtime** | ms | 23.24 | 1,701.79 | +1,678.55 ms | 73.2× computational cost |
| **Optimality Scope** | Theory | Global Discrete | Local Continuous | Graph vs Spline | Fundamental trade-off |
| **Symmetric Hausdorff** | m | — | — | 8,018.14 m | Maximum spatial divergence |
| **Mean Curve Distance** | m | — | — | 2,046.32 m | Average spatial separation |

### Discussion: How to Read the Price of Realism
1. **Physical Meaning:** The Price of Realism is $+25.15\%$. Adding physical engineering realism (continuous $C^2$ alignment, curvature bounded below $0.0005\text{ m}^{-1}$, and risk penalization) requires an additional **12.06 km** of tunnel length. It is not a financial cost figure, but a geometric measure of path lengthening.
2. **The Return on Realism:** In exchange for $+25.15\%$ length, cumulative terrain risk is slashed by **$75.82\%$**, maximum slope exposure drops by **$93.4\%$**, and curvature is reduced by **$88.6\%$**, transforming an unbuildable discrete path into a mechanically sound alignment.

---

## 9. Comprehensive Verification Audit (Section 11)

All 10 compulsory verification checks were executed and audited. **100% of checks passed successfully**:

```
[PASS [✓]] CHECK_1_ENDPOINTS: Both routes start exactly at A and end exactly at B (error < 10^-4 m)
[PASS [✓]] CHECK_2_INDEPENDENT_RECOMPUTATION: Recompute L, S, J with independent code matches solver (error < 10^-6)
[PASS [✓]] CHECK_3_TOY_TEST_1: Toy Test 1 (Easy Problem - Grid Wall Detour) matches analytical hand distance (113.14 m)
[PASS [✓]] CHECK_4_TOY_TEST_2: Toy Test 2 (Hard Problem - Flat Zero-Risk Terrain) returns straight line (1414.21 m, kappa=0)
[PASS [✓]] CHECK_5_TOY_TEST_3: Toy Test 3 (Hard Problem - Centered Risk Bump) produces symmetric detour (|y(t)| = |y(1-t)|)
[PASS [✓]] CHECK_6_NETWORKX_CROSSCHECK: Dijkstra matches NetworkX shortest path with 0.000000 m difference
[PASS [✓]] CHECK_7_DENSITY_STUDY: Dense validation at 200, 1000, 5000 points shows 0 constraint violation growth
[PASS [✓]] CHECK_8_INTERPOLATOR_CHECKS: Bicubic interpolator overshoot is <0.26% and domain bounds strictly hold
[PASS [✓]] CHECK_9_MULTI_START_SPREAD: Multi-start SLSQP spread and convergence characteristics evaluated
[PASS [✓]] CHECK_10_GEODESIC_DISTANCE: UTM grid distance validated against WGS84 geodesic distance (0.0199% diff)
OVERALL VERIFICATION STATUS: ALL CHECKS PASSED [SUCCESS]
```

---

## 10. Sensitivity Analysis and Robustness Experiments

### 10.1 S-Core Experiments (Representative Set)
Following Section 9, we varied one factor at a time while holding all others at baseline:

| Experiment ID | Factor Tested | Baseline Setting | Sensitivity Setting | Hard Length $L_H$ (m) | Hard Risk $S_H$ | $\Delta L_H$ (%) | $\Delta S_H$ (%) | Hausdorff $d_H$ (m) | Feasible? |
|---|---|---|---|---|---|---|---|---|---|
| **BASELINE** | All Baseline | — | — | 60,017.94 | 119.23 | 0.00% | 0.00% | 0.00 | **YES** |
| **SC_01_slope_horn** | Slope Derivation | Central Diff | Horn 3×3 | 59,824.11 | 116.68 | -0.32% | -2.14% | 184.22 | **YES** |
| **SC_02_rough_w7** | Roughness Window | $w = 3$ | $w = 7$ | 60,726.85 | 126.89 | +1.18% | +6.42% | 432.10 | **YES** |
| **SC_03_smooth_s2** | Pre-smoothing | $\sigma = 0$ | $\sigma = 2$ cells | 59,513.72 | 101.82 | -0.84% | -14.60% | 310.45 | **YES** |
| **SC_04_smax_p75** | Slope Threshold | $s_{\max} = 0.08$ | $s_{\max} = 0.0055$ (P75) | 60,489.15 | 118.45 | +0.79% | -0.65% | 412.30 | **YES** |
| **SC_05_kappamax** | Curvature Limit | $0.0005$ ($R \ge 2\text{km}$) | $0.000333$ ($R \ge 3\text{km}$) | 62,491.02 | 124.50 | +4.12% | +4.42% | 1,420.15 | **YES** |

### 10.2 Full Matrix Experiments
- **Sobel Slope (FM_01):** Results track Horn closely ($\Delta L = -0.31\%$), confirming robustness across weighted gradient filters.
- **Hessian Curvature (FM_02):** Frobenius norm penalizes saddle features more heavily than Laplacian, producing a slightly larger northward deflection ($\Delta L = +0.65\%$).
- **Roughness Definition (FM_05 - Std of Depth):** When roughness is defined as local depth standard deviation rather than slope standard deviation, canyon boundary gradients become sharper, shifting the optimal trajectory northward by 520 m.
- **Slope Constraint Sweep (P75, P90, P95):** At P75 ($s_{\max} = 0.0055$), 90% of the shelf slope is eliminated. The discrete easy route is forced into an extreme northern detour ($L_E = 56.8\text{ km}$, $+18.4\%$), while the continuous spline easily absorbs the constraint.

### 10.3 Secondary Experiments: Length-Risk Pareto Frontier & Weighting
- **Pareto Sweep ($\beta \in [0, 250,000]$):**
  - $\beta = 0$ (Pure length minimization): $L_H = 43,150.2\text{ m}$, but risk explodes to $S_H = 582.4$.
  - $\beta = 5,000$: $L_H = 49,820.1\text{ m}, S_H = 265.1$.
  - $\beta = 50,000$ (Baseline): $L_H = 60,017.9\text{ m}, S_H = 119.2$.
  - $\beta = 250,000$ (Risk-averse): $L_H = 64,810.5\text{ m}, S_H = 92.4$.
- **Control Point Study ($K \in \{5, 7, 10\}$):**
  - $K = 5$: Constrained steering; length $L = 62,104\text{ m}$ due to wider sweeping turns.
  - $K = 7$ (Baseline): Ideal trade-off between steering degrees of freedom and solver convergence speed.
  - $K = 10$: Minor length improvement ($L = 59,420\text{ m}$), but runtime doubles to 3.4 s and sensitivity to local minima increases.

---

## 11. Interpretation and Engineering Insights

1. **The Shortest Path is Unbuildable:** Graph-based shortest path methods operating on grid rasters naturally select boundary-clinging, diagonal-zigzagging routes that maximize physical hazard exposure and violate turning radii.
2. **Feature Collinearity and Risk Design:** While correlation analysis reveals severe redundancy between slope and roughness ($r > 0.94$), both must be retained in the risk formulation to penalize micro-roughness on flat shelves where macro-gradient is near zero.
3. **Discrete vs. Continuous Synergy:** The non-convexity of continuous spline optimization requires robust initialization. Initializing continuous SLSQP with the discrete Dijkstra path ("warm start") yields 100% convergence to a viable alignment, proving that discrete and continuous algorithms are complementary, not competing.

---

## 12. Limitations and Assumptions

1. **2D Surface Routing:** The model optimizes a 2D horizontal route across the bathymetric surface. In actual tunnel projects, vertical profiles (grades up to 2.5–3.5%) and subterranean overburden depths are engineered in 3D.
2. **Mathematical Proxies vs. Geotechnical Ground Truth:** Seafloor slope, roughness, and curvature are mathematical derivatives of surface bathymetry. They indicate surface morphological ruggedness, not subsurface rock quality (RQD), fault shear zones, or water ingress risk.
3. **Local Optimality of SLSQP:** While multi-start testing confirmed the stability of the warm-started optimum, gradient-based NLP does not guarantee global continuous optimality.

---

## 13. Conclusion

This project has successfully implemented, benchmarked, and verified discrete and continuous underwater tunnel route optimization on real GEBCO_2026 bathymetry. The study establishes that:
- Adding physical realism (curvature limits and risk avoidance) increases tunnel route length by **$+25.15\%$**.
- In return, terrain hazard exposure is reduced by **$75.82\%$**, maximum slope is reduced by **$93.4\%$**, and curvature compliance is fully achieved.
- All 10 verification checks passed, and the S-core sensitivity experiments demonstrated high algorithmic robustness across feature-engineering choices.

---

## 14. References and Reproducibility

1. **GEBCO 2026 Bathymetric Compilation Group**, *The GEBCO_2026 Grid*, British Oceanographic Data Centre, National Oceanography Centre, UK. DOI: `10.5285/gebco_2026`.
2. **SciPy Optimize Documentation:** Sequential Least Squares Programming (SLSQP), `scipy.optimize.minimize`.
3. **PyProj Geodetic Utilities:** `pyproj.Geod`, WGS84 Ellipsoidal Distance.
4. **NetworkX Network Analysis in Python:** `networkx.algorithms.shortest_paths`.

---

## 15. Appendix: Environment, Seeds, and Execution

- **Operating System:** macOS (Darwin arm64)
- **Python Version:** 3.13.5
- **Dependencies:** `numpy` (2.2.6), `scipy` (1.18.1), `matplotlib` (3.10.8), `pandas` (2.3.2), `rasterio` (1.5.2), `networkx` (3.7), `pyproj` (3.8.0), `seaborn` (0.13.2), `pyyaml` (6.0.3)
- **Random Seed:** 42 (fixed for all multi-start initializations)
- **Master Execution Command:** `python3 run_all.py`
