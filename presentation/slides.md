# Underwater Tunnel Route Optimization: Balancing Feasibility, Length, Terrain Risk, and Curvature

**A Comparative Study in the Hudson Canyon Continental Margin (GEBCO_2026)**

---

## Slide 1: Title & Overview (0:20)
* **Title:** Underwater Tunnel Route Optimization: Feasibility, Terrain Risk, and the Price of Realism
* **Presenters:** Optimization Research Group
* **Course:** Optimization & Numerical Methods
* **Key Idea:** We solve two route optimization problems on identical real-world GEBCO bathymetry:
  1. **Easy Problem:** Shortest feasible discrete path on an 8-neighbor grid graph via Dijkstra.
  2. **Hard Problem:** Smooth continuous alignment minimizing length and terrain risk via SciPy SLSQP.
* **Core Takeaway:** *The shortest feasible tunnel is not always the best tunnel.* Real engineering requires curvature continuity, risk penalization, and smooth alignment.

---

## Slide 2: Problem & Motivation (1:00)
* **The Engineering Challenge:** Subsea tunnels (e.g., Channel Tunnel, Seikan, Ryfast) face steep slopes, submarine trenches, and structural curvature constraints.
* **Research Question:** How do seafloor terrain characteristics and feature-engineering choices change route feasibility, length, terrain risk, and robustness?
* **What This Project Is and Is Not:**
  * **Is:** A rigorous 2D route-planning and mathematical optimization benchmark evaluating discrete vs. continuous formulations.
  * **Is Not:** A geotechnical soil stability or structural borehole study. Seafloor depth, slope, roughness, and curvature serve as mathematical proxies for terrain suitability.

---

## Slide 3: Dataset, Projected Coordinates, & Physical Units (1:00)
* **Dataset:** GEBCO_2026 Global Bathymetric Grid (15 arc-second native resolution) from NERC EDS British Oceanographic Data Centre (BODC).
* **Region:** Hudson Canyon Continental Margin ($49.5\text{ km} \times 49.5\text{ km}$, $111 \times 111$ cells).
  * Features flat continental shelf (50–100 m) cut by a deep submarine canyon gorge (>1300 m).
  * 100% marine water coverage (no land masking artifacts or edge interpolator leaks).
* **Metric Reprojection:** Reprojected from WGS84 geographic coordinates to **UTM Zone 18N (EPSG:32618)** with regular metric spacing $\Delta x = \Delta y = 450.0\text{ m}$.
* **Geodesic Validation:** Straight-line distance $A \to B$ in UTM grid is $42,452.92\text{ m}$; WGS84 geodesic distance is $42,444.46\text{ m}$ (Difference: $+8.46\text{ m}$, **0.0199% error**).
* **Depth Interpretation:** $d(x, y) = -z(x, y)$ represents **seafloor depth**, serving as a terrain clearance proxy—not the tunnel subterranean overburden depth.

---

## Slide 4: Data Analysis 1: Maps and Distributions (Visual Slide 1) (1:30)
*Refer to Figure 1 (Bathymetry Overview) and Figure 2 (Feature Spatial Maps).*
* **Depth ($d$):** Mean 144.2 m, Median 96.9 m, Max 1312.6 m. Heavy right-skew (skewness = 3.90).
* **Slope ($s = |\nabla z|$):** Derived via central differences in metres. Shelf slope < 0.3° ($s < 0.005$); canyon walls reach 20.5° ($s = 0.373$). Extreme skewness = 3.12.
* **Roughness ($r$):** Local standard deviation of slope ($w = 3$). Flat shelf roughness $\approx 0.001$, canyon break roughness reaches 0.104.
* **Curvature ($c = |\nabla^2 z|$):** 5-point Laplacian stencil. Sharpest inflection at shelf break ($c_{\max} = 0.0011\text{ m}^{-1}$).
* **Composite Risk Field $R(x, y)$:**
  $$R(x, y) = 0.5 \cdot \tilde{s}^2 + 0.3 \cdot \tilde{r} + 0.2 \cdot \tilde{c}$$
  Squaring normalized slope heavily penalizes steep walls while leaving the continental shelf near zero risk ($R < 0.02$).

---

## Slide 5: Data Analysis 2: Correlations, Spatial Autocorrelation, & PCA (Visual Slide 2) (1:30)
*Refer to Figure 3 (Distributions & Correlations) and Figure 4 (Moran's I & TID).*
* **Correlation Matrices:**
  * Depth vs Slope: Pearson $r = +0.776$, Spearman $\rho = +0.741$.
  * Slope vs Roughness: Pearson $r = +0.941$, Spearman $\rho = +0.923$ (high redundancy between steepness and local variance).
  * Curvature vs Slope: Pearson $r = +0.551$, Spearman $\rho = +0.638$.
* **Spatial Autocorrelation (Moran's $I$):**
  * Depth: $I = 0.974$; Slope: $I = 0.940$; Risk: $I = 0.959$.
  * Neighboring cells exhibit extreme spatial clustering ($p \ll 0.001$). Naive p-values are invalid; correlations reflect spatial co-occurrence, not causation.
  * Spatial thinning check (every 4th cell) confirms correlation stability ($r_{s,r} = +0.932$).
* **PCA Dimensionality Reduction:** PC1 captures **84.3%** of total variance with positive loadings across all three features, isolating the canyon corridor as the dominant risk feature.
* **Data Quality (TID Layer):** 28.8% surveyed by high-precision multibeam sonar (TID 11); 71.2% interpolated from pre-generated grid (TID 17).

---

## Slide 6: Easy Formulation — Discrete Shortest Feasible Route (1:00)
* **Mathematical Formulation:**
  $$\min_{P \in \mathcal{P}} L(P) = \sum_{e \in P} \|e\|$$
  $$\text{subject to } s(v) \le s_{\max} = 0.08, \quad d_{\min} = 50\text{ m} \le d(v) \le d_{\max} = 400\text{ m}$$
* **Graph Model:** 8-neighbor grid graph; orthogonal edges $450\text{ m}$, diagonal edges $450\sqrt{2}\approx 636.4\text{ m}$.
* **Solver:** Dijkstra algorithm (nonnegative Euclidean edge weights).
* **Optimality Scope:** **Global discrete optimum** on the graph. It is globally minimal in distance over graph moves, but lacks physical smoothness.
* **Results:**
  * Route Length $L_E = 47,955.84\text{ m}$ (47.96 km).
  * Cumulative Terrain Risk $S_E = 493.12$.
  * Runtime: $23.2\text{ ms}$; Nodes Expanded: 9,439.
  * **Fatal Limitation:** Discrete turns reach $45^\circ$. Fitted curvature $\kappa = 0.004260\text{ m}^{-1}$ (Radius $R = 235\text{ m}$), violently violating tunnel boring machine limits!

---

## Slide 7: Hard Formulation — Continuous Smooth Spline Alignment (1:30)
* **Mathematical Formulation:**
  $$\min_{\mathbf{p}} J = \alpha \cdot L(p) + \beta \cdot S(p) = \int_0^1 \|p'(t)\| dt + 50000 \int_0^1 R(p(t)) \|p'(t)\| dt$$
  $$\text{subject to } d_{\min} \le d(p(t)) \le d_{\max}, \quad s(p(t)) \le s_{\max}, \quad |\kappa_{\text{route}}(t)| \le \kappa_{\max} = 0.0005\text{ m}^{-1}$$
* **Parameterization:** $C^2$ natural cubic spline with $K=7$ interior control points ($2K=14$ decision variables). Endpoints fixed at $p(0) = A$, $p(1) = B$.
* **Field Interpolation:** Bicubic `RectBivariateSpline` on $d, s, R$. Overshoot strictly bounded (<0.26%).
* **Numerical Integration:** Discretized at $M=200$ points ($M_{\text{dense}}=2000$ points for validation).
* **Multi-Start Strategy:** Warm start (Dijkstra fit), straight line, plus 8 random sinusoidal perturbations.
* **Optimality Scope:** **Local continuous optimum** on a non-convex manifold via SLSQP.

---

## Slide 8: Optimal Route Alignments on the Risk Field (1:30)
*Refer to Figure 5 (Easy vs Hard Routes Overlay on Risk Map).*
* **Straight Line Corridor (White):** 42.45 km. Crosses the canyon head directly; intersects steep 10.7° slopes and deep water (389 m), incurring fatal feasibility violations.
* **Easy Route (Cyan):** 47.96 km. Hugs the shelf edge, making rapid, jagged 45° turns along the boundary of the feasible corridor. Accumulates high cumulative risk ($S_E = 493.12$).
* **Hard Route (Green):** 60.02 km. Bends gracefully northward onto the flat continental shelf, completely bypassing the high-risk canyon incision before descending smoothly onto the eastern terrace.
* **Engineering Geometry:** The continuous spline creates an alignment suitable for rail/highway tunneling with minimum turn radius $R_{\min} \ge 2,053\text{ m}$ ($\kappa_{\max} = 0.000487\text{ m}^{-1}$).

---

## Slide 9: Comparison Table & The Price of Realism (1:30)
| Metric | Unit | Easy Route (Dijkstra) | Hard Route (SLSQP Spline) | Difference / Impact |
|---|---|---|---|---|
| **Route Length $L$** | metres | 47,955.84 m | 60,017.94 m | **+12,062.10 m (+25.15%)** |
| **Terrain Risk $S$** | dimensionless | 493.1159 | 119.2297 | **-373.8862 (-75.82%)** |
| **Objective $J$** | weighted | 24,703,748.82 | 6,021,503.03 | -18,682,245.79 |
| **Max Curvature $\kappa$** | $\text{m}^{-1}$ (Limit: 0.0005) | 0.004260 | 0.000487 | **Easy Violates (x8.5); Hard Feasible** |
| **Max Slope $s$** | $\text{m/m}$ (Limit: 0.08) | 0.0739 | 0.0049 | -0.0690 (Hard is 15x flatter) |
| **Depth Range $d$** | metres ([50, 400]) | [73.1, 191.2] m | [73.3, 118.5] m | Both strictly within limits |
| **Runtime** | milliseconds | 23.24 ms | 1,701.79 ms | Fast graph search vs. iterative NLP |
| **Optimality** | theoretical | Global Discrete | Local Continuous | Graph optimum vs. Spline optimum |

* **Price of Realism = +25.15%:** Adding engineering realism (route smoothness and risk minimization) lengthens the tunnel by 12.06 km.
* **Risk Reduction = 75.82%:** In return, cumulative terrain risk drops by over three-quarters, and slope exposure drops by 93%.

---

## Slide 10: S-core Sensitivity Analysis (1:30)
*Refer to Figure 8 (S-core Sensitivity Overlays and Robustness Metrics).*
* **1. Slope Method (Central vs. Horn):** Horn 3x3 smoothing slightly lowers peak slope. $\Delta L = -0.32\%$, $\Delta S = -2.14\%$. Route trajectory is highly stable ($d_H = 184\text{ m}$).
* **2. Roughness Window ($w=3 \to 7$):** Wider window broadens risk boundaries. $\Delta L = +1.18\%$, $\Delta S = +6.42\%$. Route detours slightly farther north ($d_H = 432\text{ m}$).
* **3. Smoothing ($\sigma=0 \to 2$):** Pre-smoothing eliminates micro-topographic noise. $\Delta L = -0.84\%$, $\Delta S = -14.6\%$. Route remains robust ($d_H = 310\text{ m}$).
* **4. Slope Limit ($s_{\max} = 0.08 \to \text{P75}=0.0055$):** Strictest test! Eliminates 90% of the shelf slope. The easy path is forced into an extreme northward detour ($\Delta L_E = +18.4\%$), whereas the hard spline smoothly adjusts.
* **5. Curvature Limit ($\kappa_{\max} = 0.0005 \to 0.000333$ / $R_{\min}=3000\text{ m}$):** Flattens turns further; $\Delta L = +4.12\%$.

---

## Slide 11: Verification & Optimality Types (1:00)
*Refer to Figure 10 (Verification Toy Problems).*
* **All 10 Compulsory Verification Checks Passed (100%):**
  1. *Endpoints:* $A$ and $B$ match within $10^{-4}\text{ m}$.
  2. *Recomputation:* Independent code verifies $L, S, J$ within floating-point tolerance.
  3. *Toy 1 (Easy):* Dijkstra matches exact analytical detour around grid wall ($113.14\text{ m}$, err < $10^{-6}\text{ m}$).
  4. *Toy 2 (Hard):* Flat zero-risk test returns analytical straight line ($1414.21\text{ m}$, $\kappa = 0$, err < $0.05\text{ m}$).
  5. *Toy 3 (Hard):* Centered Gaussian bump produces perfectly symmetric detour ($|y(t)| = |y(1-t)|$, asym < $10^{-4}\text{ m}$).
  6. *NetworkX:* Cross-check confirms zero difference ($0.0\text{ m}$) against library shortest path.
  7. *Dense Validation:* Density study at 200, 1000, 5000 points proves zero constraint violation growth.
  8. *Interpolator:* Overshoot strictly bounded below 0.26%.
  9. *Multi-start:* Evaluated across 10 distinct initializations.
  10. *Geodesic:* Metric UTM distance validated against WGS84 geodesic (<0.02% diff).

---

## Slide 12: Conclusion & Engineering Limitations (0:30)
* **Key Takeaway:** The "Price of Realism" (+25.15% length) is an essential trade-off to transform an unbuildable discrete path (with 45° kinks and extreme risk) into a feasible, smooth continuous tunnel alignment.
* **Pareto Frontier:** Sweeping $\beta$ exposes the direct trade-off between tunnel excavation distance and seafloor hazard exposure.
* **Limitations:**
  * 2D proxy modeling: does not model vertical tunnel profile or 3D subterranean geology.
  * Mathematical terrain proxies (slope, roughness, curvature) are indicators of seafloor morphology, not geotechnical borehole strength measurements.
* **Recommendation:** Continuous spline optimization initialized from discrete graph warm-starts provides an optimal hybrid paradigm for infrastructure routing.
