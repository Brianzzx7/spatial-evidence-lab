"""
Multiscale education diagnostics on the Georgia county dataset.

Model:  PctBach ~ PctRural + PctPov + PctBlack      (159 counties, 1990 census)

Steps
  1. Load polygons (shapefile ships WITHOUT a .prj) and infer the CRS by testing
     candidate UTM zones against the latitude/longitude columns.
  2. Global OLS + Moran's I on the residuals (queen contiguity) -> is there
     spatial structure left over?
  3. GWR (adaptive bisquare, AICc golden-section search) and check the result
     against the reference output of the GWR4 software shipped with the data.
  4. MGWR on standardised variables: one bandwidth per covariate.
  5. Maps of local coefficients, masked where not significant
     (multiple-testing corrected t-values).

Run:  python analysis.py      (figures -> figures/, numbers -> outputs/)
"""
from pathlib import Path
import json
import os

import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

HERE = Path(__file__).resolve().parent
FIG, OUT = HERE / "figures", HERE / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

Y_VAR = "PctBach"
X_VARS = ["PctRural", "PctPov", "PctBlack"]
NAMES = ["Intercept"] + X_VARS

BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"
INK, MUTED, SURFACE = "#0b0b0b", "#8a8984", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#86b6ef", MID, "#f0a3a2", RED])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e",
})

# ------------------------------------------------------------------ data + CRS
data_dir = Path(os.path.dirname(libpysal.__file__)) / "examples" / "georgia"
gdf = gpd.read_file(data_dir / "G_utm.shp")
assert gdf.crs is None, "expected a shapefile without .prj"

crs_check = {}
for epsg in (26916, 26917):                       # NAD83 / UTM 16N and 17N
    ll = gpd.points_from_xy(gdf.X, gdf.Y, crs=epsg).to_crs(4326)
    crs_check[epsg] = float(np.median(np.hypot(ll.x - gdf.Longitud, ll.y - gdf.Latitude)))
EPSG = min(crs_check, key=crs_check.get)
gdf = gdf.set_crs(EPSG)

y = gdf[[Y_VAR]].to_numpy(float)
X = gdf[X_VARS].to_numpy(float)
coords = list(zip(gdf.X, gdf.Y))
n = len(gdf)

# ------------------------------------------------------------------ global OLS
Xc = np.hstack([np.ones((n, 1)), X])
beta_ols, *_ = np.linalg.lstsq(Xc, y, rcond=None)
res_ols = (y - Xc @ beta_ols).ravel()
rss = float(res_ols @ res_ols)
k = Xc.shape[1]
sigma2_ml = rss / n
llf = -n / 2 * (np.log(2 * np.pi * sigma2_ml) + 1)
aicc_ols = -2 * llf + 2 * (k + 1) + 2 * (k + 1) * (k + 2) / (n - k - 2)   # GWR4 convention adds sigma
r2_ols = 1 - rss / float(((y - y.mean()) ** 2).sum())

W = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
W.transform = "r"


def moran(resid):
    m = Moran(np.asarray(resid).ravel(), W, permutations=999)
    return round(float(m.I), 4), round(float(m.p_sim), 4)


# ------------------------------------------------------------------ GWR
gwr_selector = Sel_BW(coords, y, X, kernel="bisquare", fixed=False)
gwr_bw = gwr_selector.search(criterion="AICc")
gwr = GWR(coords, y, X, gwr_bw, kernel="bisquare", fixed=False).fit()

# check against the GWR4 reference run shipped with the data (same model spec)
ref = pd.read_csv(data_dir / "georgia_BS_NN_listwise.csv")
ref.columns = [c.strip() for c in ref.columns]
gwr4 = {"bandwidth": 90.398227, "AICc": 896.462831, "R2": 0.592415}   # from georgia_BS_NN_summary.txt
coef_corr = {nm: float(np.corrcoef(gwr.params[:, j], ref[f"est_{nm}"])[0, 1]) for j, nm in enumerate(NAMES)}

# ------------------------------------------------------------------ MGWR
ys = (y - y.mean()) / y.std()
Xs = (X - X.mean(0)) / X.std(0)
mgwr_selector = Sel_BW(coords, ys, Xs, multi=True, kernel="bisquare", fixed=False)
mgwr_bws = mgwr_selector.search(multi_bw_min=[2])
mgwr = MGWR(coords, ys, Xs, mgwr_selector, kernel="bisquare", fixed=False).fit()
mgwr_t = mgwr.filter_tvals()                        # zeroed where not significant (corrected alpha)

gwr_std_bw = Sel_BW(coords, ys, Xs, kernel="bisquare", fixed=False).search(criterion="AICc")
gwr_std = GWR(coords, ys, Xs, gwr_std_bw, kernel="bisquare", fixed=False).fit()

# Diagnostics expressed on the original response scale make the model-selection
# story easier to audit alongside the local coefficient maps.
gwr_aicc_improvement = float(aicc_ols - gwr.aicc)
ols_moran_i = float(Moran(np.asarray(res_ols).ravel(), W, permutations=0).I)
gwr_moran_i = float(Moran(np.asarray(gwr.resid_response).ravel(), W, permutations=0).I)
residual_dependence_reduction = 1 - abs(gwr_moran_i) / abs(ols_moran_i)
bandwidth_spread = float(max(mgwr_bws) - min(mgwr_bws))

results = {
    "n_counties": n,
    "inferred_crs": f"EPSG:{EPSG}",
    "crs_check_median_error_deg": {f"EPSG:{k_}": round(v, 4) for k_, v in crs_check.items()},
    "ols": {"R2": round(r2_ols, 4), "AICc": round(float(aicc_ols), 2),
            "coefficients": dict(zip(NAMES, np.round(beta_ols.ravel(), 4).tolist())),
            "residual_moran_I_p": moran(res_ols)},
    "gwr": {"bandwidth_neighbours": float(gwr_bw), "R2": round(float(gwr.R2), 4),
            "AICc": round(float(gwr.aicc), 2), "residual_moran_I_p": moran(gwr.resid_response)},
    "gwr4_reference": gwr4,
    "gwr_vs_gwr4_coefficient_correlation": {k_: round(v, 4) for k_, v in coef_corr.items()},
    "mgwr_standardised": {"bandwidths": dict(zip(NAMES, [float(b) for b in mgwr_bws])),
                          "R2": round(float(mgwr.R2), 4), "AICc": round(float(mgwr.aicc), 2),
                          "residual_moran_I_p": moran(mgwr.resid_response),
                          "share_significant": dict(zip(NAMES, np.round((mgwr_t != 0).mean(0), 3).tolist()))},
    "gwr_standardised": {"bandwidth": float(gwr_std_bw), "AICc": round(float(gwr_std.aicc), 2)},
    "spatial_scale_diagnostics": {
        "gwr_aicc_improvement_over_ols": round(gwr_aicc_improvement, 2),
        "residual_moran_abs_reduction": round(float(residual_dependence_reduction), 4),
        "mgwr_bandwidth_spread_neighbours": round(bandwidth_spread, 1),
    },
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))

coef_table = gdf[["AreaKey"]].copy()
for j, nm in enumerate(NAMES):
    coef_table[f"gwr_{nm}"] = gwr.params[:, j]
    coef_table[f"mgwr_{nm}"] = mgwr.params[:, j]
    coef_table[f"mgwr_t_{nm}"] = mgwr.tvalues[:, j]
coef_table.to_csv(OUT / "local_coefficients.csv", index=False)
print(json.dumps(results, indent=2))

# ------------------------------------------------------------------ figures
def clean(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


# Fig 1: dependent variable + OLS residuals
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
gdf.plot(column=Y_VAR, cmap=SEQ, edgecolor=SURFACE, linewidth=0.4, ax=axes[0], legend=True,
         legend_kwds={"shrink": 0.6, "label": "% with bachelor's degree"})
axes[0].set_title("Outcome: PctBach (1990)")
lim = np.abs(res_ols).max()
gdf.assign(r=res_ols).plot(column="r", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), edgecolor=SURFACE,
                           linewidth=0.4, ax=axes[1], legend=True,
                           legend_kwds={"shrink": 0.6, "label": "OLS residual (pct points)"})
I, p = results["ols"]["residual_moran_I_p"]
axes[1].set_title(f"Global OLS residuals: Moran's I = {I:.2f} (p = {p:.3f})")
for ax in axes:
    clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig1_outcome_and_ols_residuals.png", dpi=200)
plt.close(fig)

# Fig 2: MGWR local coefficients, non-significant counties greyed out
fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
for j, (ax, nm) in enumerate(zip(axes, NAMES)):
    vals = mgwr.params[:, j]
    lim = max(np.abs(vals).max(), 1e-6)
    sig = mgwr_t[:, j] != 0
    g = gdf.assign(v=vals)
    if sig.any():
        g[sig].plot(column="v", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), edgecolor=SURFACE, linewidth=0.4, ax=ax)
    if (~sig).any():
        g[~sig].plot(color="#e2e1dc", edgecolor=SURFACE, linewidth=0.4, hatch="////", ax=ax)
    sm = plt.cm.ScalarMappable(cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, orientation="horizontal", pad=0.02)
    cb.locator = MaxNLocator(5)
    cb.update_ticks()
    ax.set_title(f"{nm}\nbandwidth = {int(mgwr_bws[j])} neighbours")
    clean(ax)
fig.suptitle("MGWR local coefficients (standardised). Hatched grey = not significant after correction",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "fig2_mgwr_coefficients.png", dpi=200)
plt.close(fig)

# Fig 3: replication check against GWR4
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for j, (ax, nm) in enumerate(zip(axes, NAMES)):
    ax.scatter(ref[f"est_{nm}"], gwr.params[:, j], s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.set_title(f"{nm}  (r = {coef_corr[nm]:.3f})")
    ax.set_xlabel("GWR4 (reference)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("This script (mgwr)")
fig.suptitle(f"Replication check: bandwidth {gwr_bw:.0f} vs GWR4 {gwr4['bandwidth']:.1f}, "
             f"AICc {gwr.aicc:.1f} vs {gwr4['AICc']:.1f}", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig3_gwr4_replication_check.png", dpi=200)
plt.close(fig)
