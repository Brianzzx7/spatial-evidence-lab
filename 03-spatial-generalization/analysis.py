"""
Spatial generalization stress test on the Meuse heavy-metal dataset.

Question: how much does ordinary (random) k-fold CV overstate the accuracy of a
spatial prediction model when the samples are spatially autocorrelated?

Data: 155 topsoil samples from the Meuse floodplain (NL), zinc in ppm, plus a
3103-cell prediction grid and the river outline (R package `sp`).
CRS: Dutch RD New (EPSG:28992), metres.

Models (target = log10 zinc)
  * RF-covariates : random forest on distance-to-river, flood frequency, soil type
  * RF-coords     : the same + raw x/y coordinates (a common but risky shortcut)
  * Kriging       : ordinary kriging with a spherical variogram (pykrige)

Validation schemes
  * random 10-fold CV (x3 repeats)
  * spatial block CV: k-means clusters of coordinates as groups (x3 seeds)
  * buffered leave-one-out: drop training points within r metres of the test point

Run:  python analysis.py      (figures -> figures/, numbers -> outputs/)
"""
from pathlib import Path
import json
import tempfile
import urllib.request
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skgstat as skg
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
FIG, OUT, DATA = HERE / "figures", HERE / "outputs", HERE / "data"
for d in (FIG, OUT, DATA):
    d.mkdir(exist_ok=True)

CRS = 28992
SEEDS = [0, 1, 2]
N_TREES = 100
RADII = [0, 100, 250, 500, 750, 1000]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, RIVER = "#0b0b0b", "#8a8984", "#fcfcfb", "#b7d3f6"
COLORS = {"RF-covariates": BLUE, "RF-coords": ORANGE, "Kriging": AQUA}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e",
})


# ------------------------------------------------------------------ data
def load(name):
    """Read a dataset from the CRAN `sp` package, caching it as CSV in data/."""
    csv = DATA / f"{name}.csv"
    if not csv.exists():
        import pyreadr
        tmp = Path(tempfile.gettempdir()) / f"{name}.rda"
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/cran/sp/master/data/{name}.rda", tmp)
        df = next(iter(pyreadr.read_r(tmp).values()))
        df.to_csv(csv, index=False)
    return pd.read_csv(csv)


meuse = load("meuse")
grid = load("meuse.grid")
riv = load("meuse.riv")

pts = gpd.GeoDataFrame(meuse, geometry=gpd.points_from_xy(meuse.x, meuse.y), crs=CRS)
river = gpd.GeoSeries([Polygon(riv.iloc[:, :2].to_numpy())], crs=CRS)
y = np.log10(meuse["zinc"].to_numpy())
XY = meuse[["x", "y"]].to_numpy(float)


def features(df, with_coords):
    f = pd.DataFrame({"dist": df["dist"]})
    for c in ["ffreq", "soil"]:
        for lvl in sorted(meuse[c].unique()):
            f[f"{c}_{lvl}"] = (df[c] == lvl).astype(int)
    if with_coords:
        f["x"], f["y"] = df["x"], df["y"]
    return f.to_numpy(float)


F_cov = features(meuse, False)
F_xy = features(meuse, True)

# ------------------------------------------------------------------ variogram
vario = skg.Variogram(XY, y, model="spherical", n_lags=15, maxlag=1500)
vario_range = float(vario.parameters[0])
vario_sill, vario_nugget = float(vario.parameters[1]), float(vario.parameters[2])


# ------------------------------------------------------------------ models
def fit_predict(model, tr, te, seed=0):
    if model == "Kriging":
        ok = OrdinaryKriging(XY[tr, 0], XY[tr, 1], y[tr], variogram_model="spherical")
        pred, _ = ok.execute("points", XY[te, 0], XY[te, 1])
        return np.asarray(pred)
    F = F_cov if model == "RF-covariates" else F_xy
    rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    rf.fit(F[tr], y[tr])
    return rf.predict(F[te])


MODELS = list(COLORS)


def scores(pred, obs):
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    r2 = float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    return rmse, r2


def run_cv(splitter_fn):
    out = {m: [] for m in MODELS}
    for seed in SEEDS:
        for m in MODELS:
            pred = np.empty_like(y)
            for tr, te in splitter_fn(seed):
                pred[te] = fit_predict(m, tr, te, seed)
            out[m].append(scores(pred, y))
    return {m: np.array(v).mean(0).round(4).tolist() for m, v in out.items()}   # [RMSE, R2]


def random_splits(seed):
    return KFold(10, shuffle=True, random_state=seed).split(XY)


def block_labels(seed):
    return KMeans(10, n_init=10, random_state=seed).fit_predict(XY)


def spatial_splits(seed):
    return GroupKFold(10).split(XY, groups=block_labels(seed))


CV_CACHE = OUT / ".cv_cache.json"
if CV_CACHE.exists():
    cv_random, cv_spatial = json.loads(CV_CACHE.read_text())
else:
    cv_random, cv_spatial = run_cv(random_splits), run_cv(spatial_splits)
    CV_CACHE.write_text(json.dumps([cv_random, cv_spatial]))
print("k-fold CV done", flush=True)

# buffered leave-one-out
D = cdist(XY, XY)
# (slow: ~2,800 model fits; per-radius results are checkpointed so the run can be resumed)
CACHE = OUT / ".buffered_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for r in RADII:
    if str(r) in cache:
        continue
    cache[str(r)] = {}
    for m in MODELS:
        pred = np.empty_like(y)
        for i in range(len(y)):
            tr = np.where((D[i] > r) & (np.arange(len(y)) != i))[0]
            pred[i] = fit_predict(m, tr, np.array([i]))[0]
        cache[str(r)][m] = scores(pred, y)[0]
    CACHE.write_text(json.dumps(cache))
    print(f"buffer {r} m done", flush=True)
buffered = {m: [cache[str(r)][m] for r in RADII] for m in MODELS}

# nearest-neighbour distance: sample->sample vs prediction grid->sample
nn_samples = np.sort(D, axis=1)[:, 1]
nn_grid = cdist(grid[["x", "y"]].to_numpy(float), XY).min(1)

# ------------------------------------------------------------------ prediction map
rf_final = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf_final.fit(F_cov, y)
grid["pred_zinc_ppm"] = 10 ** rf_final.predict(features(grid, False))
grid[["x", "y", "pred_zinc_ppm"]].to_csv(OUT / "rf_prediction_grid.csv", index=False)

validation_penalty = {
    model: float(cv_spatial[model][0] / cv_random[model][0] - 1)
    for model in MODELS
}
buffered_penalty = {
    model: float(buffered[model][-1] / buffered[model][0] - 1)
    for model in MODELS
}

results = {
    "n_samples": int(len(y)),
    "crs": f"EPSG:{CRS}",
    "variogram_log10_zinc": {"model": "spherical", "range_m": round(vario_range, 1),
                             "sill": round(vario_sill, 4), "nugget": round(vario_nugget, 4)},
    "median_nn_distance_m": {"sample_to_sample": float(np.median(nn_samples)),
                             "grid_to_sample": float(np.median(nn_grid))},
    "random_10fold_[rmse,r2]": cv_random,
    "spatial_block_10fold_[rmse,r2]": cv_spatial,
    "buffered_loo_rmse": {"radii_m": RADII, **{m: np.round(v, 4).tolist() for m, v in buffered.items()}},
    "spatial_generalization_diagnostics": {
        "spatial_block_rmse_penalty": {m: round(v, 4) for m, v in validation_penalty.items()},
        "one_km_buffered_rmse_penalty": {m: round(v, 4) for m, v in buffered_penalty.items()},
    },
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# ------------------------------------------------------------------ figures
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
xmin, ymin, xmax, ymax = pts.total_bounds


def frame(ax):
    ax.set_xlim(xmin - 200, xmax + 200)
    ax.set_ylim(ymin - 280, ymax + 200)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([xmin, xmin + 500], [ymin - 120] * 2, color=INK, lw=2)
    ax.text(xmin + 250, ymin - 150, "500 m", ha="center", va="top", fontsize=8)


# Fig 1: samples, spatial blocks, variogram
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
ax = axes[0]
river.plot(ax=ax, color=RIVER, zorder=0)
sc = ax.scatter(XY[:, 0], XY[:, 1], c=meuse["zinc"], cmap=SEQ, norm=plt.matplotlib.colors.LogNorm(),
                s=28, edgecolor=INK, linewidth=0.4, zorder=2)
fig.colorbar(sc, ax=ax, shrink=0.6, label="zinc (ppm, log scale)")
ax.set_title("155 topsoil samples")
frame(ax)

ax = axes[1]
river.plot(ax=ax, color=RIVER, zorder=0)
lab = block_labels(0)
for g in range(10):
    sel = XY[lab == g]
    hull = gpd.GeoSeries(gpd.points_from_xy(sel[:, 0], sel[:, 1])).union_all().convex_hull.buffer(40)
    gpd.GeoSeries([hull]).plot(ax=ax, color="#f0efec", edgecolor=MUTED, linewidth=1, zorder=1)
    cx, cy = sel.mean(0)
    ax.text(cx, cy, str(g + 1), ha="center", va="center", fontsize=9, weight="bold", color=INK, zorder=3)
ax.scatter(XY[:, 0], XY[:, 1], s=8, color=MUTED, zorder=2)
ax.set_title("Spatial CV: 10 k-means blocks")
frame(ax)

ax = axes[2]
ax.scatter(vario.bins, vario.experimental, color=INK, s=24, zorder=3, label="empirical")
h = np.linspace(0, vario.bins.max(), 200)
ax.plot(h, vario.fitted_model(h), color=BLUE, lw=2, label="spherical fit")
ax.axvline(vario_range, color=MUTED, ls="--", lw=1)
ax.text(vario_range - 15, 0.005, f"range {vario_range:.0f} m", color=INK, fontsize=9, ha="right")
ax.set_xlabel("lag distance (m)")
ax.set_ylabel("semivariance of log10 zinc")
ax.set_title("Variogram: samples are correlated up to ~range")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig1_data_blocks_variogram.png", dpi=200)
plt.close(fig)

# Fig 2: random vs spatial CV
fig, ax = plt.subplots(figsize=(8, 4.4))
xs = np.arange(len(MODELS))
wbar = 0.36
r_vals = [cv_random[m][0] for m in MODELS]
s_vals = [cv_spatial[m][0] for m in MODELS]
b1 = ax.bar(xs - wbar / 2 - 0.01, r_vals, wbar, color=BLUE, label="random 10-fold")
b2 = ax.bar(xs + wbar / 2 + 0.01, s_vals, wbar, color=ORANGE, label="spatial block 10-fold")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}",
                ha="center", fontsize=9, color=INK)
ax.set_xticks(xs, MODELS)
ax.set_ylabel("RMSE (log10 zinc), lower is better")
ax.set_title("Random CV looks better than it is")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(s_vals + r_vals) * 1.18)
fig.tight_layout()
fig.savefig(FIG / "fig2_random_vs_spatial_cv.png", dpi=200)
plt.close(fig)

# Fig 3: buffered LOO
fig, ax = plt.subplots(figsize=(8, 4.6))
for m in MODELS:
    ax.plot(RADII, buffered[m], color=COLORS[m], lw=2, marker="o", ms=5, label=m)
    ax.text(RADII[-1] + 20, buffered[m][-1], m, color=INK, va="center", fontsize=9)
ax.axvline(vario_range, color=MUTED, ls="--", lw=1)
ax.text(vario_range + 10, ax.get_ylim()[0] + 0.01, "variogram range", color="#52514e", fontsize=8)
ax.axvline(np.median(nn_grid), color=MUTED, ls=":", lw=1)
ax.text(np.median(nn_grid) + 10, ax.get_ylim()[1] - 0.005, "typical distance from a grid cell\nto its nearest sample",
        color="#52514e", fontsize=8, va="top")
ax.set_xlabel("exclusion buffer around each test point (m)")
ax.set_ylabel("RMSE (log10 zinc)")
ax.set_title("Buffered leave-one-out: error grows as training data moves away")
ax.set_xlim(-30, RADII[-1] + 230)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig3_buffered_loo.png", dpi=200)
plt.close(fig)

# Fig 4: prediction map
fig, ax = plt.subplots(figsize=(6.2, 7))
river.plot(ax=ax, color=RIVER, zorder=0)
cell = 40.0
gx0, gy0 = grid.x.min(), grid.y.min()
nx_, ny_ = int(round((grid.x.max() - gx0) / cell)) + 1, int(round((grid.y.max() - gy0) / cell)) + 1
raster = np.full((ny_, nx_), np.nan)
raster[((grid.y - gy0) / cell).round().astype(int), ((grid.x - gx0) / cell).round().astype(int)] = grid.pred_zinc_ppm
sc = ax.imshow(raster, origin="lower", cmap=SEQ, norm=plt.matplotlib.colors.LogNorm(), zorder=1,
               extent=(gx0 - cell / 2, gx0 + (nx_ - 0.5) * cell, gy0 - cell / 2, gy0 + (ny_ - 0.5) * cell))
ax.scatter(XY[:, 0], XY[:, 1], s=6, color=INK, zorder=2, label="samples")
fig.colorbar(sc, ax=ax, shrink=0.6, label="predicted zinc (ppm)")
ax.set_title("RF-covariates prediction on 40 m grid")
ax.legend(frameon=False, loc="upper left", fontsize=8)
frame(ax)
fig.tight_layout()
fig.savefig(FIG / "fig4_prediction_map.png", dpi=200)
plt.close(fig)
