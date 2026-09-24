"""
Network exposure audit: Euclidean versus street-network accessibility.

Steps
  1. Load deaths, pumps and streets (shipped with libpysal) and reproject from
     Web Mercator (EPSG:3857) to British National Grid (EPSG:27700).
  2. Repair the street network topology (snap T-junctions, node intersections)
     and build a weighted graph with networkx.
  3. Assign every death location to its nearest pump by straight-line distance
     (Thiessen / Voronoi polygons) and by walking distance along the streets.
  4. Monte-Carlo test: are deaths closer to the Broad Street pump than
     households placed at random on the street network would be?
  5. Weighted kernel density estimate (KDE) of deaths.

Run:  python analysis.py      (figures -> figures/, numbers -> outputs/)
"""
from pathlib import Path
import json
import os

import geopandas as gpd
import libpysal
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import snap, split, unary_union, voronoi_diagram
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
FIG, OUT = HERE / "figures", HERE / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

BNG = 27700                     # British National Grid, metres
RNG = np.random.default_rng(1854)
N_SIM = 999

# palette (validated reference palette: blue for the focal pump, neutral ink for the rest)
BLUE, BLUE_LIGHT = "#2a78d6", "#cde2fb"
INK, MUTED, STREET, SURFACE = "#0b0b0b", "#8a8984", "#c9c8c2", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
})

# --------------------------------------------------------------------------- data
data_dir = Path(os.path.dirname(libpysal.__file__)) / "examples" / "snow_maps"
deaths_wm = gpd.read_file(data_dir / "SohoPeople.shp")
pumps_wm = gpd.read_file(data_dir / "SohoWater.shp")
streets_wm = gpd.read_file(data_dir / "Soho_Network.shp")
assert deaths_wm.crs.to_epsg() == 3857

deaths = deaths_wm.to_crs(BNG)
pumps = pumps_wm.to_crs(BNG).reset_index(drop=True)
streets = streets_wm.to_crs(BNG)

# the dataset has no pump names: identify Broad Street by its known location
broad_st_ll = gpd.GeoSeries([Point(-0.136668, 51.513341)], crs=4326).to_crs(BNG).iloc[0]
BROAD = int(pumps.distance(broad_st_ll).idxmin())
pumps["name"] = ["pump " + str(i) for i in pumps.index]
pumps.loc[BROAD, "name"] = "Broad Street"

deaths = deaths[deaths["Count"] > 0].reset_index(drop=True)   # locations with recorded deaths
w = deaths["Count"].to_numpy(float)

# why reprojecting matters: Web Mercator stretches distances by 1/cos(latitude)
bwm = pumps_wm.geometry.iloc[BROAD]
d_wm = deaths_wm[deaths_wm["Count"] > 0].distance(bwm).mean()
d_bng = deaths.distance(pumps.geometry.iloc[BROAD]).mean()
mercator_ratio = d_wm / d_bng

# --------------------------------------------------------------- network topology
ends = MultiPoint([p for ln in streets.geometry for p in (ln.coords[0], ln.coords[-1])])
snapped = [snap(ln, ends, 0.5) for ln in streets.geometry]      # close T-junction gaps
noded = unary_union(snapped)                                     # split at intersections


def key(xy):
    return (round(xy[0], 1), round(xy[1], 1))


def insert_points(lines, pts):
    """Split street segments at pump locations so each pump becomes a graph node."""
    lines = list(lines)
    for p in pts:
        tree = STRtree(lines)
        i = tree.nearest(p)
        ln = lines[i]
        q = ln.interpolate(ln.project(p))
        if q.distance(Point(ln.coords[0])) < 0.2 or q.distance(Point(ln.coords[-1])) < 0.2:
            continue
        parts = split(snap(ln, q, 1e-6), q)
        lines[i:i + 1] = list(parts.geoms)
    return lines


edges = insert_points(noded.geoms, pumps.geometry)
G = nx.Graph()
for ln in edges:
    G.add_edge(key(ln.coords[0]), key(ln.coords[-1]), length=ln.length, geom=ln)
assert nx.is_connected(G), "street network should be a single component"

edge_list = list(G.edges(data=True))
edge_geoms = [e[2]["geom"] for e in edge_list]
edge_tree = STRtree(edge_geoms)


def pump_node(p):
    i = edge_tree.nearest(p)
    u, v, dat = edge_list[i]
    return u if Point(u).distance(p) <= Point(v).distance(p) else v


pump_nodes = [pump_node(p) for p in pumps.geometry]
pump_access = np.array([Point(n).distance(p) for n, p in zip(pump_nodes, pumps.geometry)])
dist_from_pump = [nx.single_source_dijkstra_path_length(G, n, weight="length") for n in pump_nodes]


def network_dist(points):
    """(n_points, n_pumps) walking distance: point -> nearest street -> pump."""
    out = np.empty((len(points), len(pumps)))
    for r, p in enumerate(points):
        i = edge_tree.nearest(p)
        u, v, dat = edge_list[i]
        ln = dat["geom"]
        t = ln.project(p)
        # orient offset t relative to u
        if Point(ln.coords[0]).distance(Point(u)) > 0.2:
            t = ln.length - t
        access = ln.distance(p)
        for k in range(len(pumps)):
            out[r, k] = access + min(t + dist_from_pump[k][u], ln.length - t + dist_from_pump[k][v]) + pump_access[k]
    return out


def euclid_dist(points):
    xy = np.array([[p.x, p.y] for p in points])
    pxy = np.array([[p.x, p.y] for p in pumps.geometry])
    return np.linalg.norm(xy[:, None, :] - pxy[None, :, :], axis=2)


D_net = network_dist(deaths.geometry)
D_euc = euclid_dist(deaths.geometry)
deaths["pump_net"] = D_net.argmin(1)
deaths["pump_euc"] = D_euc.argmin(1)
deaths["dist_broad_net"] = D_net[:, BROAD]
deaths["dist_broad_euc"] = D_euc[:, BROAD]

share_net = w[deaths.pump_net == BROAD].sum() / w.sum()
share_euc = w[deaths.pump_euc == BROAD].sum() / w.sum()
reassigned = int(w[deaths.pump_net != deaths.pump_euc].sum())

# ----------------------------------------------------------- Monte-Carlo test
lengths = np.array([g.length for g in edge_geoms])


def random_points_on_network(n):
    idx = RNG.choice(len(edge_geoms), size=n, p=lengths / lengths.sum())
    return [edge_geoms[i].interpolate(RNG.uniform(0, edge_geoms[i].length)) for i in idx]


sim_share = np.empty(N_SIM)
sim_mean_dist = np.empty(N_SIM)
for s in range(N_SIM):
    pts = random_points_on_network(len(deaths))
    Ds = network_dist(pts)
    sim_share[s] = w[Ds.argmin(1) == BROAD].sum() / w.sum()
    sim_mean_dist[s] = np.average(Ds[:, BROAD], weights=w)
obs_mean_dist = np.average(D_net[:, BROAD], weights=w)
p_share = (1 + (sim_share >= share_net).sum()) / (N_SIM + 1)
p_dist = (1 + (sim_mean_dist <= obs_mean_dist).sum()) / (N_SIM + 1)

# ----------------------------------------------------------------- polygons
xmin, ymin, xmax, ymax = unary_union([streets.union_all(), MultiPoint(list(pumps.geometry))]).bounds
study = box(xmin - 20, ymin - 20, xmax + 20, ymax + 20)
vor = voronoi_diagram(MultiPoint(list(pumps.geometry)), envelope=study)
vor = gpd.GeoDataFrame(geometry=[g.intersection(study) for g in vor.geoms], crs=BNG)
vor["pump"] = [int(pumps.distance(g.representative_point()).idxmin()) for g in vor.geometry]

seg = gpd.GeoDataFrame(geometry=edge_geoms, crs=BNG)
seg["pump"] = network_dist([g.interpolate(0.5, normalized=True) for g in edge_geoms]).argmin(1)

# ---------------------------------------------------------------------- KDE
xy = np.vstack([deaths.geometry.x, deaths.geometry.y])
kde = gaussian_kde(xy, weights=w, bw_method=0.25)
gx, gy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))
dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
peak = (gx.ravel()[dens.argmax()], gy.ravel()[dens.argmax()])
peak_to_broad = Point(peak).distance(pumps.geometry.iloc[BROAD])

# ------------------------------------------------------------------ figures
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def base(ax):
    streets.plot(ax=ax, color=STREET, linewidth=1.2, zorder=1)
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(ymin - 20, ymax + 20)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_pumps(ax):
    others = pumps.drop(index=BROAD)
    others.plot(ax=ax, marker="^", color=SURFACE, edgecolor=INK, markersize=55, linewidth=1.2, zorder=5)
    pumps.loc[[BROAD]].plot(ax=ax, marker="^", color=BLUE, edgecolor=INK, markersize=120, linewidth=1.2, zorder=6)
    b = pumps.geometry.iloc[BROAD]
    ax.annotate("Broad Street pump", (b.x, b.y), xytext=(-40, 40), textcoords="offset points",
                fontsize=9, color=INK, weight="bold", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
                path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])


def scalebar(ax, length=100):
    x0, y0 = xmin + 10, ymin - 5
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=2)
    ax.text(x0 + length / 2, y0 + 12, f"{length} m", ha="center", fontsize=8, color=INK)


# Fig 1: overview + KDE
fig, ax = plt.subplots(figsize=(8, 7.4))
base(ax)
ax.contourf(gx, gy, dens, levels=np.linspace(dens.max() * 0.08, dens.max(), 8), cmap=SEQ, alpha=0.9, zorder=0,
            extend="neither")
ax.scatter(deaths.geometry.x, deaths.geometry.y, s=6 + 10 * w, color=INK, alpha=0.75,
           edgecolor=SURFACE, linewidth=0.5, zorder=4, label="deaths (size = count)")
draw_pumps(ax)
scalebar(ax)
ax.set_title("Cholera deaths, Soho 1854: weighted kernel density")
ax.legend(loc="upper right", frameon=False, fontsize=8)
fig.text(0.02, 0.02, f"EPSG:27700. {int(w.sum())} deaths at {len(deaths)} addresses. "
         f"KDE peak lies {peak_to_broad:.0f} m from the Broad Street pump.", fontsize=8, color="#52514e")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIG / "fig1_kde_overview.png", dpi=200)
plt.close(fig)

# Fig 2: Voronoi vs network service areas
fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
for ax, mode in zip(axes, ["euc", "net"]):
    base(ax)
    if mode == "euc":
        vor.plot(ax=ax, color=[BLUE_LIGHT if p == BROAD else "#f0efec" for p in vor.pump],
                 edgecolor=MUTED, linewidth=0.8, zorder=0)
        ax.set_title(f"Straight-line (Voronoi): {share_euc:.0%} of deaths nearest Broad St")
    else:
        seg[seg.pump != BROAD].plot(ax=ax, color=MUTED, linewidth=1.6, zorder=2)
        seg[seg.pump == BROAD].plot(ax=ax, color=BLUE, linewidth=2.6, zorder=3)
        ax.set_title(f"Walking distance (network): {share_net:.0%} nearest Broad St")
    col = deaths[f"pump_{mode}"] == BROAD
    ax.scatter(deaths.geometry.x[~col], deaths.geometry.y[~col], s=6 + 8 * w[~col], color=MUTED,
               edgecolor=SURFACE, linewidth=0.5, zorder=4, label="death, nearer another pump")
    ax.scatter(deaths.geometry.x[col], deaths.geometry.y[col], s=6 + 8 * w[col], color=BLUE,
               edgecolor=SURFACE, linewidth=0.5, zorder=4, label="death, nearest Broad St")
    draw_pumps(ax)
    scalebar(ax)
axes[0].legend(loc="upper right", frameon=False, fontsize=8)
fig.text(0.02, 0.02, f"{reassigned} deaths change their nearest pump when distance is measured along streets.",
         fontsize=9, color="#52514e")
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(FIG / "fig2_voronoi_vs_network.png", dpi=200)
plt.close(fig)

# Fig 3: Monte-Carlo
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(sim_share, bins=30, color=MUTED, edgecolor=SURFACE, linewidth=1)
ax.axvline(share_net, color=BLUE, lw=2)
ax.text(share_net, ax.get_ylim()[1] * 0.92, f"  observed {share_net:.0%}", color=INK, va="top")
ax.set_xlabel("Share of deaths whose nearest pump (walking) is Broad Street")
ax.set_ylabel("Simulations")
ax.set_title(f"{N_SIM} random placements on the street network (p = {p_share:.3f})")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.tight_layout()
fig.savefig(FIG / "fig3_monte_carlo.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ evidence diagnostics
assignment_disagreement = reassigned / w.sum()
network_concentration_lift = share_net / sim_share.mean()
observed_distance_reduction = 1 - obs_mean_dist / sim_mean_dist.mean()

# ------------------------------------------------------------------ outputs
results = {
    "death_addresses": int(len(deaths)),
    "deaths": int(w.sum()),
    "pumps": int(len(pumps)),
    "street_edges_after_noding": G.number_of_edges(),
    "broad_street_pump_index": BROAD,
    "share_nearest_broad_euclidean": round(float(share_euc), 4),
    "share_nearest_broad_network": round(float(share_net), 4),
    "deaths_reassigned_euclid_to_network": reassigned,
    "mean_walk_to_broad_m_observed": round(float(obs_mean_dist), 1),
    "mean_walk_to_broad_m_random_mean": round(float(sim_mean_dist.mean()), 1),
    "share_nearest_broad_random_mean": round(float(sim_share.mean()), 4),
    "p_value_share": round(float(p_share), 4),
    "p_value_mean_distance": round(float(p_dist), 4),
    "mean_network_over_euclid_ratio": round(float((D_net[:, BROAD] / D_euc[:, BROAD]).mean()), 3),
    "web_mercator_distance_inflation": round(float(mercator_ratio), 3),
    "kde_peak_to_broad_m": round(float(peak_to_broad), 1),
    "network_exposure_diagnostics": {
        "assignment_disagreement_share": round(float(assignment_disagreement), 4),
        "broad_street_concentration_vs_network_null": round(float(network_concentration_lift), 2),
        "observed_walk_distance_reduction_vs_network_null": round(float(observed_distance_reduction), 4),
    },
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))
deaths.drop(columns="geometry").assign(x=deaths.geometry.x, y=deaths.geometry.y).to_csv(
    OUT / "deaths_nearest_pump.csv", index=False)
print(json.dumps(results, indent=2))
