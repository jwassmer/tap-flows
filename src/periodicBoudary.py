# %%
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import geopandas as gpd
import rioxarray as rx
from rasterstats import zonal_stats

from src import osmGraphs as og
from src import Plotting as pl

GHS_FILEPATH = "data/GHS/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif"


def ghs_city(buffered_gdf, GHS_FILEPATH=GHS_FILEPATH):
    clipped = og.clip_to_gdf(buffered_gdf, GHS_FILEPATH)
    reprojeced = clipped.rio.reproject(buffered_gdf.crs)
    return reprojeced


def boundary_nodes(G):
    nodes = ox.graph_to_gdfs(G, edges=False)
    nodes["boundary"] = False
    points = np.array(list(zip(nodes.x, nodes.y)))
    hull = ConvexHull(points)
    hull_idxs = hull.vertices
    nodes.iloc[hull_idxs, nodes.columns.get_loc("boundary")] = True

    polygon = Polygon(points[hull_idxs])


def boundary_nodes_with_pop(G, buffer_metre=10_000):
    nodes = ox.graph_to_gdfs(G, edges=False)
    nodes["boundary"] = False
    points = np.array(list(zip(nodes.x, nodes.y)))
    hull = ConvexHull(points)
    hull_idxs = hull.vertices
    nodes.iloc[hull_idxs, nodes.columns.get_loc("boundary")] = True

    hull_polygon = Polygon(points[hull_idxs])
    hull_gdf = gpd.GeoDataFrame(geometry=[hull_polygon]).set_crs(nodes.crs)

    buffered_gdf = hull_gdf.to_crs(3043).buffer(buffer_metre).to_crs(hull_gdf.crs)

    vor = og.compute_voronoi_polys_of_nodes(nodes[nodes["boundary"]], mask=buffered_gdf)
    vor["voronoi"] = vor["voronoi"].apply(
        lambda x: x.difference(hull_gdf.geometry.unary_union)
    )

    clipped_ghs = ghs_city(buffered_gdf)

    stats = zonal_stats(
        vor["voronoi"],
        clipped_ghs.data[0],
        affine=clipped_ghs.rio.transform(),
        stats="sum",
        nodata=np.nan,
        all_touched=False,
    )

    vor["population"] = [s["sum"] for s in stats]
    nx.set_node_attributes(G, nodes["boundary"], "boundary")
    return vor, clipped_ghs


def set_boundary_population(G, buffer_metre=10_000):
    vor, _ = boundary_nodes_with_pop(G, buffer_metre=buffer_metre)
    population = nx.get_node_attributes(G, "population")
    for n in G.nodes:
        if n in vor.index:
            population[n] += vor.loc[n, "population"]
    nx.set_node_attributes(G, population, "population")
    return G


# %%

if __name__ == "__main__":
    G, city_boundary = og.osmGraph("Cologne,Germany", return_boundary=True)

    buffer_metre = 10_000

    vor, clipped_ghs = boundary_nodes_with_pop(G, buffer_metre=buffer_metre)
    G = set_boundary_population(G, buffer_metre=buffer_metre)
    nodes, edges = ox.graph_to_gdfs(G)

    # nodes = nodes.sort_values(by="population", ascending=True)
    # nodes.plot(column="population", cmap="Reds", legend=True)
    # nodes.population.max()

    # %%

    fig, ax = plt.subplots(figsize=(8, 6))

    # vor.plot(ax=ax, zorder=2, column="population", alpha=0.5, cmap="binary_r")
    vor.boundary.plot(ax=ax, color="white", zorder=3, linewidth=1)
    clipped_ghs.plot(ax=ax, cmap="cividis", vmax=200, vmin=10)
    edges.plot(ax=ax, linewidth=0.2, edgecolor="white")
    nodes[nodes["boundary"]].plot(ax=ax, markersize=10, zorder=3, color="red")


# %%
