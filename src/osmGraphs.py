# %%
import osmnx as ox
import networkx as nx
import numpy as np
import warnings
from rasterstats import zonal_stats
import rioxarray as riox
import geopandas as gpd
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, MultiPoint, Polygon
import requests
from io import StringIO
import pandas as pd

from sklearn.cluster import KMeans

from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import matplotlib as mpl

from src import periodicBoudary as pb

PATH_TO_GHSL_TIF = "data/GHS/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif"


def demand_list(G, commodity, gamma=0.1):
    nodes, edges = ox.graph_to_gdfs(G)
    nodes["source_node"] = False

    demands = []
    tot_pop = 0
    for idx, row in commodity.iterrows():
        pop_com = row["population"]
        vor_geom = row["voronoi"]
        source_node = idx
        target_nodes = nodes[~nodes["geometry"].within(vor_geom)].index

        # random_com_node = np.random.choice(com_nodes)
        nodes.loc[source_node, "source_node"] = True

        P = dict(zip(G.nodes(), np.zeros(G.number_of_nodes())))
        # for node in com_nodes:
        P[source_node] = pop_com * gamma  # / len(com_nodes)
        # P[random_com_node] = pop_com
        for node in target_nodes:
            P[node] = -pop_com * gamma / len(target_nodes)
        # print(sum(P.values()))

        demands.append(list(P.values()))
        # print(c, pop_com)
        tot_pop += pop_com
    # print("Total Population:", tot_pop)
    return demands, nodes


def select_evenly_distributed_nodes(G, N):
    """
    Selects N points from the GeoDataFrame such that they are evenly spatially distributed.

    Parameters:
    - G (Graph): graph whch nodes containing spatially distributed points.
    - N (int): Number of points to select.

    Returns:
    - GeoDataFrame: GeoDataFrame containing the selected points.
    """
    gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    # Ensure the GeoDataFrame has the necessary geometry column
    if gdf.geometry.name != "geometry":
        raise ValueError("GeoDataFrame must have a geometry column named 'geometry'")

    # Extract the coordinates of the points
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=N, n_init=100, random_state=0)
    kmeans.fit(coords)

    # Find the closest point to each cluster centroid
    selected_indices = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(coords - centroid, axis=1)
        closest_index = np.argmin(distances)
        selected_indices.append(closest_index)

    # Create a new GeoDataFrame with the selected points
    selected_gdf = gdf.iloc[selected_indices]

    mask = gdf.geometry.union_all().convex_hull.buffer(1e-3)
    voronoi_gdf = compute_voronoi_polys_of_nodes(selected_gdf, mask=mask)

    for i, row in voronoi_gdf.iterrows():
        vor_geom = row["voronoi"]
        vorpop = gdf[gdf.geometry.within(vor_geom)]["population"].sum()
        voronoi_gdf.loc[i, "population"] = vorpop

    # gdf.loc[gdf.geometry.within(voronoi_gdf.geometry), "population"]
    voronoi_gdf.set_geometry("geometry", inplace=True)
    voronoi_gdf.set_crs(gdf.crs, inplace=True)
    return voronoi_gdf


def assign_nodes_to_districts(nodes, districts):
    """
    Assign nodes to districts based on the district boundaries.

    Parameters:
        nodes (geopandas.GeoDataFrame): The input nodes as a GeoDataFrame.
        districts (geopandas.GeoDataFrame): The district boundaries as a GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: A modified GeoDataFrame with added district information.
    """
    # Assign nodes to districts
    nodes["district"] = None
    for i, district in districts.iterrows():
        nodes.loc[nodes.within(district["geometry"]), "district"] = district["name"]

    return nodes


def get_city_and_district_boundaries(city_name):
    # Geocode the city name to get its latitude and longitude
    geolocator = Nominatim(user_agent="city_boundaries_app")
    location = geolocator.geocode(city_name)

    if not location:
        return None, None

    # Use OSMnx to get the city boundary
    city_boundary = ox.geocode_to_gdf(city_name)

    # Use OSMnx to get the city districts (administrative boundaries level 10)
    districts10 = ox.features.features_from_place(
        city_name, tags={"admin_level": "10"}
    ).loc["relation", :]

    districts9 = ox.features.features_from_place(
        city_name, tags={"admin_level": "9"}
    ).loc["relation", :]

    return city_boundary, districts10, districts9


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Generate finite Voronoi polygons based on a given Voronoi diagram.

    Parameters:
        vor: Voronoi object representing a Voronoi diagram.
        radius: Maximum distance from the Voronoi center to the finite polygons (default: None).

    Returns:
        tuple: A tuple containing two elements:
            - List of regions, where each region is represented by a list of vertices.
            - Numpy array of vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()  # vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def compute_voronoi_polys_of_nodes(nodes, mask=None):
    """
    Compute Voronoi polygons for a set of nodes.

    Parameters:
        nodes (geopandas.GeoDataFrame): The input nodes as a GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: A modified GeoDataFrame with added Voronoi polygon geometries.
    """
    vor_nodes = nodes.copy()
    points = np.array(list(zip(np.array(vor_nodes.x), np.array(vor_nodes.y))))
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    if mask is None:
        pts = MultiPoint([Point(i) for i in points])
        mask = pts.convex_hull

    if isinstance(mask, gpd.GeoDataFrame) or isinstance(mask, gpd.GeoSeries):
        mask = mask.geometry.union_all().convex_hull

    voronoi_polys = gpd.GeoSeries(
        [Polygon(vertices[region]) for region in regions]
    ).set_crs(nodes.crs)
    voronoi_polys = voronoi_polys.intersection(mask)

    vor_nodes["voronoi"] = list(voronoi_polys)
    vor_nodes = vor_nodes.set_geometry("voronoi").set_crs(nodes.crs)

    return vor_nodes


def clip_to_gdf(voronoi_nodes, raster_path):
    """
    Clip a raster to the extent of a GeoDataFrame representing Voronoi nodes.

    Parameters:
        voronoi_nodes: GeoDataFrame representing Voronoi nodes.
        raster_path: Path to the raster file.

    Returns:
        xarray.DataArray: Clipped raster data.
    """
    # Read raster
    raster = riox.open_rasterio(raster_path, masked=True)

    # update crs
    crs = raster.spatial_ref.crs_wkt
    voronoi_nodes = voronoi_nodes.to_crs(crs)

    # clip to convex hull of gdf
    clipped = raster.rio.clip_box(*voronoi_nodes.union_all().bounds)
    clipped = clipped.rio.clip([voronoi_nodes.union_all().convex_hull])
    return clipped


def population_from_raster_to_gdf(
    gdf,
    raster_path=PATH_TO_GHSL_TIF,
):
    """
    Calculate population-related attributes for a GeoDataFrame based on a raster dataset.

    Parameters:
        gdf: GeoDataFrame containing Voronoi polygons.
        raster_path: Path to the raster dataset.

    Returns:
        gdf: GeoDataFrame with population-related attributes added.
    """
    org_crs = gdf.crs
    clipped_raster = clip_to_gdf(gdf, raster_path)

    # update crs
    crs = clipped_raster.spatial_ref.crs_wkt
    gdf = gdf.to_crs(crs)

    # upsampling raster would come here.
    # Memory intensive, thus leaving out atm. Maybe better though

    # warnings.filterwarnings("ignore")
    stats = zonal_stats(
        gdf["voronoi"],
        clipped_raster.data[0],
        affine=clipped_raster.rio.transform(),
        stats="sum",
        nodata=np.nan,
        all_touched=False,
    )

    gdf["population"] = [s["sum"] for s in stats]

    # warn if population smaller zero
    if gdf["population"].min() < 0:
        warnings.warn("At least one value for population smaller zero")
        gdf.loc[gdf["population"] < 0, "population"] = 0

    if gdf["population"].isna().sum() > 0:
        print("At least one nan value for population. Setting na to zero")
        gdf.loc[gdf["population"].isna(), "population"] = 0

    pop_dens = gdf["population"] / (gdf["voronoi"].area * 1e-6)
    vor_area = gdf["voronoi"].area * 1e-6

    gdf["population_density"] = pop_dens
    gdf["voronoi_area"] = vor_area
    gdf = gdf.to_crs(org_crs)
    return gdf


def raster_population_to_voronois(
    nodes,
    raster_path=PATH_TO_GHSL_TIF,
):
    voronoi_nodes = compute_voronoi_polys_of_nodes(nodes)
    voronoi_nodes = population_from_raster_to_gdf(voronoi_nodes, raster_path)

    return voronoi_nodes


def population_to_graph_nodes(graph):
    nodes = ox.graph_to_gdfs(graph, edges=False)
    voronoi_nodes = raster_population_to_voronois(nodes)

    nx.set_node_attributes(graph, voronoi_nodes["population"], "population")
    nx.set_node_attributes(
        graph, voronoi_nodes["population_density"], "population_density"
    )
    nx.set_node_attributes(graph, voronoi_nodes["voronoi"], "voronoi")

    return graph


def map_highway_to_number_of_lanes(highway_type):
    """
    Map a highway type to the corresponding number of lanes.

    Parameters:
        highway_type (str): The highway according to OSM tags.

    Returns:
        int: The number of lanes.
    """
    if highway_type == "motorway" or highway_type == "trunk":
        return 4
    elif highway_type == "primary":
        return 3
    elif (
        highway_type == "secondary"
        or highway_type == "motorway_link"
        or highway_type == "trunk_link"
        or highway_type == "primary_link"
    ):
        return 2
    else:
        return 1


def set_number_of_lanes(G):
    """
    Set the number of lanes attribute for each edge in the graph.

    Parameters:
        G (networkx.MultiDiGraph): Input road network graph.

    Returns:
        networkx.MultiDiGraph: Updated road network graph with the 'lanes' attribute set for each edge.
    """
    edges = ox.graph_to_gdfs(G, nodes=False)
    lanes = []

    # Iterate over the rows of the DataFrame
    for k, v in edges[["lanes", "highway"]].iterrows():
        lane_value = v["lanes"]
        highway_type = v["highway"]

        if isinstance(lane_value, list):
            # Convert list elements to float and compute the mean
            lane_result = np.mean(list(map(float, lane_value)))
        else:
            if isinstance(lane_value, str):
                # If lane_value is a string, convert it to float (assuming this is correct in context)
                lane_result = float(lane_value)
            else:
                if np.isnan(lane_value):
                    # If lane_value is NaN, map the highway type to the number of lanes
                    lane_result = map_highway_to_number_of_lanes(highway_type)
                else:
                    # Otherwise, use the lane_value as it is
                    lane_result = lane_value

        # Append the result to the lanes list
        lanes.append(lane_result)
    edges["lanes"] = lanes
    nx.set_edge_attributes(G, edges["lanes"], "lanes")
    return G


def effective_travel_time(edge, gamma=1):
    l = edge["length"]
    m = edge["lanes"]
    v = edge["speed_kph"] / 3.6
    tr = 2
    d = 5
    walking_speed = 1.4
    tmax = l / walking_speed
    tmin = l / v

    teff = lambda x: (gamma * tr * l * x) / (l * m - gamma * d * x)

    teff_cond = lambda x: (
        tmax if teff(x) > tmax else tmin if teff(x) < tmin else teff(x)
    )
    return teff_cond


def linear_function(edge, gamma=1):
    tr = 2
    d = 5
    m = edge["lanes"]
    l = edge["length"]
    v = edge["speed_kph"] / 3.6
    walking_speed = 1.4

    t_max = l / walking_speed
    t_min = l / v
    xmax = (l * m) / (walking_speed * (gamma * tr + (gamma * d) / walking_speed))
    beta = t_min
    alpha = (t_max - beta) / xmax
    # alpha = gamma * tr / m  # taylor expand

    xmin = (t_min - beta) / alpha

    edge["alpha"] = alpha
    edge["beta"] = beta
    edge["xmax"] = xmax
    edge["xmin"] = xmin
    edge["tmax"] = t_max
    edge["tmin"] = t_min

    f = lambda x: alpha * x + beta
    # f_cond = lambda x: (t_max if f(x) > t_max else t_min if f(x) < t_min else f(x))

    return f


def set_effective_travel_time(G, gamma=1):
    for i, j, edge in G.edges(data=True):
        edge["tt_function"] = linear_function(edge, gamma)
        # edge["energy_function"] = potential_energy(edge, gamma)
    return G


def osmGraph(
    place_name,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]',
    return_boundary=False,
    heavy_boundary=False,
    **kwargs,
):
    # Fetch the road network graph for the specified place
    city_boundary, districts10, districts9 = get_city_and_district_boundaries(
        place_name
    )

    city_hull = city_boundary.geometry.convex_hull

    graph = ox.graph_from_polygon(city_hull.iloc[0], custom_filter=highway_filter)
    gcc_nodes = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(gcc_nodes)

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    # print(graph)
    graph = population_to_graph_nodes(graph)

    nodes, edges = ox.graph_to_gdfs(graph)
    districts = districts9
    nodes = assign_nodes_to_districts(nodes, districts)
    if nodes["district"].isnull().all():
        districts = districts10
        nodes = assign_nodes_to_districts(nodes, districts)

    nodes["district"] = nodes["district"].fillna("no_district")
    graph = ox.graph_from_gdfs(nodes, edges)
    graph = set_number_of_lanes(graph)
    graph = set_effective_travel_time(graph)

    if heavy_boundary:
        buffer_meter = kwargs.pop("buffer_meter", 10_000)
        graph = pb.set_boundary_population(graph, buffer_metre=buffer_meter)

    print(graph)
    if return_boundary:
        return graph, city_boundary
    return graph


# %%

if __name__ == "__main__":
    # Define the place name or address
    place_name = "Cologne, Germany"

    G = osmGraph(place_name, heavy_boundary=True)

    nodes, edges = ox.graph_to_gdfs(G)
    # %%

    # Example usage

    N = 100
    nodes_select = select_evenly_distributed_nodes(nodes, N)

    fig, ax = plt.subplots(figsize=(10, 10))
    # nodes.plot(ax=ax, marker=".", zorder=2, color="lightgrey")
    edges.plot(ax=ax, zorder=2, color="white", linewidth=0.5)
    nodes_select.set_geometry("voronoi").plot("population", ax=ax, markersize=50)
    nodes_select.plot(ax=ax, color="red", markersize=10, zorder=2)

# %%
