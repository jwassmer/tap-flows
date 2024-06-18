# %%
import osmnx as ox
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
import cvxpy as cp


from src import Plotting as pl


def source_sink_vector(G):
    sources = G.source_nodes
    targets = G.target_nodes
    total_load = G.total_load

    node_dict = dict(zip(G.nodes, range(G.number_of_nodes())))

    sources = [node_dict[s] for s in sources]
    targets = [node_dict[t] for t in targets]

    P = np.zeros(G.number_of_nodes())
    P[sources] = total_load / len(sources)
    P[targets] = -total_load / len(targets)

    return P


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
    v = edge["speed_ms"]
    tr = 2
    d = 5
    walking_speed = 1.4
    tmax = l / walking_speed
    tmin = l / v

    teff = lambda x: (gamma * tr * l * x) / (l * m - gamma * d * x)

    teff_cond = lambda x: (
        t_max if teff(x) > tmax else tmin if teff(x) < tmin else teff(x)
    )
    return teff_cond


def lin_potential_energy(edge):
    alpha = edge["alpha"]
    beta = edge["beta"]

    xmax = edge["xmax"]
    xmin = edge["xmin"]
    tmax = edge["tmax"]
    tmin = edge["tmin"]

    f = lambda x: 1 / 2 * alpha * x**2 + beta * x
    # f_cond = lambda x: (x * tmax if x > xmax else x * tmin if x < xmin else f(x))

    return f


def total_social_cost(G, kwd="flow"):
    return sum(
        [
            G.edges[e][kwd] * G.edges[e]["tt_function"](G.edges[e][kwd])
            for e in G.edges(keys=True)
        ]
    )


def linear_function(edge, gamma=1):
    tr = 2
    d = 5
    m = edge["lanes"]
    l = edge["length"]
    v = edge["speed_ms"]
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


def convex_optimization_kcl_tap(G, solver=cp.OSQP, verbose=False):
    # Get the number of edges and nodes
    num_edges = G.number_of_edges()

    # Create the edge incidence matrix E
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    betas = np.array([tt_funcs[e](0) for e in G.edges(keys=True)])
    alphas = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges(keys=True)])

    P = np.array(list(nx.get_node_attributes(G, "P").values()))

    # Define the flow variable f_e
    f = cp.Variable(num_edges)

    objective = cp.Minimize(cp.sum(alphas @ f**2 / 2 + betas @ f))

    # Constraints: E @ f = p
    xmax = np.array(list(nx.get_edge_attributes(G, "xmax").values()))
    # tmax = 100000 * np.ones(num_edges)

    constraints = [
        E @ f == P,
        f >= np.zeros(num_edges),
        f <= xmax,
    ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=solver, verbose=verbose)

    # Get the results
    flow_values = f.value

    flow_values, problem.value
    return flow_values, problem.value


# %%
with open("data/cologne_full_graph.pickle", "rb") as f:
    G = pickle.load(f)
    G = set_number_of_lanes(G)
    G = set_effective_travel_time(G)
    # G = ox.convert.to_digraph(G)

    gcc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(gcc_nodes)  # .copy()

    nodes = list(G.nodes)
    source_idx = 1500
    sources = [nodes[source_idx]]
    targets = np.delete(nodes, source_idx)

    G.source_nodes = sources
    G.target_nodes = targets


# %%

loads = [1, 10, 20]
for load in loads:
    G.total_load = load
    P = source_sink_vector(G)
    Pdict = dict(zip(G.nodes(), P))
    nx.set_node_attributes(G, Pdict, "P")

    F = convex_optimization_kcl_tap(G)[0]
    vmin = 1e-1

    nodes, edges = ox.graph_to_gdfs(G)
    edges["flow"] = F
    nx.set_edge_attributes(G, edges["flow"], "flow")
    edges = edges.sort_values(by="flow", ascending=True)

    util = edges["flow"] / edges["xmax"]
    edges["util"] = util
    # edges = edges.sort_values(by="util", ascending=True)

    cmap = mpl.colormaps.get_cmap("cividis")
    cmap.set_under("lightgrey")
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=F.max())

    fig, ax = plt.subplots(figsize=(10, 10))
    edges.plot(ax=ax, column="flow", cmap=cmap, norm=norm, zorder=1)
    nodes.loc[G.source_nodes].plot(ax=ax, color="red", zorder=2)
    # nodes.loc[G.target_nodes].plot(ax=ax, color="red", zorder=2)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, extend="min"
    )
    print("Social Cost:", total_social_cost(G) / edges.flow.sum())
    # print(edges.flow.sum())


# %%


edges[edges["flow"] > edges["xmax"]][["flow", "xmax"]]

# %%
# %%


edge = list(G.edges(data=True))[0][-1]
l, m, v = edge["length"], edge["lanes"], edge["speed_ms"]
gamma, tr, d = 1, 2, 5
walking_speed = 1.4
t_max = l / walking_speed
t_min = l / v

eff_func = effective_travel_time(edge)
linear_func = linear_function(edge)
# potential_energy_func = potential_energy(edge)


xmax = edge["xmax"]
xmin = edge["xmin"]
beta = edge["beta"]


# Generate L values
x_values = np.linspace(0, int(xmax * 1.2), 100)

t_eff_values = [eff_func(x) for x in x_values]
linear_values = [linear_func(x) for x in x_values]


# Plot the original function and the linear function
plt.figure(figsize=(10, 6))
plt.plot(
    x_values,
    t_eff_values,
    label="Original Function $t_{ij, \mathrm{eff}}(L_{ij})$",
    color="blue",
)
plt.plot(
    x_values,
    linear_values,
    label="Linear Function $t(x) = \\alpha x + \\beta$",
    color="red",
    linestyle="--",
)

plt.axhline(y=t_max, color="green", linestyle="-.", label="$t_{\\mathrm{max}}$")
plt.axhline(y=t_min, color="orange", linestyle="-.", label="$t_{\\mathrm{min}}$")
plt.axvline(x=xmax, color="green", linestyle="-.")
plt.axvline(x=xmin, color="orange", linestyle="-.")
plt.xlabel("$L_{ij}$")
plt.ylabel("$t_{ij, \mathrm{eff}}(L_{ij})$")
plt.title("Original Function and Linear Function")
plt.legend()
plt.grid(True)
plt.show()

# %%


# %%
