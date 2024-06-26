# %%
import time
import numpy as np
import pickle
import networkx as nx
import osmnx as ox
import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from src import multiCommodityTAP as mc

# %%


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
        tmax if teff(x) > tmax else tmin if teff(x) < tmin else teff(x)
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


def solve_multicommodity_tap(G, demands):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    start_time = time.time()
    A = -nx.incidence_matrix(G, oriented=True).toarray()

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    beta = np.array([tt_funcs[e](0) for e in G.edges(keys=True)])
    alpha = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges(keys=True)])

    # Number of edges
    num_edges = G.number_of_edges()

    # Number of commodities
    num_commodities = len(demands)

    # Variables for the flow on each edge for each commodity
    flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
    # flows = cp.Variable((num_commodities, num_edges))  # , nonneg=True)
    # Combine the constraints for flow conservation
    constraints = []
    for k in range(num_commodities):
        constraints.append(A @ flows[k] == demands[k])

    # Objective function
    total_flow = cp.sum(flows)
    objective = cp.Minimize(
        cp.sum(cp.multiply(alpha, total_flow**2))
        + cp.sum(cp.multiply(beta, total_flow))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    conv_time = time.time() - start_time
    print("Time:", conv_time, "s")

    return total_flow.value, conv_time


def OD_matrix(G):
    num_nodes = G.number_of_nodes()
    A = -np.ones((num_nodes, num_nodes))
    np.fill_diagonal(A, num_nodes - 1)
    return A


# %%
with open("data/cologne_graph.pickle", "rb") as f:
    G = pickle.load(f)
    G = set_number_of_lanes(G)
    G = set_effective_travel_time(G)
    # G = ox.convert.to_digraph(G)

    gcc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(gcc_nodes)  # .copy()

    A = OD_matrix(G)

    # F = solve_multicommodity_tap(G, demands)

# %%

time_list = []
numK_list = range(1, 25)

for numK in numK_list:
    demands = [a for a in A[:numK]]
    F, t = solve_multicommodity_tap(G, demands)
    print("Number of commodities:", numK)
    time_list.append(t)


# %%

X = np.log10(np.array(numK_list)).reshape(-1, 1)
y = np.log10(np.array(time_list))

reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_

print("Slope:", slope)

# Plotting the fit line and error
y_pred = reg.predict(X)
error = y - y_pred

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(numK_list, time_list, marker="o")
ax.plot(numK_list, 10**y_pred, color="red", linestyle="--")
ax.fill_between(
    numK_list, 10 ** (y_pred - error), 10 ** (y_pred + error), color="gray", alpha=0.2
)


Kmax = 6000
ymax = reg.predict(np.array([[np.log10(Kmax)]]))
ax.scatter([Kmax], [10**ymax], color="red", marker="x", s=100)

ax.grid()
ax.set_xlabel("Number of commodities")
ax.set_ylabel("Time (s)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title(
    f"Time Complexity, num(E)={G.number_of_edges()}, num(N)={G.number_of_nodes()}"
)
plt.show()

fig.savefig("figs/time_complexity.png", dpi=300)

# %%

# %%
