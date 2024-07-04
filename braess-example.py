# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src import Plotting as pl
from src import ConvexOptimization as co
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

pl.mpl_params(fontsize=14)


# %%
def get_beta(G):
    alpha = np.array([G[u][v]["tt_function"](0) for u, v in G.edges()])
    return alpha


def get_alpha(G):
    beta = np.array(
        [G[u][v]["tt_function"](1) - G[u][v]["tt_function"](0) for u, v in G.edges()]
    )
    return beta


def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    nash_flow = nx.get_edge_attributes(G, "load")
    for e, f in lambda_funcs.items():
        beta = f(0)
        alpha = f(1) - beta
        x = nash_flow[e]
        if x != 0:
            K = x / (2 * beta + alpha * x)
        else:
            K = 0
        G.edges[e]["weight"] = K
    return G


def linear_function(alpha, beta, x):
    return alpha + beta * x


def braessGraph():
    G = nx.DiGraph()

    a, b, c, d = 0, 1, 2, 3

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0.5)}),
            (b, {"pos": (0.5, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 0.5)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: n / 100 + 10}),
            (b, d, {"tt_function": lambda n: n / 1000 + 25}),
            (a, c, {"tt_function": lambda n: n / 1000 + 25}),
            (c, d, {"tt_function": lambda n: n / 100 + 10}),
            (b, c, {"tt_function": lambda n: n / 1000 + 1}),
        ]
    )

    # G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


def social_cost_e(G, f):
    f = np.array(f)
    tt_f = nx.get_edge_attributes(G, "tt_function")
    beta = np.array([tt_f[e](0) for e in G.edges()])
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    return alpha * f**2 + beta * f


# %%


G = braessGraph()
# G.remove_edge(1, 2)
# G.add_edge(2, 1, **G[1][2])
load = 1000

P = [load, -0, -0, -1000]
demand = [P]


tapflow = mc.solve_multicommodity_tap(G, demand, social_optimum=False)
nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")

pl.graphPlotCC(G, cc=tapflow, edge_labels=dict(zip(G.edges, np.round(tapflow, 2))))
print(mc.price_of_anarchy(G, demand))
# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
betas = np.linspace(-5, 20, num=100)
labels = ["so_p", "ue_p", "so_m", "ue_m"]

edge = (1, 2)

sc_df = pd.DataFrame(index=betas, columns=labels)
for beta in betas:
    G.edges[edge]["tt_function"] = lambda n: n / 1000 + beta

    f_so_p = tap.social_optimum(G, P, positive_constraint=True)
    f_ue_p = tap.user_equilibrium(G, P, positive_constraint=True)
    f_so_m = tap.social_optimum(G, P, positive_constraint=False)
    f_ue_m = tap.user_equilibrium(G, P, positive_constraint=False)

    scs = []
    for f in [f_so_p, f_ue_p, f_so_m, f_ue_m]:
        sc = tap.social_cost(G, f) / load

        social_cost = tap.social_cost(G, f) / load
        scs.append(social_cost)
    sc_df.loc[beta] = scs


ax.plot(betas, sc_df["so_p"], label="SO Positive", linewidth=10, color="#abd9e9")
ax.plot(betas, sc_df["ue_p"], label="UE Positive", linewidth=5, color="#2c7bb6")
ax.plot(betas, sc_df["so_m"], label="SO Negative", linestyle="--", color="#d7191c")
ax.plot(betas, sc_df["ue_m"], label="UE Negative", linestyle="--", color="#fdae61")

ax.set_xlabel("Beta")
ax.set_ylabel("Social Cost")


# Prepare the data
data = sc_df["ue_p"].iloc[:25]
X = np.array(data.index).reshape(-1, 1)
y = np.array(data).reshape(-1, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Print the results
print("Slope:", slope)
print("Intercept:", intercept)

# Generate the predicted values
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_pred)

ax.plot(
    X_pred,
    y_pred,
    color="black",
    label=f"Linear Fit, m={round(slope,2)}",
    linestyle="-",
    linewidth=1,
)
ax.grid()
ax.legend()


# %%


def superL(G):
    tt_f = nx.get_edge_attributes(G, "tt_function")
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    # beta = np.array([tt_f[e](0) for e in G.edges()])

    E = -nx.incidence_matrix(G, oriented=True)

    kappa = 1 / alpha
    # nx.set_edge_attributes(G, dict(zip(G.edges, kappa)), "kappa")
    # L = nx.laplacian_matrix(G, weight="kappa").toarray()
    # superL = L + L.T
    # np.fill_diagonal(superL, 0)
    # np.fill_diagonal(superL, -np.sum(superL, axis=0))
    return E @ np.diag(kappa) @ E.T


def Gamma_matrix(G):
    alpha = get_alpha(G)
    beta = get_beta(G)
    nx.set_edge_attributes(G, dict(zip(G.edges, beta / alpha)), "gamma")
    A = nx.adjacency_matrix(G, weight="gamma")
    Gamma = A - A.T
    return Gamma


# %%
load = 300
G = braessGraph()
G.add_edge(1, 2, tt_function=lambda n: n * 1000 + 15)
# G.remove_edge(1, 2)
alpha = get_alpha(G)
beta = get_beta(G)
P = [load, 0, 0, -load]

E = -nx.incidence_matrix(G, oriented=True).toarray()
Gamma = Gamma_matrix(G).toarray()
one_v = np.ones(G.number_of_nodes())
L = superL(G)


f_ue, lamb_ue = tap.linearTAP(G, P)
f_so, lamb_so = tap.linearTAP(G, P, social_optimum=True)

sc_ue = tap.social_cost(G, f_ue)
sc_so = tap.social_cost(G, f_so)

lamb_ue - lamb_so
print("F_SO:", 2 * f_so)
print("F_UE:", f_ue)
print("Delta F:", f_ue - f_so)
print("Delta lambda:", lamb_ue - lamb_so)
print("Delta SC:", sc_ue - sc_so)


# %%

delta_lambda = -np.linalg.pinv(L) @ P
delta_lambda

# %%

delta_f = E.T @ (2 * lamb_ue - lamb_so) - beta
delta_f

# %%


def fue_minus_fso(G):
    alpha = get_alpha(G)
    beta = get_beta(G)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    Gamma = Gamma_matrix(G).toarray()
    L = superL(G)
    # f_ue, lamb_ue = tap.linearTAP(G, P)
    # f_so, lamb_so = tap.linearTAP(G, P, social_optimum=True)
    # delta_lambda = -np.linalg.pinv(L) @ P
    # elta_f = E.T @ (2 * lamb_ue - lamb_so) - beta
    one = np.ones(G.number_of_nodes())
    delta_f = (E.T @ np.linalg.pinv(L) @ Gamma @ one - beta) / (2 * alpha)
    return delta_f


def fue_plus_fso(G, P):
    P = np.array(P)
    alpha = get_alpha(G)
    beta = get_beta(G)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    Gamma = Gamma_matrix(G).toarray()
    one = np.ones(G.number_of_nodes())
    L = superL(G)
    delta_f = (
        E.T @ np.linalg.pinv(L) @ (2 * P + 3 / 2 * Gamma @ one) / (alpha)
        - 3 / 2 * beta / alpha
    )
    return delta_f


print(np.isclose(fue_minus_fso(G), (f_ue - f_so)))
print(np.isclose(fue_plus_fso(G, P), (f_ue + f_so)))


def delta_SC(G, P):
    alpha = get_alpha(G)
    beta = get_beta(G)

    delta = alpha * fue_plus_fso(G, P) * fue_minus_fso(G) + beta * fue_minus_fso(G)
    return delta


np.isclose(sum(delta_SC(G, P)), sc_ue - sc_so)

# %%
delta_SC(G, P)
# %%

print(fue_minus_fso(G))
# %%
print(fue_plus_fso(G, P))
# %%

sum(E.T @ lamb_ue / alpha)


# %%
def sc_sum(G, P):
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    Gamma = Gamma_matrix(G).toarray()
    alpha = get_alpha(G)
    beta = get_beta(G)
    one = np.ones(G.number_of_nodes())

    lam = E.T @ np.linalg.pinv(E @ np.diag(1 / alpha) @ E.T) @ (P + Gamma @ one)

    return sum(lam * (lam - beta) / alpha)


alphas = np.linspace(1, 1000, num=50)
betas = np.linspace(0, 1e2, num=100)
sc_df = pd.DataFrame(index=alphas, columns=betas)

for a in alphas:
    for b in betas:
        G.edges[1, 2]["tt_function"] = lambda n: n / a + b
        sc = sc_sum(G, P)
        sc_df.loc[a, b] = sc / load

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Assuming the variables `betas`, `alphas`, and `sc_df` are defined as follows for demonstration:
X, Y = np.meshgrid(betas, alphas)
Z = sc_df.values.astype(float)

# Creating the interactive plot using Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

fig.update_layout(
    scene=dict(xaxis_title="Beta", yaxis_title="Alpha", zaxis_title="Social Cost"),
    width=800,
    height=600,
    title="Interactive 3D Surface Plot",
)

fig.show()


# %%

# %%
