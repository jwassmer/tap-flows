# %%
import networkx as nx
import numpy as np

np.set_printoptions(suppress=True, precision=3)
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src import TAPOptimization as tap
from src import Plotting as pl
from sklearn.linear_model import LinearRegression
from src import SocialCost as sc

pl.mpl_params(fontsize=14)


# %%

G = tap.random_graph(
    seed=42,
    num_edges=25,
    num_nodes=20,
    alpha=1,
    beta=10,
)

# G = braessGraph()

E = -nx.incidence_matrix(G, oriented=True).toarray()
P = np.zeros(G.number_of_nodes())
load = 500
source = 15
P[source] = load
targets = [16]  # np.delete(np.arange(G.number_of_nodes()), source)
P[targets] = -load / len(targets)
# %%

f_ue = tap.user_equilibrium(G, P, positive_constraint=True)
f_, lamb_ue = tap.linearTAP(G, P)
print(tap.social_cost(G, f_) / load)

pl.graphPlotCC(G, cc=f_ue)  # , edge_labels=dict(zip(G.edges, f_ue)))

slopes = {}
for e in G.edges():
    s = sc.slope_social_cost(G, P, e)
    slopes[e] = s
slopes


# %%
edge = (19, 11)
alphas = np.linspace(1, 100, 25)
alphas = alphas[alphas != 0]
betas = np.linspace(-500, 50, 25)

sc_so_df = pd.DataFrame(index=alphas, columns=betas)
sc_ue_df = pd.DataFrame(index=alphas, columns=betas)
sc_lap_df = pd.DataFrame(index=alphas, columns=betas)
pos_constraint = False
for a in alphas:
    for b in betas:
        G.edges[edge]["tt_function"] = lambda n: n / a + b
        # G.edges[edge[::-1]]["tt_function"] = lambda n: n / a + b
        # f_ue, lamb_ue = tap.linearTAP(G, P)
        # f_so, lamb_so = tap.linearTAP(G, P, social_optimum=True)
        f_ue = tap.user_equilibrium(G, P, positive_constraint=pos_constraint)
        f_so = tap.social_optimum(G, P, positive_constraint=pos_constraint)
        sc_so = tap.social_cost(G, f_so)
        sc_ue = tap.social_cost(G, f_ue)
        # sc_lap = tap.social_cost(G, f_lap)
        sc_ue_df.loc[a, b] = sc_ue / load
        sc_so_df.loc[a, b] = sc_so / load
        # sc_lap_df.loc[a, b] = sc_lap / load

# %%
# Assuming the variables betas, alphas, and sc_df are defined as follows for demonstration:
X, Y = np.meshgrid(betas, alphas)
# Z = sc_ue_df.values.astype(float)
Z1 = sc_so_df.values.astype(float)
Z2 = sc_ue_df.values.astype(float)
Z3 = sc_lap_df.values.astype(float)

# Create labels for each surface plot
labels1 = np.array([f"SO: {z:.2f}" for z in Z1.flatten()]).reshape(Z1.shape)
labels2 = np.array([f"UE: {z:.2f}" for z in Z2.flatten()]).reshape(Z2.shape)
labels3 = np.array([f"LAP: {z:.2f}" for z in Z3.flatten()]).reshape(Z3.shape)


fig = go.Figure()
# Creating the interactive plot using Plotly
fig.add_trace(
    go.Surface(
        z=Z1,
        x=Y,
        y=X,
        colorscale="Cividis",
        name="Social Optimum",
        text=labels1,
        hoverinfo="x+y+text",
    )
)
fig.add_trace(
    go.Surface(
        z=Z2,
        x=Y,
        y=X,
        colorscale="Viridis",
        name="User Equilibrium",
        showscale=False,
        opacity=0.9,
        text=labels2,
        hoverinfo="x+y+text",
    )
)


fig.update_layout(
    scene=dict(
        xaxis_title="1/Alpha",
        yaxis_title="Beta",
        zaxis_title="Social Cost",
        # zaxis=dict(range=[-500, Z2.max()]),
    ),
    width=800,
    height=600,
    title="Interactive 3D Surface Plot",
)

fig.show()

# %%


tt_fs = nx.get_edge_attributes(G, "tt_function")
alpha_arr = np.array([tt_fs[e](1) - tt_fs[e](0) for e in G.edges()])
beta_arr = np.array([tt_fs[e](0) for e in G.edges()])

beta_e = np.linspace(-1e3, 1e3, 100)

edge_idx = list(G.edges).index(edge)


sc_beta = []
for i, bet in enumerate(beta_e):
    beta_arr[edge_idx] = bet
    s = sc._social_cost_from_vecs(G, alpha_arr, beta_arr, P)
    sc_beta.append(s)


# Reshape the data for linear regression
X = np.array(beta_e).reshape(-1, 1)
y = np.array(sc_beta)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)
y_intercept = model.intercept_
slope = model.coef_[0]
# Generate predictions using the linear regression model
y_pred = model.predict(X)

# Plot the linear regression line and the data points
plt.plot(beta_e, sc_beta, label="Data", linewidth=2, marker="o")
plt.plot(beta_e, y_pred, color="red", label="Linear Regression", linestyle="--")
plt.xlabel("Beta")
plt.ylabel("Social Cost")
plt.grid()
plt.legend()
print("Slope:", slope)
print("Intercept:", y_intercept)

print("Slope from function:", sc.slope_social_cost(G, P, edge))


# %%
