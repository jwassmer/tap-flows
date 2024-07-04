# %%
import networkx as nx
import numpy as np
import osmnx as ox

np.set_printoptions(suppress=True, precision=3)
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import tqdm

from src import TAPOptimization as tap
from src import Plotting as pl
from sklearn.linear_model import LinearRegression
from src import SocialCost as sc

from src import osmGraphs as og

pl.mpl_params(fontsize=14)


def interactive_mesh_plot_sc(alphas, betas, dfs):
    # Assuming the variables betas, alphas, and sc_df are defined as follows for demonstration:
    X, Y = np.meshgrid(betas, alphas)
    # Z = sc_ue_df.values.astype(float)
    fig = go.Figure()
    opacity = 1
    legend = True
    cmaps = ["Cividis", "Viridis", "Reds", "Blues"]
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    for i, df in enumerate(dfs):
        Z1 = df.values.astype(float)
        # Z2 = sc_ue_df.values.astype(float)
        # Z3 = sc_lap_df.values.astype(float)

        # Create labels for each surface plot
        # labels1 = np.array([f"SO: {z:.2f}" for z in Z1.flatten()]).reshape(Z1.shape)
        # labels2 = np.array([f"UE: {z:.2f}" for z in Z2.flatten()]).reshape(Z2.shape)
        # labels3 = np.array([f"LAP: {z:.2f}" for z in Z3.flatten()]).reshape(Z3.shape)

        # Creating the interactive plot using Plotly

        if i >= 1:
            opacity = 0.8
            legend = False
        fig.add_trace(
            go.Surface(
                z=Z1,
                x=Y,
                y=X,
                colorscale=cmaps[i],
                opacity=opacity,
                showscale=legend,
                # name="Social Optimum",
                # text=labels1,
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


def social_cost_alpha_beta_df(
    G,
    edge,
    alphas=(1 / np.linspace(1, 100, 25)),
    betas=np.linspace(-500, 50, 25),
    positive_constraint=False,
):

    alphas = alphas[alphas != 0]

    sc_so_df = pd.DataFrame(index=alphas, columns=betas)
    sc_ue_df = pd.DataFrame(index=alphas, columns=betas)
    # sc_lap_df = pd.DataFrame(index=alphas, columns=betas)
    for a in tqdm.tqdm(alphas):
        for b in betas:
            G.edges[edge]["alpha"] = a
            G.edges[edge]["beta"] = b
            if not positive_constraint:
                f_ue, lamb_ue = tap.linearTAP(G, P)
                f_so, lamb_so = tap.linearTAP(G, P, social_optimum=True)
            else:
                f_ue = tap.user_equilibrium(
                    G, P, positive_constraint=positive_constraint
                )
                f_so = tap.social_optimum(G, P, positive_constraint=positive_constraint)
            sc_so = tap.social_cost(G, f_so)
            sc_ue = tap.social_cost(G, f_ue)
            # sc_lap = tap.social_cost(G, f_lap)
            sc_ue_df.loc[a, b] = sc_ue / load
            sc_so_df.loc[a, b] = sc_so / load
            # sc_lap_df.loc[a, b] = sc_lap / load

    return sc_ue_df, sc_so_df


# %%

G = tap.random_graph(
    seed=42,
    num_edges=25,
    num_nodes=20,
    alpha=1,
    beta=1,
)

# G = braessGraph()

E = -nx.incidence_matrix(G, oriented=True).toarray()
P = np.zeros(G.number_of_nodes())
load = 500
source = 15
P[source] = load
targets = np.delete(np.arange(G.number_of_nodes()), source)
P[targets] = -load / len(targets)
# %%

f_ue = tap.user_equilibrium(G, P, positive_constraint=True)
f_, lamb_ue = tap.linearTAP(G, P)
print(tap.social_cost(G, f_) / load)

pl.graphPlotCC(G, cc=f_ue)  # , edge_labels=dict(zip(G.edges, f_ue)))

# %%
edge = (19, 11)
alphas = 1 / np.linspace(1, 100, 25)
betas = np.linspace(-1e2, 1e2, 25)
sc_ue_df, sc_so_df = social_cost_alpha_beta_df(
    G, edge, alphas, betas, positive_constraint=False
)
interactive_mesh_plot_sc(alphas, betas, [sc_so_df, sc_ue_df])


# %%
G = og.osmGraph("Potsdam,Germany")
# %%
nodes, edges = ox.graph_to_gdfs(G)
alpha_arr = np.array(edges["alpha"])
beta_arr = np.array(edges["beta"])
P = og.demands(G, 3)[0]

derivatives = sc.all_social_cost_derivatives(G, P)
f, lamb = tap.linearTAP(G, P)
fdict = dict(zip(G.edges, f))
# %%
edge = max(derivatives, key=derivatives.get)
derivatives[edge]
# fdict[edge]
# %%
x, y = edges.loc[edge]["alpha"], edges.loc[edge]["beta"]
alphas = np.linspace(x / 2, 2 * x, 25)
betas = np.linspace(y / 2, 2 * y, 25)

# df_ue, df_so = social_cost_alpha_beta_df(G, edge, alphas, betas)
# interactive_mesh_plot_sc(alphas, betas, [sc_so_df, sc_ue_df])
# %%
sc.linreg_slope_sc(G, P, edge)

# %%
derivatives[edge]
# %%


tt_fs = nx.get_edge_attributes(G, "tt_function")

beta_e = np.linspace(-10, 1e1, 5)

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
# y_intercept = model.intercept_
m = model.coef_[0]
m

# %%

plt.plot(X, y)
# %%
