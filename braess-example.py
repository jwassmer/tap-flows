# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

from src import Graphs as gr
from src import Plotting as pl
from src import TAPOptimization as tap
from src import SocialCost as sc

pl.mpl_params(fontsize=14)


def social_cost_e(G, f):
    f = np.array(f)
    alpha = nx.get_edge_attributes(G, "alpha")
    beta = nx.get_edge_attributes(G, "beta")

    alpha_arr = np.array(list(alpha.values()))
    beta_arr = np.array(list(beta.values()))

    return alpha_arr * f**2 + beta_arr * f


def compare_effective_flow(f, feff, eps=1e-5):
    # some keys will be missing in feff. add them and set to zero
    for key in f.keys():
        if key not in feff:
            feff[key] = 0

    diff = {}
    for k, v in feff.items():
        diff[k] = v - f[k]

    max_diff = max(np.abs(list(diff.values())))

    return max_diff


# %%
load = 100
G = gr.triangularLattice(1, alpha=1e-1, beta="random")

P = np.zeros(G.number_of_nodes())
P[0] = load
P[-1] = -load

f0 = tap.user_equilibrium(G, P, positive_constraint=True)
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = plt.Normalize(1e-1, max(f0))

pl.graphPlot(G, ec=f0, cmap=cmap, norm=norm)


# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
betas = np.linspace(0, 20, num=100)
labels = ["so_p", "ue_p", "so_m", "ue_m"]

edge = (3, 4)

sc_df = pd.DataFrame(index=betas, columns=labels)
for beta in betas:
    G.edges[edge]["beta"] = beta

    f_so_p = tap.social_optimum(G, P, positive_constraint=True)
    f_ue_p = tap.user_equilibrium(G, P, positive_constraint=True)
    f_so_m = tap.social_optimum(G, P, positive_constraint=False)
    f_ue_m = tap.user_equilibrium(G, P, positive_constraint=False)

    scs = []
    for f in [f_so_p, f_ue_p, f_so_m, f_ue_m]:

        social_cost = tap.social_cost(G, f)
        scs.append(social_cost)
    sc_df.loc[beta] = scs


# ax.plot(betas, sc_df["so_p"], label="SO Positive", linewidth=10, color="#abd9e9")
# ax.plot(betas, sc_df["ue_p"], label="UE Positive", linewidth=5, color="#2c7bb6")
ax.plot(betas, sc_df["so_m"], label="SO Negative", linestyle="--", color="#d7191c")
ax.plot(betas, sc_df["ue_m"], label="UE Negative", linestyle="--", color="#fdae61")

ax.set_xlabel("Beta")
ax.set_ylabel("Social Cost")


# Prepare the data
data = sc_df["ue_p"]
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

# ax.plot(
#    X_pred,
#    y_pred,
#    color="black",
#    label=f"Linear Fit, m={round(slope,2)}",
#    linestyle="-",
#    linewidth=1,
# )
ax.grid()
ax.legend()


# %%


load = 100

G = gr.random_graph(20, alpha="random", beta="random")
P = np.zeros(G.number_of_nodes())
P[0] = load
sinks = np.delete(np.arange(G.number_of_nodes()), 0)
P[sinks] = -sum(P) / len(sinks)
fp = tap.user_equilibrium(G, P)
fp_dict = dict(zip(G.edges, fp))

Gs = gr.flow_subgraph(G, dict(zip(G.edges, fp)), eps=1e-5)
fs = tap.linearTAP(Gs, P)[0]


diff = compare_effective_flow(fp_dict, dict(zip(Gs.edges, fs)))
print(diff)

cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = plt.Normalize(1e-3, max(fp))


pl.graphPlot(Gs, ec=fs, cmap=cmap, norm=norm, show_labels=True)

beta_derivatives = sc.all_social_cost_derivatives(G, P)


for e in G.edges:
    sc_beta = beta_derivatives[e]
    fval = fp_dict[e]

    if fval > 1e-3 and sc_beta < 0:
        print(e, sc_beta, fval)


sc_list = []

betas = np.linspace(0, 100, num=50)

edge = (14, 18)
m = sc.linreg_slope_sc(G, P, edge)


for beta in betas:
    G.edges[edge]["beta"] = beta
    f = tap.user_equilibrium(G, P, positive_constraint=True)
    sc_list.append(tap.social_cost(G, f))

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
ax.plot(betas, sc_list)
ax.grid()


# %%


edge = (17, 59)
betas = np.linspace(0, 5, num=50)

social_cost_list = []
eff_social_cost_list = []

Gs = gr.flow_subgraph(G, dict(zip(G.edges, fp)))

# fig, ax = plt.subplots(len(betas), 1, figsize=(5, 30), sharey=True)
for beta in betas:
    Gs.edges[edge]["beta"] = beta
    G.edges[edge]["beta"] = beta

    f = tap.user_equilibrium(G, P)
    fs = tap.linearTAP(Gs, P)[0]

    all_fs_pos = np.all(fs > 0)
    print(all_fs_pos)

    f_dict = dict(zip(G.edges, f))
    fs_dict = dict(zip(Gs.edges, fs))

    diff = compare_effective_flow(f_dict, fs_dict)
    print(diff)

    social_cost_list.append(tap.social_cost(G, f))
    eff_social_cost_list.append(tap.social_cost(Gs, fs))

    # pl.graphPlot(G, ec=fp, cmap=cmap, norm=norm, ax=ax[betas.tolist().index(beta)])


# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
ax.plot(betas, social_cost_list)
ax.plot(betas, eff_social_cost_list)
ax.set_xlabel("Beta")
ax.set_ylabel("Social Cost")
ax.grid()
# %%


# %%
