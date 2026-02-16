# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd


from src import osmGraphs as og
from src import multiCommodityTAP as mc

# %%


cities = [
    "Berlin,Germany",
    "Cologne,Germany",
    "Potsdam,Germany",
    "Heidelberg,Germany",
    "Prenzlauer Berg,Berlin,Germany",
    "Nippes,Cologne,Germany",
]

kappa_max = 200
kappa_step = 10

kappas = range(10, kappa_max, kappa_step)

df = pd.DataFrame(index=kappas)
for city in cities:
    print(city)
    G, districts = og.osmGraph(city, return_districts=True)
    selected_nodes = og.select_evenly_distributed_nodes(G, 30)

    demands, nodes = og.demand_list(
        G,
        commodity=selected_nodes,
    )

    F_list = []
    kappa = range(10, min([len(G.nodes), kappa_max]), kappa_step)
    for k in kappa:
        print(k)
        selected_nodes = og.select_evenly_distributed_nodes(G, k)
        demands, nodes = og.demand_list(
            G,
            commodity=selected_nodes,
        )
        F = mc.solve_multicommodity_tap(
            G, demands, social_optimum=False, max_iter=50_000, eps_rel=1e-5
        )
        F_list.append(F)

    Farr = np.array(F_list)
    F_conv = Farr - Farr[-1, :]

    # Calculate the z-scores of the time series data
    z_scores = np.abs(zscore(F_conv, axis=1))

    # Define a threshold to identify outliers
    threshold = 3

    # Mask outliers with NaN
    F_conv_no_outliers = np.where(z_scores > threshold, np.nan, F_conv)

    # Replace NaNs with the column mean
    col_mean = np.nanmean(F_conv_no_outliers, axis=0)
    inds = np.where(np.isnan(F_conv_no_outliers))
    F_conv_no_outliers[inds] = np.take(col_mean, inds[1])

    var = np.var(F_conv_no_outliers, axis=1)
    if len(var) < len(kappas):
        var = np.append(var, [np.nan] * (len(kappas) - len(var)))

    df[city] = var


df.to_csv("data/commodity_convergence.csv")

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for city in cities:
    var = df[city]

    ax.plot(df.index, var, marker=".", label=city)
    ax.grid()
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\sigma^2(f^{\kappa}_e - f_e)$")
    ax.legend()
# plt.plot(Farr - Farr[-1, :], marker=".")
# plt.grid()
# %%
