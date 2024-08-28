# %%

from src import osmGraphs as og

import numpy as np
import matplotlib.pyplot as plt

# %%

alpha = 0.15
beta = 4
c = 60

bpr = lambda x: t_min * (1 + alpha * (x / c) ** beta)
teff = lambda x: (gamma * tr * l * x) / (l * m - gamma * d * x)


G = og.osmGraph("Heidelberg,Germany")

# %%

edge = list(G.edges(data=True))[3][-1]
l, m, v = edge["length"], edge["lanes"], edge["speed_kph"] / 3.6
gamma, tr, d = 1, 2, 5
walking_speed = 1.4
t_max = l / walking_speed
t_min = l / v


eff_func = og.effective_travel_time(edge)
linear_func = og.linear_function(edge)
alpha = round(linear_func(1) - linear_func(0), 1)
beta = round(linear_func(0), 1)
# potential_energy_func = potential_energy(edge)


xmax = edge["xmax"]
xmin = edge["xmin"]
# beta = edge["beta"]


# Generate L values
x_values = np.linspace(0, int(np.ceil(xmax * 1.2)), 100)

bpr_values = bpr(x_values)  # [bpr(x) for x in x_values]
t_eff_values = [eff_func(x) for x in x_values]
linear_values = [linear_func(x) for x in x_values]


# Plot the original function and the linear function
plt.figure(figsize=(10, 6))
# plt.plot(x_values, bpr_values, label="BPR Function", color="Black", linewidth=2)
plt.plot(
    x_values,
    t_eff_values,
    label="Daganzo model $c_{D, \mathrm{eff}}(f_{e})$",
    color="blue",
)
plt.plot(
    x_values,
    linear_values,
    label=rf"Linear $c(f_e) = {alpha}[s] f_e + {beta} [s]$",
    color="red",
    linestyle="--",
)

plt.axhline(y=t_max, color="green", linestyle="-.", label="$t_{\\mathrm{max}}$")
plt.axhline(y=t_min, color="orange", linestyle="-.", label="$t_{\\mathrm{min}}$")
plt.axvline(x=xmax, color="green", linestyle="-.")
plt.axvline(x=xmin, color="orange", linestyle="-.")
plt.xlabel("$f_{e}$")
plt.ylabel("$c_{D, \mathrm{eff}}(f_{e}) [s]$")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# %%


c = 0.5  # assuming a capacity for demonstration purposes

x = np.linspace(0, 20, 100)  # avoid division by zero at high x

y_teff = teff(x)
y_bpr = bpr(x)

# Plot both functions
plt.figure(figsize=(10, 6))
plt.plot(x, y_teff, label="Given Function teff(x)", color="blue")
plt.plot(x, y_bpr, label="BPR Function", color="red", linestyle="dashed")
plt.xlabel("Traffic Flow (x)")
plt.ylabel("Travel Time")
plt.title("Comparison of Given Function and BPR Function")
plt.legend()
plt.grid(True)
plt.show()

# %%
