# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("save_n835_od100.pkl", "rb") as f:
    res = pickle.load(f)

slope_edge_dict = res["slope_edge_dict"]
slopes_arr = np.array([*slope_edge_dict.values()])
sorted_arr = np.sort(slopes_arr)
# %%
print(res["f_mat"].shape)
print(len(slope_edge_dict))
# %%
plt.hist(slopes_arr, bins=200)
plt.xlabel("Slope")
plt.ylabel("Hist")
# %%
plt.plot(sorted_arr, np.linspace(0, 1, len(sorted_arr)))
plt.xlabel("Slope")
plt.ylabel("Cummulative propability")
# %%
print("Fraction Braess", (slopes_arr < 0).mean())
print("Time num", round(res["t_num"] / 60, 2), "min")
print("Time slopes", round(res["t_inv"] / 60, 2), "min")

# %%
