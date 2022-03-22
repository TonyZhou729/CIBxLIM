import numpy as np
import matplotlib.pyplot as plt

xp_arr = np.loadtxt("xp_arr.txt") # z dim 
xpp_arr = np.loadtxt("xpp_arr.txt") # l dim
b_dI_dz = np.loadtxt("b_dI_dz.txt")

idx = [100, 600, 900]
for i in idx:
    plt.plot(xpp_arr, b_dI_dz[:, i], label="{:.4f} Mpc".format(xp_arr[i]))
plt.xlabel(r"$x_{\parallel}'$")
plt.ylabel(r"$b(z) \times \frac{dI}{dz}\ [Jy]$")
plt.legend()
plt.show()






