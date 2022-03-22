import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

xp_arr = np.loadtxt("xp_arr.txt")
kpp_arr = np.loadtxt("kpp_arr.txt")
tabbed = np.loadtxt("real.txt") + 1j * np.loadtxt("imag.txt")

raw_path = "/mount/citadel1/zz1994/codes/CIBxLIM/tabs/Halo_CIB/1000x2000"
raw = np.loadtxt(raw_path + "/b_dI_dz.txt")
xpp_arr = np.loadtxt(raw_path + "/xpp_arr.txt")



# Manual Comparison
i = 500
kpp = kpp_arr[i]
expo = np.exp(1j*kpp*xpp_arr)
integ = integrate.simps(raw.T * expo, x=xpp_arr, axis=1)
plt.plot(xp_arr, abs(tabbed[i]), label="Tabbed")
plt.plot(xp_arr, abs(integ), "--", label="Integration")
plt.legend()
plt.show()







"""
idx = np.where(kpp_arr >= 0)
kpp_arr = kpp_arr[idx]
for i in [100, 300, 700, 800]:
    #plt.loglog(kpp_arr[idx], abs(CIB[idx, i][0]), label=i)
    plt.plot(xp_arr, CIB[i].imag, label=kpp_arr[i])
plt.legend()
plt.show()
"""
