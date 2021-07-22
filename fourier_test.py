import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from scipy.interpolate import interp1d
import time
from numba import njit

CIB = CIBxLIM()
larr = np.linspace(158*3.5, 158*4.5, 1000)
zarr = np.linspace(CIB.z_min, 5, 1000)
#kp = np.linspace(0, 1, 10)
kpp = 0.003
ell = 10

chi = lambda z : np.array(cosmo.comoving_distance(z))
l_from_z = lambda z : 158*(1+z) # Convert z to line frequency.
z_from_chi = interp1d(chi(zarr), zarr) # Interpolate z off of comoving distance.

#print(CIB.V_tilde(kpp, kpp, ell))

cls50 = np.loadtxt("cls50.txt")
cls100 = np.loadtxt("cls.txt")
cls200 = np.loadtxt("cls200.txt")

ell=np.arange(200)
plt.loglog(ell, cls50[0], label="N=50")
plt.loglog(ell, cls100[0, :200], label="N=100")
plt.loglog(ell, cls200[0], label="N=200")
plt.legend()
plt.show()


"""
# Making Cls
ell = np.arange(200)
kpp, cls = CIB.fourier_Cl(ell)
np.savetxt("cls50.txt", cls)
#np.savetxt("kpp.txt", kpp)


for i, cl in enumerate(cls):
    plt.loglog(ell, cl, label="{:.4f}".format(kpp[i]))
plt.legend()
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_{\ell}$")
plt.show()
"""

"""
# F_hat test
start = time.perf_counter()
#kval, f1 = CIB.F_hat(kpp, zarr)
#karr = np.loadtxt("FHATdata/k.txt")
#func_k = np.loadtxt("FHATdata/FHAT.txt", dtype=np.complex_)
#kval = karr[500]
#f1 = func_k[500]
print(time.perf_counter() - start)

start = time.perf_counter()
f2 = CIB.F_hat_int2(kval, zarr)
print(time.perf_counter() - start)

print("k value is {}".format(kval))

plt.plot(zarr, abs(f1), label="FFT")
plt.plot(zarr, abs(f2), "--", label="Simpsons")

plt.xlabel("z")
plt.ylabel(r"$\hat{F}$")
plt.title(r"$k_{\parallel}' = $" + "{:.4f} 1/Mpc".format(kval))
plt.legend()
plt.show()
"""








