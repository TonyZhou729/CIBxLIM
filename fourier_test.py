import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from scipy.interpolate import interp1d
import time

CIB = CIBxLIM()
larr = np.linspace(158*3.5, 158*4.5, 1000)
zarr = np.linspace(CIB.z_min, CIB.z_max, 8000)
#kp = np.linspace(0, 1, 10)
kp = 0.2
ell = 10

chi = lambda z : np.array(cosmo.comoving_distance(z))
l_from_z = lambda z : 158*(1+z) # Convert z to line frequency.
z_from_chi = interp1d(chi(zarr), zarr) # Interpolate z off of comoving distance.

"""
#print(CIB.CIBmodel.CIB_model(larr, zarr).shape)
kpp_arr = np.linspace(0.02, 0.48, 6)
for kpp in kpp_arr:
    plt.plot(zarr, CIB.F_hat(kpp, zarr), label="{:.3f}".format(kpp))
plt.xlabel("z")
plt.ylabel(r"$\hat{F}(k_{\parallel}', z)$")
plt.legend()
plt.show()
"""

start = time.perf_counter()
print(CIB.fourier_Cl(0.1, 20))
print(time.perf_counter() - start)
























"""
xparr1 = np.linspace(1000, 2000, 1000)
xparr2 = np.linspace(6000, 7000, 1000)

kparr1 = np.fft.fftfreq(xparr1.size, d=xparr1[1] - xparr1[0])
kparr2 = np.fft.fftfreq(xparr2.size, d=xparr2[1] - xparr2[0])
#print(kparr1[np.where(kparr1>=0)])
#print(kparr2[np.where(kparr2>=0)])

ll1, zz1 = np.meshgrid(larr, z_from_chi(xparr1))
ll2, zz2 = np.meshgrid(larr, z_from_chi(xparr2))

func1 = CIB.CIBmodel.CIB_model(ll1, zz1)
func2 = CIB.CIBmodel.CIB_model(ll2, zz2)

funck1 = np.fft.fft(func1, axis=-1)
funck2 = np.fft.fft(func2, axis=-1)

plt.plot(kparr1, funck1[:, 3])
plt.plot(kparr2, funck2[:, 3])
plt.show()
"""
