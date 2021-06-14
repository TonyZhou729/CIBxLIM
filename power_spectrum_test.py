from CIBxLIM import *
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from scipy import integrate

kpp = 0.5
ell = 100

CIB = CIBxLIM()
CIB.kp = kpp
CIB.kpp = kpp
CIB.ell = ell
#CIB.Vegas5D(kpp, ell, 2e5)

"""
z = np.linspace(CIB.z_min, CIB.z_max, 1000)
u = np.linspace(3.5, 4.5, 1010)
zz, uu = np.meshgrid(z, u)
func2D = CIB.integrand_Im_simpsons(zz, uu)
print(func2D.shape)
func1D = integrate.simps(func2D, x=z)
print(func1D)
res = integrate.simps(func1D, x=u)
print(res)
"""

### Plotting ###
k = np.linspace(0, 1, 1000)
z = np.linspace(CIB.z_min, CIB.z_max, 1000)
u = np.linspace(3.5, 4.5, 1000)
test_z = [3]
for zt in test_z:
    f = CIB.integrand_Im_simpsons(zt, u)
    plt.plot(u, f, label="z={}".format(zt))
#plt.yscale("log")
plt.legend()
plt.xlabel(r"$u$")
plt.ylabel(r"$Jy\ Mpc^{3/2}$")
plt.show()
