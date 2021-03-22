import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as cosmo
from colossus.lss import mass_function
from astropy.cosmology import Planck15 as acosmo

cosmo.setCosmology("planck15")
mfunc = lambda M_h, z : mass_function.massFunction(M_h, z, mdef="200m", model="tinker08")

h = np.array(acosmo.H(0)/100)
print(h)

M = np.linspace(1e2, 1e17, 5000)
plt.loglog(M/h, h**3*mfunc(M, 0))
plt.xlabel(r"Halo Mass $[M_{\odot} h^{-1}]$")
plt.ylabel(r"Halo Function $[h^3 \rm Mpc^{-3}]$")
plt.show()


