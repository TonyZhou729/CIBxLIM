import numpy as np
import matplotlib.pyplot as plt
import emis
from astropy.cosmology import Planck15 as cosmo
import scipy.constants as const

LinearModel = emis.LinearModel() # Linear CIB emissitivity model from Maniyar 2018
z = LinearModel.redshifts # Effective redshift points from the SED data.
c = const.c/1000 # Speed of light in units km/s

def CIB_model(nu, z):
    # Derivative of CIB intensity w.r.t. redshift, input into the powerspectrum calculation.
    a = 1/(1+z)
    res = c/np.array(cosmo.H(z)) * a * LinearModel.j(nu, z)
    return res

for f in LinearModel.freqs:
    plt.plot(z, CIB_model(f, z), label="{} GHz".format(f))
plt.legend()
plt.xlabel("Redshift z")
plt.ylabel(r"$dI_{\nu}/dz [\rm Jy]$")
plt.yscale("log")
plt.show()
