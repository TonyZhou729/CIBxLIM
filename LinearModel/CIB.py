import numpy as np
import matplotlib.pyplot as plt
import emis
from astropy.cosmology import Planck15 as cosmo
import scipy.constants as const
from halomod.bias import Tinker10
from colossus.lss import bias
from colossus.cosmology import cosmology as ccosmo
from colossus.lss import peaks

LinearModel = emis.LinearModel() # Linear CIB emissitivity model from Maniyar 2018
z = LinearModel.redshifts # Effective redshift points from the SED data.
c = const.c/1000 # Speed of light in units km/s

def CIB_model(nu, z):
    # Derivative of CIB intensity w.r.t. redshift, input into the powerspectrum calculation.
    a = 1/(1+z)
    res = c/np.array(cosmo.H(z)) * a * LinearModel.j(nu, z)
    return res

#LinearModel.plot_emissitivity(normal=True)

for f in LinearModel.freqs:
    model = CIB_model(f,z)
    #plt.plot(z, CIB_model(f, z), label="{} GHz".format(f))
    plt.plot(z, model/np.trapz(model, x=z), label="{} GHz".format(f))    
plt.legend()
plt.xlim((0, 5))
plt.xlabel("Redshift z")
plt.ylabel(r"$dI_{\nu}/dz [\rm Jy]$")
#plt.yscale("log")
plt.show()

