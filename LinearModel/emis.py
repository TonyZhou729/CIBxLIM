import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as const
from astropy.cosmology import Planck15 as cosmo
import time
from astropy.io import fits

"""
Linear CIB model for the emissitivity j. A good approximation to the total Halo model
at large scales (Small k) which is mostly what we're interested in. In the future
this will serve as a good reference to the Halo model.

See Maniyar 2018 for details.
"""
# Comoving distance shorthand in Mpc
chi = lambda z : np.array(cosmo.comoving_distance(z))
# Kennicutt constant in M_sun/yr
K = 1e-10

# Fitting parameters:
alpha = 0.007
beta = 3.590
gamma = 2.453
delta = 6.578

# Star formation rate density function
def rho_SFR(z):
    # In units of M_sun/yr/Mpc^3
    numerator = (1+z)**beta
    denominator = 1+((1+z)/gamma)**delta
    return alpha * numerator / denominator

# SED in Jy L_sun
hdulist = fits.open("../SED.fits")
redshifts = hdulist[1].data
SED = hdulist[0].data[:-1] # Don't include 3000GHz
hdulist.close()

freqs = np.array([100, 143, 217, 353, 545, 857], dtype="float64")

# Emissitivity
def j(nu, z):
    assert np.any(np.isin(freqs, nu)), "Frequency must be one of [100, 143, 217, 353, 545, 857] GHz"
    res = rho_SFR(z) * (1+z) * SED[np.where(freqs==nu)][0] * chi(z)**2 / K
    return res
    
# Test
for f in freqs:
    plt.plot(redshifts, j(f, redshifts), label="{} GHz".format(f))
plt.yscale("log")
plt.xlabel("Redshift z")
plt.ylabel(r"Emissitivity $[Jy L_{\odot}/Mpc]$")
plt.legend()
plt.show()













