import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as const
from astropy.cosmology import Planck15 as cosmo
import time
from astropy.io import fits


class LinearModel():
    """
    Linear CIB model for the emissitivity j. A good approximation to the total Halo model
    at large scales (Small k) which is mostly what we're interested in. In the future
    this will serve as a good reference to the Halo model.

    See Maniyar 2018 for details.
    """

    def __init__(self):

        # Comoving distance shorthand in Mpc
        self.chi = lambda z : np.array(cosmo.comoving_distance(z))
        # Kennicutt constant in M_sun/yr
        self.K = 1e-10

        # Fitting parameters:
        self.alpha = 0.007
        self.beta = 3.590
        self.gamma = 2.453
        self.delta = 6.578

        # SED in Jy L_sun
        hdulist = fits.open("../SED.fits")
        self.redshifts = hdulist[1].data
        self.SED = hdulist[0].data # Don't include 3000GHz
        hdulist.close()

        self.freqs = np.array([100, 143, 217, 353, 545, 857, 3000], dtype="float64")
    
    # Star formation density function
    def rho_SFR(self, z):
        # In units of M_sun/yr/Mpc^3
        numerator = (1+z)**self.beta
        denominator = 1+((1+z)/self.gamma)**self.delta
        return self.alpha * numerator/denominator

    # Emissitivity
    def j(self, nu, z):
        assert np.any(np.isin(self.freqs, nu)), "Frequency must be one of [100, 143, 217, 353, 545, 857] GHz"
        res = self.rho_SFR(z) * (1+z) * self.SED[np.where(self.freqs==nu)][0] * self.chi(z)**2 / self.K
        return res

    def plot_emissitivity(self, freqs=None, normal=False):
        if freqs == None:
            freqs = self.freqs
        # Plot the emissitivity functions
        for f in freqs:
            j_func = self.j(f, self.redshifts)
            if normal:
                plt.plot(self.redshifts, j_func/np.trapz(y=j_func, x=self.redshifts), label="{} GHz".format(f))
            else:
                plt.plot(self.redshifts, self.j(f, self.redshifts), label="{} GHz".format(f))
        plt.yscale("log")
        plt.xlabel("Redshift z")
        plt.ylabel(r"Emissitivity $[\rm Jy L_{\odot}/\rm Mpc]$")
        plt.legend()
        plt.show()


