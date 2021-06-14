import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.constants as const
from astropy.cosmology import Planck15 as cosmo
import time
from astropy.io import fits
from scipy import interpolate
import astropy.units as u

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

        path = "/mount/citadel1/zz1994/codes/CIBxLIM/SEDdata"
        self.SED = np.loadtxt("{}/SEDtable.txt".format(path))
        self.redshifts = np.loadtxt("{}/SEDredshifts.txt".format(path))
        self.wavelengths = np.loadtxt("{}/SEDwavelengths.txt".format(path))
        self.chi_peaks = np.loadtxt("{}/chi_peaks.txt".format(path))
        
        self.interp_SED = interpolate.RectBivariateSpline(self.wavelengths, self.redshifts, self.SED.T)

        self.interp_chi_peaks = interpolate.interp1d(self.wavelengths,
                                                     self.chi_peaks,
                                                     kind="linear")

    # Star formation density function
    def rho_SFR(self, z):
        # In units of M_sun/yr/Mpc^3
        numerator = (1+z)**self.beta
        denominator = 1+((1+z)/self.gamma)**self.delta
        return self.alpha * numerator/denominator

    # Emissitivity, output shape should be the same as l and z. 
    def j(self, l, z):
        res = self.rho_SFR(z) * (1+z) * self.interp_SED(l, z, grid=False) * self.chi(z)**2 / self.K
        return res

    def CIB_model(self, l, z):
        # Derivative of CIB intensity w.r.t. redshift, input into the powerspectrum calculation.
        c = const.c/1000 # Speed of light in units km/s
        a = 1/(1+z)
        res = c/np.array(cosmo.H(z)) * a * self.j(l, z)
        return res
        

    def plot_emissitivity(self, z, freqs=None, wavelengths=None, freq_unit=u.GHz, wave_unit = u.um, normal=False):
        # Plot the emissitivity j for given frequncy or array of frequencies.
        # Default unit is GHz for frequency and micrometer for wavelength. 
        if freqs is None:
            assert wavelengths is not None, "Must input either frequency or wavelengths"
            wavelengths *= wave_unit
            wavelengths = wavelengths.to(u.um)
        else:
            freqs *= freq_unit
            freqs = freqs.to(u.Hz)
            c = const.c * u.m * u.Hz
            wavelengths = c/freqs
            wavelengths = wavelengths.to(u.um)
        
        wavelengths = np.array(wavelengths)
        for l in wavelengths:
            j_func = self.j(l, z)
            if normal:
                plt.plot(z, j_func/np.trapz(y=j_func, x=z), label="{:.3f} um".format(l))
            else:
                plt.plot(z, j_func, label="{:.3f} um".format(l))
        #plt.yscale("log")
        plt.xlabel("Redshift z")
        plt.ylabel(r"Emissitivity $[\rm Jy L_{\odot}/\rm Mpc]$")
        plt.legend()
        plt.show()

    def plot_CIB_model(self, z, freqs=None, wavelengths=None, freq_unit=u.GHz, wave_unit = u.um, normalize=True, logx=False, logy=False):
        # Plot the CIB intensity for given frequncy or array of frequencies.
        # Default unit is GHz for frequency and micrometer for wavelength. 
        if freqs is None:
            assert wavelengths is not None, "Must input either frequency or wavelengths"
            wavelengths *= wave_unit
            wavelengths = wavelengths.to(u.um)
        else:
            freqs *= freq_unit
            freqs = freqs.to(u.Hz)
            c = const.c * u.m * u.Hz
            wavelengths = c/freqs
            wavelengths = wavelengths.to(u.um)
        
        wavelengths = np.array(wavelengths)
        for l in wavelengths:
            CIB_func = self.CIB_model(l, z)
            if normalize:
                plt.plot(z, CIB_func/np.trapz(y=CIB_func, x=z), label="{:.3f} um".format(l))
            else:
                plt.plot(z, CIB_func, label="{:.3f} um".format(l))
        if logx:
            plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.xlabel("Redshift z")
        plt.ylabel(r"$\frac{dI_{\lambda}}{dz} [Jy]$")
        plt.legend()
        plt.show()
