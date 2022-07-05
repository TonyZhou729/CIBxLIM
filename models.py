import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
#from astropy.io import fits
import astropy.units as u
import scipy.constants as const
from scipy.integrate import simps
from scipy import interpolate
import time
import util

path = "/mount/citadel1/zz1994/codes/CIBxLIM" # Enter CIBxLIM directory here. 
KC = 1e-10 # Kennicutt Constant

class LinearModel():
    ### THE LINEAR MODEL IS OBSOLETE ###
    """
    Linear CIB model for the emissitivity j. A good approximation to the total Halo model
    at large scales (Small k) which is mostly what we're interested in. In the future
    this will serve as a good reference to the Halo model.

    See Maniyar 2018 for details.
    """

    def __init__(self):        
        # Fitting parameters:
        self.alpha = 0.007
        self.beta = 3.590
        self.gamma = 2.453
        self.delta = 6.578

        SEDpath = path + "/tabs/SEDdata"
        self.SED = np.loadtxt("{}/SEDtable.txt".format(path))
        self.redshifts = np.loadtxt("{}/SEDredshifts.txt".format(path))
        self.wavelengths = np.loadtxt("{}/SEDwavelengths.txt".format(path))
        self.chi_peaks = np.loadtxt("{}/chi_peaks.txt".format(path))
        
        self.interp_SED = interpolate.RectBivariateSpline(self.wavelengths, self.redshifts, self.SED.T)        

    # Star formation density function
    def rho_SFR(self, z):
        # In units of M_sun/yr/Mpc^3
        numerator = (1+z)**self.beta
        denominator = 1+((1+z)/self.gamma)**self.delta
        return self.alpha * numerator/denominator

    # Emissitivity, output shape should be the same as l and z. 
    def b_j(self, l, z):
        res = self.rho_SFR(z) * (1+z) * self.interp_SED(l, z, grid=True) * util.chi(z)**2 / K

        return res

    def b_dI_dz(self, l, z):
        # Derivative of CIB intensity w.r.t. redshift, input into the powerspectrum calculation.
        c = const.c/1000 # Speed of light in units km/s
        a = 1/(1+z)
        res = c/np.array(cosmo.H(z)) * a * self.b_j(l, z)
        return res
        

class HaloModel():
    
    def __init__(self):
        SEDpath = path + "/tabs/SEDdata"
        self.chi_peaks = np.loadtxt("{}/chi_peaks.txt".format(SEDpath))
        self.SED = np.loadtxt("{}/SEDtable.txt".format(SEDpath))
        self.redshifts = np.loadtxt("{}/SEDredshifts.txt".format(SEDpath))
        self.wavelengths = np.loadtxt("{}/SEDwavelengths.txt".format(SEDpath))                                                     
        self.interp_SED = interpolate.RectBivariateSpline(self.wavelengths, self.redshifts, self.SED.T)
        self.fsub = 0 # Assume simple model with no sub halo. 
        self.hmf = util.get_hmf_interpolator() * np.log(10) # dn/dlog10M, Inputs are z, mh
        self.bias = util.get_halo_bias_interpolator() # Halo Bias, Inputs are z, mh

    # Center halo emissitivity
    def djc_dlogM(self, l, mh, z, di=-1): 
        res = np.zeros((l.size, mh.size, z.size), dtype="float64")
        mheff = mh * (1-self.fsub) # Effective central halo mass fraction. 
        hmf_part = self.hmf(z, mh).T # Takes full halo mass, not just the central. Shape is (mh, z)           
        # All parts not involving the wavelength dimension within SED.
        # Equation 12 in Maniyar 2020 without the last S_nu_eff term.
        rest = hmf_part * util.SFR(mheff, z, di) * (1+z) * util.chi(z)**2 / KC # Shape is (l, z)
        
        # Loop through wavelengths for SED. 
        SED = self.interp_SED(l, z)
        for i in range(l.size):
            res[i, :, :] = rest * SED[i, :]
        return res
    
    # Halo Bias x Emissitivity, intergrated over halo masses
    def b_j(self, l, z, di=-1):
        logmh = np.linspace(10, 18, 3000) # Range will cover all possible derivatives.
        mh = 10**logmh
        djc_dlogM = self.djc_dlogM(l, mh, z, di)        
        integrand = self.bias(z, mh).T * djc_dlogM # Shape is (l, mh, z)        
        res = simps(integrand, x=logmh, axis=1) # Integrate over mh, shape is (l, z)
        return res

    # Input to Cl calculation, b x dI/dz
    # Optionally specify di = 0, 1, 2, 3, to obtain derivative w.r.t. halo model params.
    # Parameters are (eta_max, log(M_max), sigma_Mh0, tau), see util.py for specifics.
    def b_dI_dz(self, l, z, di=-1):
        c = const.c/1000
        a = 1/(1+z) # Scale factor
        res = c/cosmo.H(z).value * a * self.b_j(l, z, di)
        return res




