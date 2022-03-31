import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
import scipy.constants as const
from scipy import integrate

import util

# Crude model for the CII emission. 
# Quantities included are the Luminosity(from SFR) to Intensity conversion, as well as
# the linear bias.

class CII():
    def __init__(self, z, di=-1):
        self.z = z # Requested Redshift
        self.nu = const.c/ (158 * 1e-6) / 1e9 # Rest frequnecy of CII, converted from rest wavelength of 158 microns.
        self.mh = np.linspace(1e10, 1e16, 10000)
        hmf_func = util.get_hmf_interpolator()
        bias_func = util.get_halo_bias_interpolator()
        
        # Convenient self reference values
        self.hmf = hmf_func(self.z, self.mh)
        self.halo_bias = bias_func(self.z, self.mh)
        self.L = self.Luminosity(di).T
        self.I_mean = np.mean(self.Intensity(di))
        self.b_mean = np.mean(self.bias())

    def Intensity(self, di=-1):
        # Constants
        res = (const.c/1000)/4/np.pi/self.nu/np.array(cosmo.H(self.z)) # Units L_sun * Mpc / GHz    
        rho_L = self.rho_L(di)        
        res *= self.rho_L(di) # Units 1/Mpc^3
        res = res * u.L_sun / u.GHz / u.Mpc**2
        res = res.to(u.Jy)
        res /= u.Jy
        return res

    def rho_L(self, di=-1):
        #plt.loglog(self.mh, self.hmf_model.dn_dm())
        #plt.show()
        #integ = self.Luminosity() * self.hmf_model.dn_dm()
        integ = self.L * self.hmf / self.mh # Last division converts dn/dlnM to dn/dM
        return integrate.simps(integ, x=self.mh)

    def Luminosity(self, di=-1):
        # alpha and beta value taken from Leung 2020 (https://arxiv.org/abs/2004.11912)
        model = "Yang21"
        
        if model == "Leung20":
            alpha = 0.66
            beta = 6.82
        elif model == "Silva15":
            alpha = 1.00
            beta = 6.9647
        elif model == "Chung20":
            alpha = 1.40 - 0.07*self.z
            beta = 7.1 - 0.07*self.z
        elif model == "Schaerer20":
            alpha = 1.02
            beta = 6.90
        elif model == "Yang21":
            alpha = 1.26
            beta = 7.10

        _SFR = util.SFR(self.mh, self.z, di=-1)        
        if di == -1:
            res=10**(alpha*np.log10(_SFR) + beta)
        else:
            _dSFR = util.SFR(self.mh, self.z, di)
            res = alpha * 10**beta * _SFR**(alpha-1) * _dSFR
        return res

    def bias(self):        
        integ1 = self.L * self.halo_bias * (self.hmf / self.mh)
        integ2 = self.L * (self.hmf / self.mh)
        res = integrate.simps(integ1, x=self.mh) / integrate.simps(integ2, x=self.mh)
        return res

    def plot_bias(self):
        #plt.plot(self.mh, self.hmf_model.b_nu())
        plt.plot(self.mh, self.halo_bias)
        plt.xscale("log")
        plt.show()
