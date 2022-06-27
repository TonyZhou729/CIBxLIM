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
    def __init__(self, z, di=-1, Lmodel="Yang21", params=[0.42, 12.94, 1.75, 1.17, 0, 0]):
        self.z = z # Requested Redshift
        self.nu = const.c/ (158 * 1e-6) / 1e9 # Rest frequnecy of CII, converted from rest wavelength of 158 microns.
        self.mh = np.linspace(1e10, 1e16, 10000)
        self.Lmodel = Lmodel
        self.params = params
        hmf_func = util.get_hmf_interpolator()
        bias_func = util.get_halo_bias_interpolator()

        # Convenient self reference values
        self.hmf = hmf_func(self.z, self.mh) * np.log(10) # Coverts dn/dlnM to dn/dlog10M
        self.halo_bias = bias_func(self.z, self.mh)        
        if di == -1:
            self.factor = self.Intensity(di) * self.bias(di)
        else:            
            self.factor = self.bias(di=-1) * self.Intensity(di=di) + self.Intensity(di=-1) * self.bias(di=di)
            #self.factor = self.bias(di=-1) * self.Intensity(di=di)
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
        integ = self.Luminosity(di=di).T * self.hmf / self.mh # Last division converts dn/dlog10M to dn/dM
        return integrate.simps(integ, x=self.mh)

    def Luminosity(self, di=-1):
        model = self.Lmodel
        
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
        else:
            alpha = self.params[4]
            beta = self.params[5]
        
        _SFR = util.SFR(self.mh, self.z, di=-1, params=self.params[:4])        
        if di == -1:
            #res=10**(alpha*np.log10(_SFR) + beta)
            res = 10**beta * _SFR**alpha
        elif di < 4: # Halo SFR parameters.
            _dSFR = util.SFR(self.mh, self.z, di)            
            res = alpha * 10**beta * _SFR**(alpha-1) * _dSFR
        elif di == 4: # Power law parameter alpha.
            res = 10**beta * _SFR**alpha * np.log(_SFR)
        else: # Power law parameter beta.
            res = 10**beta * _SFR**alpha * np.log(10)
        return res

    def bias(self, di):        
        integ1 = self.Luminosity(di=-1).T * self.halo_bias * (self.hmf / self.mh)
        integ2 = self.Luminosity(di=-1).T * (self.hmf / self.mh)
        I1 = integrate.simps(integ1, x=self.mh)
        I2 = integrate.simps(integ2, x=self.mh)
        if di == -1:
            res = I1 / I2
        else: # Need derivatives            
            dinteg1 = self.Luminosity(di=di).T * self.halo_bias * (self.hmf / self.mh)
            dinteg2 = self.Luminosity(di=di).T * (self.hmf / self.mh)
            dI1 = integrate.simps(dinteg1, x=self.mh)
            dI2 = integrate.simps(dinteg2, x=self.mh)            
            res = (I2 * dI1 - I1 * dI2) / I2**2 # Good'ol Quotient rule!
        return res

    def plot_bias(self):
        #plt.plot(self.mh, self.hmf_model.b_nu())
        plt.plot(self.mh, self.halo_bias)
        plt.xscale("log")
        plt.show()
