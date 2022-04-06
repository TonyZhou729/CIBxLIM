import numpy as np
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import scipy.constants as const
from scipy.integrate import simps
import util

class Fisher_calculator():

    """
    Object for computing Fisher Matrix of (I_CIB + I_CII) x g formalism for SFR parameters.
    Parameters are various Cl objects relevant to the covariance inputs.

    Params:
    ------
    dCl_dpi : Cl_Ig derivatives w.r.t. SFR parameters, shape should be (4, ell_size, k_size)
    Cl_Ig : Main Intensity cross Galaxy power spectrum.
    Cl_II : Intensity auto power spectrum, equal to CIBxCIB + 2CIBxCII + CIIxCII.
    Cl_gg : Galaxy auto power spectrum.    
    """

    def __init__(self, dCl_dpi, Cl_Ig, Cl_II, Cl_gg, ell, kp):
        # Needed Power Spectra
        self.dCl_dpi = dCl_dpi
        self.Cl_Ig = Cl_Ig
        self.Cl_II = Cl_II
        self.Cl_gg = Cl_gg
        self.kp = kp
        self.ell = ell

        # Survey Params
        self.fsky = 0.007
        self.R = 512
        self.theta = 1e-3
        self.P_N = 4.58e11 # Jy^2/sr^2 x (Mpc/h)^3, intensity noise power spectrum        
        #self.P_N = 0
        self.n_g = 3e-6 # (h/Mpc)^3, galaxy number density.
        #self.n_g = 4.73e-5
        #self.n_g=100000000
        self.z_cen = 3 # Central redshift for survey window
        self.L = util.chi(3.5) - util.chi(2.5)

        self.delta_xp = (const.c / 1000) * (1+self.z_cen) / cosmo.H(self.z_cen).value / self.R
        self.V_surv = 4 * np.pi * self.fsky * self.L

    def W(self, ell, kp):
        ell_mesh, k_mesh = np.meshgrid(ell, kp) # Shape is (k, ell)      
        para = -(self.delta_xp)**2 * k_mesh**2 
        perp = -(self.theta)**2 * ell_mesh**2
        return np.exp(para+perp).T # Transpose to (ell, k)

    def input_cov(self):
        cross_part = abs(self.Cl_Ig)**2        
        I_part = self.Cl_II + self.P_N / util.chi(self.z_cen)**2 / self.W(self.ell, self.kp) 
        g_part = self.Cl_gg + 1 / util.chi(self.z_cen)**2 / self.n_g                
        return cross_part + I_part * g_part

    def input_cov2(self):
        t1 = abs(self.Cl_Ig)**2 
        t2 = self.Cl_II * self.Cl_gg
        t3 = self.Cl_II / util.chi(3)**2 / self.n_g
        t4 = self.Cl_gg * self.P_N / util.chi(3)**2 / self.W(self.ell, self.kp)
        t5 = self.P_N / util.chi(3)**4 / self.W(self.ell, self.kp) / self.n_g
        return t1+t2+t3+t4+t5    

    def get_Fisher(self):
        n = self.dCl_dpi[:, 0, 0].size # Number of parameters
        #n = 3
        res = np.zeros((n, n), dtype="float64")
        input_cov = self.input_cov()        
        for i in range(n):
            for j in range(i, n):
                #integ = 2 * (self.dCl_dpi[i] * self.dCl_dpi[j]).real / input_cov
                integ = (self.dCl_dpi[i] * self.dCl_dpi[j].conj()) + (self.dCl_dpi[i].conj() * self.dCl_dpi[j])
                integ = integ.real / input_cov                
                #integ = integ.real * np.linalg.inv(input_cov)
                #integ = integ.T
                integ = integ.T * self.ell * 2 * np.pi # Approx d^2_ell as 2π x ell d_ell, shape is now (k, ell)                
                integ1D = simps(integ, x=self.ell, axis=1)
                integ2D = simps(integ1D, x=self.kp)
                res[i, j] = integ2D
                res[j, i] = res[i, j]
        res *= self.V_surv / (2*np.pi)**3 / 2
        return res


