import numpy as np
import scipy.constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
from models import LinearModel, HaloModel
from CII import CII
from tqdm import tqdm

from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import time
import util
import multiprocessing as mp

"""
Unit System:
- Wavelength: micrometers
- Physical distance: Mpc
- H0: km/s/Mpc
- CIB Intensity: Jy
- Wavenumber: 1/Mpc
"""


class CIBxLIM():

    """
    Object for computing the emission line dependent CIB power spectrum at values of covariant
    k modes and angular l modes.

    Params:
    ------
    (str) exp : Experiment name to determine the redshift range for the target line. Currently only EXCLAIM supported.
    (str) target_line : Target line with defined restframe wavelength. Currently only CII at 158 micron supported. 
    (str) target_line_rest_wavelength : Manually input an arbitrary line's rest wavelength. Must be given if target_line is None. Default None.
    (boolean) want_function_of_K : True if output Cl is function of k_{\parallel}' at discrete values of ell. Else is function of ell at k_{\parallel}'. Default False 
    """
    def __init__(self,
                 want_halo_model=False,
                 linear_matter_power=False,
                 tab_CIB=True):

        self.z_X_min=2.5
        self.z_X_max=3.5        
        
        self.L = util.chi(self.z_X_max) - util.chi(self.z_X_min)
        self.l_X = 158 # Rest wavelength of CII, in microns.
        
        self.z_min = 0.03
        self.z_max = 5
        
        # Comoving distance to redshift interpolator
        x = np.linspace(self.z_min, self.z_max, 1000)
        y = util.chi(x)
        self.z_from_chi = interpolate.interp1d(y, x)
        
        # Import CIB model. Default use Linear Model from Maniyar 2018        
        if want_halo_model:             
            self.CIBmodel = HaloModel()
        else:
            self.CIBmodel = LinearModel() 
  
        self.CIB_setup = False
        self.CII_setup = False
        self.tab_CIB = tab_CIB

        k = np.linspace(1e-5, 10, 10000)
        z = np.linspace(self.z_min, self.z_max, 2000)
        
        self.PK_interpolator = util.get_camb_mpi(z, k) # Inputs are z, k

        self.b_g = 3 # Constant galaxy bias

    # Tabulate CIB (l, z) grid beforehand to avoid repeated calculation in FFT calls.
    # Halo model in particular is quite expensive to calculate.
    # Along with the tabulation, setup the l, z dimension arrays to fourier transform over.
    def V_tilde_setup(self):
        if self.tab_CIB:            
            self.kpp_arr, self.xp_arr, self.b_dI_dz = util.read_CIB_tab()
        else:            
            # Physical axes for Fourier transform
            xp_min = util.chi(self.z_min)
            xp_max = util.chi(self.z_max)
            xpp_min = util.chi(self.z_X_min)
            xpp_max = util.chi(self.z_X_max)
           
            self.xp_arr = np.linspace(xp_min, xp_max, 8000)
            self.xpp_arr = np.linspace(xpp_min, xpp_max, 2000)
            self.l_arr = self.l_X * (1+self.z_from_chi(self.xpp_arr))
            self.z_arr = self.z_from_chi(self.xp_arr)

            # The CIB model is a function of wavelength and redshift but is gridded
            # over the two physical dimensions. 
            self.b_dI_dz = self.CIBmodel.b_dI_dz(self.l_arr, self.z_arr)
        
    # The full V_tilde term expressed by two Fourier transforms. Fingers crossed this works
    def V_tilde_old(self, kp, kpp, ell):        
        if self.CIB_setup is False:
            self.V_tilde_setup()
            print("CIB model setup complete.")
            self.CIB_setup = True        
        # Obtain the wavemodes with fftfreq       
        kp_arr = np.fft.fftfreq(self.xp_arr.size, d=self.xp_arr[1]-self.xp_arr[0])
        kpp_arr = np.fft.fftfreq(self.xpp_arr.size, d=self.xpp_arr[1]-self.xpp_arr[0])

        # Power Spectrum, which takes a kp value to begin with, set with input kp value.        
        k_radial = np.sqrt(kp**2 + ell**2/self.xp_arr**2)
        sqrtPK = np.sqrt(self.PK_interpolator(self.z_arr, k_radial))
         
        func = self.b_dI_dz * sqrtPK 
        func *= cosmo.H(self.z_arr).value / (const.c/1000) # Shape is (xpp, xp)
        
        # Fourier Transform
        t = time.time()
        func_k_2D = np.fft.fft2(func) # Shape is (kpp, kp), corresponding to before FFT.         
        print(time.time() - t)
        func_k_2D *= (self.xpp_arr.max() - self.xpp_arr.min()) * (self.xp_arr.max() - self.xp_arr.min()) / self.xpp_arr.size / self.xp_arr.size
        kp_idx = np.argmin(abs(-kp - kp_arr)) # Technically inverse FT, take negative kp as desired mode.
        kpp_idx = np.argmin(abs(kpp - kpp_arr))
        return func_k_2D[kpp_idx, kp_idx]

    # New V_tilde implementation using single axis fourier transformed CIB look up table. 
    def V_tilde(self, kp, kpp, ell):
        if self.CIB_setup is False:
            self.V_tilde_setup()
            print("CIB model setup complete.")
            self.CIB_setup = True        
        # Power Spectrum, which takes a kp value to begin with, set with input kp value.        
        k_radial = np.sqrt(kp**2 + ell**2/self.xp_arr**2)
        z_arr = self.z_from_chi(self.xp_arr)
        sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
         
        func = self.b_dI_dz * sqrtPK 
        func *= cosmo.H(z_arr).value / (const.c/1000) # Shape is (kpp, xp)
        
        # Fourier Transform        
        #t = time.time()
        kp_arr, func_k_2D = util.fft_wrapper(func, self.xp_arr, axis=1) # Shape is (kpp, kp), corresponding to before FFT.                 
        #print(time.time() - t)
        kp_idx = np.argmin(abs(-kp - kp_arr)) # Technically inverse FT, take negative kp as desired mode.
        kpp_idx = np.argmin(abs(kpp - self.kpp_arr))
        return func_k_2D[kpp_idx, kp_idx]
               
    ### Full Complex Formulation ###

    # First we define all the new window functions
    
    def sqrtPK_FFT_old(self, kp, kpp, ell):
        xp_arr = np.linspace(util.chi(2.5), util.chi(3.5), 1000)        
        z_arr = self.z_from_chi(xp_arr)
        
        # Matter Power Spectrum Part
        k_radial = np.sqrt(kp**2 + ell**2 / xp_arr**2)
        sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
        
        # Perform FFT
        res = np.fft.fft(sqrtPK) / xp_arr.size * (xp_arr.max() - xp_arr.min())
        kp_arr = np.fft.fftfreq(xp_arr.size, d=xp_arr[1]-xp_arr[0])
        k_wanted = kpp - kp # May be negative
        idx_wanted = np.argmin(abs(k_wanted - kp_arr))        
        return res[idx_wanted]
        #return np.fft.fftshift(kp_arr), np.fft.fftshift(res)
   
    def sqrtPK_FFT(self, kp, kpp, ell):
        xp_arr = np.linspace(util.chi(2.5), util.chi(3.5), 1000)        
        z_arr = self.z_from_chi(xp_arr)
        
        # Matter Power Spectrum Part
        k_radial = np.sqrt(kp**2 + ell**2 / xp_arr**2)
        sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
        
        # Perform FFT        
        kp_arr, res = util.fft_wrapper(sqrtPK, xp_arr)
        k_wanted = kpp - kp # May be negative
        idx_wanted = np.argmin(abs(k_wanted - kp_arr))        
        return res[idx_wanted]        
        #return np.fft.fftshift(kp_arr), np.fft.fftshift(res)    

    # Assumes that CIB and CII always comes first, and galaxy goes second. 
    # No handler exists for CIBxCII.
    def mp_handler(self, s1, s2, kp_arr, kpp, ell, integrand, spc, core_num):
        start = spc * core_num
        end = spc * (core_num+1)
        if s1 == "CIB": # Either CIBxCIB or CIBxg
            for i in range(start, end):
                kp = kp_arr[i]
                V_tilde = self.V_tilde(kp, kpp, ell)
                if s2 == "CIB": # CIBxCIB
                    integrand[i] = abs(V_tilde)**2
                else:
                    PK_part = self.sqrtPK_FFT(kp, kpp, ell)
                    integrand[i] = V_tilde * PK_part * self.b_g # A complex value, we still need to modify this. 
        else: # Either CIIxCII or CIIxg, both involve square of sqrtPK_FFT
            for i in range(start, end):
                kp = kp_arr[i]
                PK_part = abs(self.sqrtPK_FFT(kp, kpp, ell))**2
                if s2 == "CII": # CIIxCII
                    integrand[i] = PK_part * self.CIImodel.I_mean**2 * self.CIImodel.b_mean**2
                else: # CIIxg
                    integrand[i] = PK_part * self.CIImodel.I_mean * self.CIImodel.b_mean * self.b_g

    def compute_cl(self, s1, s2, 
                   kpp_arr, ell_arr, 
                   steps, kp_min, kp_max,
                   cores=20, cks=False):        
        if s1 == "CII":
            if self.CII_setup is False:
                self.CIImodel = CII(np.linspace(2.5, 3.5, 1000))
                print("CII model setup complete.")
                self.CII_setup = True        
        else:
            if self.CIB_setup is False:
                self.V_tilde_setup()
                print("CIB model setup complete.")
                self.CIB_setup = True        

        spc = steps // cores # Steps of calculations Per Core.
        _range = kpp_arr.size if cks else ell_arr.size
        res = np.zeros(_range, dtype="float64") # Expecting real result. 
        
        kp_arr = np.linspace(kp_min, kp_max, steps)
        # Loop through kpp or ell, depending on desired axis.
        for i in tqdm(range(_range)):
            if cks:
                ell = ell_arr
                kpp = kpp_arr[i]
            else:
                ell = ell_arr[i]
                kpp = kpp_arr
            integrand = np.zeros(kp_arr.size, dtype="float64")
            multi_integrand = mp.Array("d", integrand) # Multiprocessing array to be shared.
            plist = []
            for j in range(cores):
                p = mp.Process(target=self.mp_handler, 
                               args=[s1, s2, kp_arr, kpp, ell, multi_integrand, spc, j])
                p.start()
                plist.append(p)
            for p in plist:
                p.join()
            res[i] = integrate.simps(multi_integrand[:], x=kp_arr)
            #print(multi_integrand[:])
        return res / util.chi(3)**2 / self.L / 2 / np.pi




