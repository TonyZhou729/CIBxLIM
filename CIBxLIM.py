import numpy as np
import scipy.constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
from models import LinearModel, HaloModel
from CII import CII
from tqdm import tqdm
from scipy import interpolate, integrate
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
    want_halo_model : Uses Maniyar20 Halo model if true, otherwise linear model from Maniyar18.
    tab_CIB : Reads in tabulated CIB if true, else will compute from scratch, which may take a minute.
    use_FFT_CIB : If true, uses FFT to compute CIB window function (V_tilde), else will be integration.
    use_FFT_CII : Same thing but for CII window function.
    di : -1 if computing Cl, [0, 1, 2, 3] if computing derivative of Cl w.r.t. Halo parameters.
    """    
    def __init__(self,
                 want_halo_model=True,                 
                 tab_CIB=True,
                 use_FFT_CIB=False,
                 use_FFT_CII=False,
                 di = -1):

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
        
        # Create CIB model.
        if want_halo_model:             
            self.CIBmodel = HaloModel()
        else:
            self.CIBmodel = LinearModel() 
  
        # Computation specifics:
        self.CIB_setup = False
        self.CII_setup = False
        self.tab_CIB = tab_CIB
        self.di = di # For fisher matrix derivatives.
        self.use_FFT_CIB = use_FFT_CIB
        self.use_FFT_CII = use_FFT_CII

        # Create matter power interpolator.         
        k = np.logspace(-6, 0, 1000, base=10)
        z = np.linspace(self.z_min, self.z_max, 2000)
        
        self.PK_interpolator = util.get_camb_mpi(z, k, nonlinear=True, use_log=True) # Inputs are z, k

        self.b_g = 3 # Constant galaxy bias

    """
    Read/Calculate CIB intensity (b(z) x dI/dz(v, z)) once to avoid repetition in window function.
    Either reads or computes based on tab_CIB parameter in __init__.
    """
    def V_tilde_setup(self):
        if self.tab_CIB:            
            if self.use_FFT_CIB:
                self.kpp_arr, self.xp_arr, self.b_dI_dz = util.read_CIB_tab(self.use_FFT_CIB, self.di)
            else:                
                xpp_arr, xp_arr, b_dI_dz = util.read_CIB_tab(self.use_FFT_CIB, di=self.di)
                self.b_dI_dz = interpolate.RectBivariateSpline(xpp_arr, xp_arr, b_dI_dz)
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
        
    """
    The subsequent four functions are different approaches to calculating the CIB window function.
    The specifics are in equation (30) of CIBxLIM Computation.
    """
    # The full V_tilde term expressed by two Fourier transforms. Fingers crossed this works
    # THIS FUNCTION IS OBSOLETE.
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

    # Uses FFT for both axes transformations. 
    # Will not be called unless self.use_CIB_FFT is true. 
    def V_tilde_FFT(self, kp, kpp, ell):        
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
        
    # Use tabulated FFT along xpp, integrate along xp.
    # Will not be called unless self.use_CIB_FFT is true. 
    def V_tilde_semiFFT(self, kp, kpp, ell):
        if self.CIB_setup is False:
            self.V_tilde_setup()
            print("CIB model setup complete.")
            self.CIB_setup = True                

        z_arr = self.z_from_chi(self.xp_arr)
        
        k_radial = np.sqrt(kp**2 + ell**2 / self.xp_arr**2)
        sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
        func = cosmo.H(z_arr).value / (const.c/1000) * self.b_dI_dz * sqrtPK
        func = func * np.exp(-1j*kp*self.xp_arr) # Only do xp dim integral        
                
        kpp_idx = np.argmin(abs(kpp - self.kpp_arr))        
        integ = integrate.simps(func[kpp_idx], self.xp_arr)        
        return integ
    
    # Function for when self.use_CIB_FFT is false. 
    def V_tilde_integ(self, kp, kpp, ell, xpp_step=1000, xp_step=500):        
        if self.CIB_setup is False:
            self.V_tilde_setup()
            print("CIB model setup complete.")
            self.CIB_setup = True                
        xp_arr = np.linspace(util.chi(0.03), util.chi(5), xp_step)
        z_arr = self.z_from_chi(xp_arr)
        xpp_arr = np.linspace(util.chi(2.5), util.chi(3.5), xpp_step)
        b_dI_dz = self.b_dI_dz(xpp_arr, xp_arr) # Shape is (xpp, xp)
        
        """
        # Quick Visualization. 
        idx = [100, 200, 699]
        for i in idx:
            plt.plot(xp_arr, b_dI_dz[i], label="{:.4f}".format(xpp_arr[i]))
        plt.legend()
        plt.show()
        """        
        
        k_radial = np.sqrt(kp**2 + ell**2 / xp_arr**2)
        sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
        func = cosmo.H(z_arr).value / (const.c/1000) * b_dI_dz * sqrtPK
        func = func * np.exp(-1j*kp*xp_arr)
        func = func.T * np.exp(1j*kpp*xpp_arr) # Shape is (xp, xpp), due to transpose.
        
        #plt.plot(xpp_arr, func[100])
        #plt.show()
        
        integ1 = integrate.simps(func, xpp_arr, axis=1)
        integ2 = integrate.simps(integ1, xp_arr)        
        return integ2           
   
    """
    The CII (or equivalently galaxy) window function, made general by _not_ including the intensity or bias terms, 
    since none of them are allowed to evolve with redshift. For details see equations (31) and (32) in CIBxLIM Computation. 
    """
    def sqrtPK(self, kp, kpp, ell, steps=2000):        
        if self.use_FFT_CII:
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
        else:
            xp_arr = np.linspace(util.chi(2.5), util.chi(3.5), steps)
            z_arr = self.z_from_chi(xp_arr)            
            expo = np.exp(1j*(kpp - kp)*xp_arr)
            
            k_radial = np.sqrt(kp**2 + ell**2 / xp_arr**2)
            sqrtPK = np.sqrt(self.PK_interpolator(z_arr, k_radial))
            return integrate.simps(sqrtPK * expo, x=xp_arr)    

    """
    Multiprocessing handler, calculates spc (steps per core) number of window function products.
    
    Params:
    ------
    s1 : First observable, "CIB", "CII" or "g"
    s2 : Second observable. (eg, s1="CIB", s2="g" will compute V_tilde x bg x sqrtPK)
    kp_arr : k_|| array, integration variable in equation (33)
    kpp : k_||', requested value.
    ell : requested value.
    integrand & integrand_imag : Multiprocessing arrays to store windowfunction products.
    spc : steps per core, should be k_|| integration step divided by number of cores.
    core_num : core index, decides which portion of integrand to calculate.
    """
    def mp_handler(self, s1, s2, kp_arr, kpp, ell, integrand, integrand_imag, spc, core_num):
        start = spc * core_num
        end = spc * (core_num+1)
        if s1 == "CIB": # One of CIBxCIB, CIBxCII or CIBxg
            for i in range(start, end):
                kp = kp_arr[i]                                
                # Calculate CIB window using appropriate method. 
                if self.use_FFT_CIB:
                    V_tilde = self.V_tilde_semiFFT(kp, kpp, ell)
                else:
                    V_tilde = self.V_tilde_integ(kp, kpp, ell)
                
                if s2 == "CIB": # CIBxCIB
                    integrand[i] = abs(V_tilde)**2
                else: # CIBxCII or CIBxg
                    PK_part = self.sqrtPK(kp, kpp, ell)
                    res = V_tilde * PK_part.conj() * self.b_g # Complex value                    
                    res = res * self.b_g if s2 == "g" else res * self.CIImodel.I_mean * self.CIImodel.b_mean
                    integrand[i] = res.real
                    integrand_imag[i] = res.imag
        else: # One of CIIxCII, CIIxg or gxg, all of which need the base window function.
            for i in range(start, end):
                kp = kp_arr[i]
                PK_part = abs(self.sqrtPK(kp, kpp, ell))**2
                if s1 == "CII" and s2 == "CII":
                    integrand[i] = PK_part * self.CIImodel.I_mean**2 * self.CIImodel.b_mean**2
                elif s1 == "CII" and s2 == "g":
                    integrand[i] = PK_part * self.CIImodel.I_mean * self.CIImodel.b_mean * self.b_g
                else:
                    integrand[i] = PK_part * self.b_g**2        
        
    """
    Method to call for computing C(k_||', ell)

    Params:
    ------    
    s1 : First observable, "CIB", "CII" or "g"
    s2 : Second observable. (eg, s1="CIB", s2="g" will compute V_tilde x bg x sqrtPK)
    kpp_arr : k_||' single value wanted or 1darray to plot over, the latter case requires over_k_axis=TRUE.
    ell_arr : ell single value wanted or 1darray to plot over, the latter case requires over_k_axis=FALSE.
    steps : Number of integration steps for k_|| integral. NOTE: STEPS / CORES MUST BE AN INTEGER!!
    kp_min : Lower bound of k_|| integral.
    kp_max : Upper bound of k_|| integral, recommended < 0.025
    cores : Number of cores for multiproceessing.
    over_k : True if output is over k at one value of ell, converse if false.
    """
    def compute_cl(self, s1, s2, 
                   kpp_arr, ell_arr, 
                   steps, kp_min, kp_max,
                   cores=20, over_k=False):        
        if s1 == "CII":
            if self.CII_setup is False:
                self.CIImodel = CII(z=np.array([3], dtype="float64"))
                print("CII model setup complete.")
                self.CII_setup = True        
        elif s1 == "CIB":
            if self.CIB_setup is False:
                self.V_tilde_setup()
                print("CIB model setup complete.")
                self.CIB_setup = True        

        complex_out = True if (s1 == "CIB" and s2 == "g") or (s1 == "CIB" and s2 == "CII") else False # Need real and imag parts if true.                 
        
        spc = steps // cores # Steps of calculations Per Core.
        _range = kpp_arr.size if over_k else ell_arr.size
        res = np.zeros(_range, dtype="complex128" if complex_out else "float64") # Expecting real result. 
        
        kp_arr = np.linspace(kp_min, kp_max, steps)
        # Loop through kpp or ell, depending on desired axis.
        for i in tqdm(range(_range)):
            if over_k:
                ell = ell_arr
                kpp = kpp_arr[i]
            else:
                ell = ell_arr[i]
                kpp = kpp_arr
            integrand = np.zeros(kp_arr.size, dtype="float64")
            mparr = mp.Array("d", integrand) # Multiprocessing array to be shared.         
            plist = []
            for j in range(cores):
                if complex_out:
                    mparr_imag = mp.Array("d", integrand) # To store complex part 
                    p = mp.Process(target=self.mp_handler, 
                                   args=[s1, s2, kp_arr, kpp, ell, mparr, mparr_imag, spc, j])
                else:
                    p = mp.Process(target=self.mp_handler, 
                                   args=[s1, s2, kp_arr, kpp, ell, mparr, 0, spc, j]) 
                p.start()
                plist.append(p)
            for p in plist:
                p.join()
            if complex_out:
                res[i] = integrate.simps(np.array(mparr[:]) + 1j * np.array(mparr_imag[:]), x=kp_arr)
            else:
                res[i] = integrate.simps(np.array(mparr[:]), x=kp_arr)            
        return res / util.chi(3)**2 / self.L / 2 / np.pi




