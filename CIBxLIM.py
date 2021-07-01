import numpy as np
import scipy.constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
from LinearModel.LinearModel import LinearModel
import camb
from camb import get_matter_power_interpolator as mpi
import vegas
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

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
                 exp="EXCLAIM",
                 target_line="CII",
                 target_line_rest_wavelength=None,
                 want_function_of_K=False,
                 verbose=True):
        
        self.verbose = verbose # Operation messages
        
        if exp == "EXCLAIM":
            self.z_X_min=2.5
            self.z_X_max=3.5
        else:
            print("Sorry, only EXCLAIM experiment is supported at the moment.")

        self.l_X = 158 # Rest wavelength of CII, in microns.
        self.ell = 0
        self.kp = 0
        self.kpp = 0

        # Import CIB model. Default use Linear Model from Maniyar 2018
        self.CIBmodel = LinearModel() 
        self.chi_p = lambda l : self.CIBmodel.interp_chi_peaks(l)
        self.z_min = self.CIBmodel.redshifts.min()
        self.z_max = self.CIBmodel.redshifts.max()
        
        """
        # Perform CAMB parameter setup according to astropy cosmology.
        # Setup Nonlinear matter power spectrum interpolator for subsequent calculations. 
        self.camb_setup()
        self.PK = mpi(self.CAMBparams, 
                 zmin=0, 
                 zmax=11, 
                 hubble_units=False, 
                 k_hunit=False) # Here I do not use Hubble units for input or output.
        
        """
        # Tuple random points P(z, k) interpolator.
        print("Loading and creating P(z, k) interpolator...")
        PK_grid = np.loadtxt("PKdata/k0-40_3500.txt")
        k = np.linspace(0, 1, 3500)
        z = np.linspace(self.z_min, self.z_max, 3500)
        self.PK_interpolator = interpolate.RectBivariateSpline(z, k, PK_grid)
        print("P(z, k) interpolator successfully created.")

        # Effective bias from the linear model:
        b0 = 0.83
        b1 = 0.742
        b2 = 0.318
        self.b_eff = lambda z : b0 + b1*z + b2*z**2

        # Comoving distance to redshift interpolator
        x = np.linspace(self.z_min, self.z_max, 1000)
        y = np.array(cosmo.comoving_distance(x))
        self.z_from_chi = interpolate.interp1d(y, x)

    def camb_setup(self):
        self.CAMBparams = camb.model.CAMBparams()
        H0 = np.array(cosmo._H0)
        h = cosmo._h # hubble unit
        ombh2 = np.array(cosmo._Ob0) * h**2 # Baryonic density * h^2
        omch2 = np.array(cosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
        self.CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
        print("CAMB setup successful")

    def z_X(self, l):
        # Redshift from which a line with rest wavelength l_X is received at l
        return l/self.l_X - 1
        
    def sinu_arg_5D(self, kpp, kp, l1, z1, l2, z2):
        x1pp = np.array(cosmo.comoving_distance(self.z_X(l1)))
        x2pp = np.array(cosmo.comoving_distance(self.z_X(l2)))
        x1p = np.array(cosmo.comoving_distance(z1))
        x2p = np.array(cosmo.comoving_distance(z2))
        return (kpp*(x1pp-x2pp) - kp*(x1p-x2p))

    def exp_arg(self, k_para_prime, k_para, l, z):
        # Shorthand for k_prime x_prime(l) - kx(z)
        
        # Physical distances, both in Mpc
        x_para_prime = np.array(cosmo.comoving_distance(self.z_X(l))) # Line redshift distance
        x_para = np.array(cosmo.comoving_distance(z)) # CIB redshift distance
       
        return (k_para_prime*x_para_prime - k_para*x_para)

##### Simpsons 1D + 2 x (Vegas 2D)

    # x contains integration variables [u, z], where u=l/l_X
    @vegas.batchintegrand
    def integrand_Re(self, x):
        ret = np.cos(self.exp_arg(self.kpp, self.kp, self.l_X*x[:, 0], x[:, 1]))
        #ret *= self.F(self.kp, self.ell, x[:, 1])
        #ret *= self.G(self.l_X*x[:, 0], x[:, 1])
        return ret

    @vegas.batchintegrand
    def integrand_Im(self, x):
        ret = np.sin(self.sinu_arg(self.kp, x[:, 0], x[:, 1]))
        ret *= self.F_full(self.kpp, self.ell, self.kp, x[:, 0], x[:, 1])
        return ret

    def integrand_Im_simpsons(self, z, u):
        ret = self.F_full(self.kpp, self.ell, self.kp, z, u)
        ret *= np.sin(self.sinu_arg(self.kp, z, u))
        return ret

    def create_vegas_integrator(self): 
        u_min = self.z_X_min+1
        u_max = self.z_X_max+1

        z_min = self.CIBmodel.redshifts.min()
        z_max = self.CIBmodel.redshifts.max()

        return vegas.Integrator([[z_min, z_max], [u_min, u_max]])

    def compute_Cl(self, kpp, ell, neval, nitn=10):
        self.kp=0.3 # Test k_parallel at 0.3 1/Mpc
        self.kpp = kpp
        self.ell = ell

        integ = self.create_vegas_integrator()

        #res_Re = integ(self.integrand_Re, nitn=nitn, neval=neval)
        res_Im = integ(self.integrand_Im, nitn=nitn, neval=neval)

        #print(res_Re.summary())
        print(res_Im.summary())

##### "The straight forward" #####
    def F_full(self, kpp, ell, kp, z, u):
        ret = (const.c / 1000) / self.chi_p(u*self.l_X) / np.array(cosmo.H(self.z_X(u*self.l_X))) # Dimensionless factor, comes from xpp ---> l.
        ret *= self.b_eff(z) # bias
        
        # Calculate matter power spectrum component.
        k = np.sqrt(kp**2 + ell**2/np.array(cosmo.comoving_distance(z)))
        ret *= np.sqrt(self.PK_interpolator(z, k, grid=False))

        ret *= self.CIBmodel.CIB_model(u*self.l_X, z) # The CIB intensity in Jy.
        
        return ret

    def sinu_arg(self, kp, z, u):
        xpp = np.array(cosmo.comoving_distance(self.z_X(u*self.l_X)))
        xp = np.array(cosmo.comoving_distance(z))
        return (self.kpp*xpp - kp*xp)

    @vegas.batchintegrand
    def integrand_full(self, x):
        # x is [kp, z1, u1, z2, u2]
        ret = self.F_full(self.kpp, self.ell, x[:, 0], x[:, 1], x[:, 2])
        ret *= self.F_full(self.kpp, self.ell, x[:, 0], x[:, 3], x[:, 4])
        #ret *= np.exp(1j * self.sinu_arg(x[:, 0], x[:, 1], x[:, 2]))
        #ret *= np.exp(-1j * self.sinu_arg(x[:, 0], x[:, 3], x[:, 4]))
        ret *= np.sin(self.sinu_arg(x[:, 0], x[:, 1], x[:, 2]) - self.sinu_arg(x[:, 0], x[:, 3], x[:, 4]))
        return ret

    def integrand_full_nonbatch(self, kp, z1, u1, z2, u2):
        # x is [kp, z1, u1, z2, u2]
        ret = self.F_full(self.kpp, self.ell, kp, z1, u1)
        ret *= self.F_full(self.kpp, self.ell, kp, z2, u2)
        #ret *= np.exp(1j * self.sinu_arg(x[:, 0], x[:, 1], x[:, 2]))
        #ret *= np.exp(-1j * self.sinu_arg(x[:, 0], x[:, 3], x[:, 4]))
        #ret *= np.sin(self.sinu_arg(kp, z1, u1) - self.sinu_arg(kp, z2, u2))
        return ret

### Fourier Analysis ### 
    
    # l, z should be arrays for FFT, kp and ell should be single values (for now).
    # Output dimension is (dim(l), dim(z))
    def fourier_integrand(self, l, z, kp, ell):
        ll, zz = np.meshgrid(l, z)
        scaling = cosmo.H(zz) / (const.c / 1000) / self.chi_p(ll) # (dim(l), dim(z))
        bias = self.b_eff(z) # (dim(z),)
        k = np.sqrt(kp**2 + ell**2 / np.array(cosmo.comoving_distance(z))**2) # Spherical k mode.
        pkz = np.sqrt(self.PK_interpolator(z, k, grid=False))
        CIB = self.CIBmodel.CIB_model(l, z) # (dim(l), dim(z))
        return scaling * bias * pkz * CIB

    def F_hat(self, kpp, z_arr):
        chi_X_min = np.array(cosmo.comoving_distance(self.z_X_min))
        chi_X_max = np.array(cosmo.comoving_distance(self.z_X_max))
        chi_X_arr = np.linspace(chi_X_min, chi_X_max, 1000) # Evenly spaced comoving distance points.
        z_X_arr = self.z_from_chi(chi_X_arr) # Convert to redshift, not evenly spaced. 
        l_arr = (1+z_X_arr) * self.l_X # In micrometers
        
        CIB = self.CIBmodel.CIB_model(l_arr, z_arr) # CIB model data grid. Shape is (dim(l), dim(z))
        chip = self.chi_p(l_arr)
        func = CIB / chip.reshape((chip.size, 1)) # CIB model / peak comoving distance.
        
        func_k = np.fft.fft(func, axis=0) # Apply Fourier transform to l axis.
        k = np.fft.fftfreq(chi_X_arr.size, d=chi_X_arr[1] - chi_X_arr[0])
        func_k = func_k[np.where(k>=0), :][0] # We only want part of function at positive k. 
        k = k[np.where(k>=0)] # Filter out negative k.

        closest_idx = np.argmin(np.abs(k - kpp)) # Index of the k value closest to input kpp.
        return func_k[closest_idx, :] # Return row of transformed function corresponding to kpp. Shape is (dim(z),)

    def V_tilde(self, kp, kpp, ell, neval=10000):
        z_arr = np.linspace(self.z_min, self.z_max, neval)
        F_hat_func = self.F_hat(kpp, z_arr)
        bias = self.b_eff(z_arr)
        xp = np.array(cosmo.comoving_distance(z_arr)) # x parallel, comoving distance at CIB redshifts.
        k_arr = np.sqrt(kp**2 + ell**2 / xp**2) # Spherical k  mode.
        pkz = np.sqrt(self.PK_interpolator(z_arr, k_arr, grid=False))

        sinu_arg = -1j * kp * xp

        func_total = np.exp(sinu_arg) * bias * pkz * F_hat_func

        return integrate.simps(y=func_total, x=z_arr)
    
    def fourier_Cl(self, kpp, ell):
        kp_arr = np.linspace(0, 0.5, 150) # k parallel to integrate over.
        V_arr = np.zeros(kp_arr.size, dtype="complex128")
        for i in range(kp_arr.size):
            V_arr[i] = self.V_tilde(kp_arr[i], kpp, ell)
        V_arr_conj = V_arr.conj() # Take conjugate.
        res = integrate.simps(y=V_arr * V_arr_conj, x=kp_arr) # Integrate V*V, along k parallel. 
        return res


