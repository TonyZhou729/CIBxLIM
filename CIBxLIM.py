import numpy as np
import scipy.constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
from LinearModel.LinearModel import LinearModel
import camb
from camb import get_matter_power_interpolator as mpi
import vegas

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
                 want_function_of_K=False):

        if exp == "EXCLAIM":
            self.z_X_min=2.5
            self.z_X_max=3.5
        else:
            print("Sorry, only EXCLAIM experiment is supported at the moment.")

        self.l_X = 158 # Rest wavelength of CII, in microns.
        self.ell = 0
        self.kpp = 0

        # Import CIB model. Default use Linear Model from Maniyar 2018
        self.CIBmodel = LinearModel() 
        self.chi_p = lambda l : self.CIBmodel.interp_chi_peaks(l)

        # Perform CAMB parameter setup according to astropy cosmology.
        # Setup Nonlinear matter power spectrum interpolator for subsequent calculations. 
        self.camb_setup()
        self.PK = mpi(self.CAMBparams, 
                 zmin=0, 
                 zmax=11, 
                 hubble_units=False, 
                 k_hunit=False) # Here I do not use Hubble units for input or output.
        
        # Effective bias from the linear model:
        b0 = 0.83
        b1 = 0.742
        b2 = 0.318
        self.b_eff = lambda z : b0 + b1*z + b2*z**2

    def camb_setup(self):
        self.CAMBparams = camb.model.CAMBparams()
        H0 = np.array(cosmo._H0)
        h = cosmo._h # hubble unit
        ombh2 = np.array(cosmo._Ob0) * h**2 # Baryonic density * h^2
        omch2 = np.array(cosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
        self.CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
        print("CAMB setup successful")

    def compute_powerspectrum(self, k_para_1, k_para_2, ell, func_of_k = False):  
        k_para = np.linspace(0, 1, 100) / u.Mpc
        func = lambda k_para : self.V_tilde_Re(k_para, k_para_1, ell)**2 + self.V_tilde_Im(k_para, k_para_2, ell)**2
        if func_of_k:
            # Is function of k_parallel'
            func_arr = np.zeros((k_para.size, k_para_1.size)) * u.Jy**2 * u.Mpc**5
        else:
            # Is function of ell at some value k_parallel'
            func_arr = np.zeros((k_para.size, ell.size)) * u.Jy**2 * u.Mpc**5
        for i, k in enumerate(k_para):
            print(i)
            func_arr[i] = func(k)
        
        # Normalizing distances
        L_prime = cosmo.comoving_distance(self.z_X_max) - cosmo.comoving_distance(self.z_X_min)
        z_avg = np.average(self.CIBmodel.redshifts)
        chi = cosmo.comoving_distance(self.z_X_max)
   
        return np.trapz(func_arr, k_para, axis=0)/2/np.pi/L_prime/(chi**2)
        
    """
    Real part of one V_tilde term. For full definition refer to Jupyter Notebook.
    2D integral over redshift and line wavelength, integrated numerically using Monte Carlo. 
    
    Params:
    ------
    (float) k_para_prime : Wavenumber, C_ell covariant matrix index.
    (float) k_para : CIB wavenumber integration variable.
    (float) ell : Angular wavenumber

    Return:
    ------
    Monte carlo integrated result of the assembled 2D function of wavelength and CIB redshift. 
    """
    def V_tilde_Re(self, k_para, k_para_prime, ell):
        func = lambda l, z : np.cos(self.sinu_arg(k_para_prime, k_para, l, z)) * self.F(k_para, ell, z) * self.G(l, z)
        return self.monte_carlo_integrator(func, ell, k_para_prime)

    """
    Imaginary part of one V_tilde term. All forms identical to V_tilde_Re but the cosine term replaced
    by a sine term due to the Euler identity expansion. Parameters and return value analogous to that 
    of V_tilde_Re.
    """
    def V_tilde_Im(self, k_para, k_para_prime, ell): 
        func = lambda l, z : np.sin(self.sinu_arg(k_para_prime, k_para, l, z)) * self.F(k_para, ell, z) * self.G(l, z)
        return self.monte_carlo_integrator(func, ell, k_para_prime)

    """
    Abbreviated function, product of the effective bias b_eff(z) and the matter power spectrum from CAMB.

    Params:
    ------
    (float) k_{para} : CIB wavenumber, integration variable in Cl
    (float) ell : Angular wavenumber
    (float) z : CIB redshift

    Return:
    ------
    b_eff(z) x sqrt(P(k_{para}, z))
    """
    def F(self, k_para, ell, z):
        chi = np.array(cosmo.comoving_distance(z)) # Comoving distance
        k = np.sqrt(k_para**2 + ell**2/chi**2) # k = sqrt(kpara^2 + kperp^2)), made dimensionless
        return self.b_eff(z) * np.sqrt(self.PK.P(z, k)).T # Units Mpc^(3/2)

    def G(self, l, z):
        c = const.c / 1000 # Speed of light in units km/s
        coef = c / self.l_X / np.array(cosmo.H(self.z_X(l))) * (u.Mpc.to(u.um)) # Units Mpc/um
        CIB = self.CIBmodel.CIB_model(l, z) # Units Jy 
        return coef * CIB / self.chi_p(l) # Units Jy/um, chi_p cancels out the Mpc. 

    def z_X(self, l):
        # Redshift from which a line with rest wavelength l_X is received at l
        return l/self.l_X - 1
        
    def sinu_arg(self, k_para_prime, k_para, l, z):
        # Shorthand for k_prime x_prime(l) - kx(z)
        
        # Physical distances, both in Mpc
        x_para_prime = cosmo.comoving_distance(self.z_X(l)) # Line redshift distance
        x_para = cosmo.comoving_distance(z) # CIB redshift distance
       
        return (k_para_prime*x_para_prime - k_para*x_para).to(u.rad, equivalencies=u.dimensionless_angles())

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
    
    def monte_carlo_integrator(self, func, ell, k_para_prime):
        # Integrate the input function in 2D over redshift and wavelength.
        N = 100 # Number of random samples
        #res = np.zeros(ell.size, dtype="float64")
        res = np.zeros(k_para_prime.size, dtype="float64") * u.Jy * u.Mpc**(3/2)

        # Wavelength bounds computed using z_X = l/l_X - 1
        l_X = self.l_X
        l_min = (self.z_X_min+1)*l_X
        l_max = (self.z_X_max+1)*l_X

        # CIB redshift bounds
        z_min = self.CIBmodel.redshifts.min()
        z_max = self.CIBmodel.redshifts.max()

        for i in range(N):
            # Generate random pair and sum func. 
            rand_l = (l_max - l_min) * np.random.random_sample() + l_min
            rand_z = (z_max - z_min) * np.random.random_sample() + z_min
            res += func(rand_l, rand_z)
        res *= (z_max-z_min) * (l_max-l_min).to(u.Mpc)/N
        return res


##### Vegas 5D #####
    """
    x : 5D list, contains integration variables [v, u1, z1, u2, z2]
    """
    def integrand(self, x):
        l_X = self.l_X
        ret = np.cos(self.sinu_arg_5D(self.kpp, x[0]*self.kpp, x[1]*l_X, x[2], x[3]*l_X, x[4]))
        ret *= self.F(x[0]*self.kpp, self.ell, x[2]) * self.G(x[1]*l_X, x[2])
        ret *= self.F(x[0]*self.kpp, self.ell, x[4]) * self.G(x[3]*l_X, x[4])
        return ret
    
    """
    x : 5D list, contains integration variables [v, u1, z1, u2, z2]
    """
    @vegas.batchintegrand
    def batch_integrand(self, x):
        l_X = self.l_X
        for i in range(5):
            print(x[:, i].shape)
        ret = np.sin(self.sinu_arg_5D(self.kpp, x[:, 0]*self.kpp, x[:, 1]*l_X, x[:, 2], x[:, 3]*l_X, x[:, 4]))[0]
        ret *= self.F(x[:, 0]*self.kpp, self.ell, x[:, 2]) * self.G(x[:, 1]*l_X, x[:, 2])[0]
        ret *= self.F(x[:, 0]*self.kpp, self.ell, x[:, 4]) * self.G(x[:, 3]*l_X, x[:, 4])[0]
        return ret

    """
    Compute the powerspectrum using above integral.
    """
    def Vegas5D(self, kpp, ell):
        self.kpp = kpp
        self.ell = ell
 
        u_min = self.z_X_min+1
        u_max = self.z_X_max+1

        z_min = self.CIBmodel.redshifts.min()
        z_max = self.CIBmodel.redshifts.max()

        v_min = 0
        v_max = 1/self.kpp
     
        L = np.array(cosmo.comoving_distance(self.z_X_max) - cosmo.comoving_distance(self.z_X_min))

        integ = vegas.Integrator([[v_min, v_max], [u_min, u_max], [z_min, z_max], [u_min, u_max], [z_min, z_max]])
        ret = integ(self.integrand, nitn=10, neval=10000)
        ret *= self.kpp * self.l_X**2 / (2*np.pi) / L
        return ret


class CIBxLIM_V2():
    
    def __init__(self):
        self.z_X_min = 2.5
        self.z_X_max = 3.5
        self.l_X = 158 # Rest wavelength of CII in microns.

         # Import CIB model. Default use Linear Model from Maniyar 2018
        self.CIBmodel = LinearModel() 
        self.chi_p = lambda l : self.CIBmodel.interp_chi_peaks(l)

        # Perform CAMB parameter setup according to astropy cosmology.
        # Setup Nonlinear matter power spectrum interpolator for subsequent calculations. 
        self.camb_setup()
        self.PK = mpi(self.CAMBparams, 
                 zmin=0, 
                 zmax=11, 
                 hubble_units=False, 
                 k_hunit=False) # Here I do not use Hubble units for input or output.
        
        # Effective bias from the linear model:
        b0 = 0.83
        b1 = 0.742
        b2 = 0.318
        self.b_eff = lambda z : b0 + b1*z + b2*z**2

        # Default k_parallel integration variable as 0-1 [Mpc]^{-1}. 
        self.kp = 0

    def camb_setup(self):
        self.CAMBparams = camb.model.CAMBparams()
        H0 = np.array(cosmo._H0)
        h = cosmo._h # hubble unit
        ombh2 = np.array(cosmo._Ob0) * h**2 # Baryonic density * h^2
        omch2 = np.array(cosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
        self.CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
        print("CAMB setup successful")
       
    # Integrand formulations:

    def F(self, k_para, ell, z):
        chi = np.array(cosmo.comoving_distance(z)) # Comoving distance
        k = np.sqrt(k_para**2 + ell**2/chi**2) # k = sqrt(kpara^2 + kperp^2)), made dimensionless
        return self.b_eff(z) * np.sqrt(self.PK.P(z, np.flip(k))) # Units Mpc^(3/2)

    def G(self, l, z):
        # Resulting shape is (shape(z),shape(l))
        c = const.c / 1000 # Speed of light in units km/s
        coef = c / self.l_X / np.array(cosmo.H(self.z_X(l))) * (u.Mpc.to(u.um)) # Units Mpc/um
        CIB = self.CIBmodel.CIB_model(l, z).T # Units Jy 
        return coef * CIB / self.chi_p(l) # Units Jy/um, chi_p cancels out the Mpc. 

    def z_X(self, l):
        # Redshift from which a line with rest wavelength l_X is received at l
        return l/self.l_X - 1


