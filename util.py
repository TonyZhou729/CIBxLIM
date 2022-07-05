import numpy as np
from astropy.cosmology import Planck18_arXiv_v2 as acosmo

from scipy import interpolate
from scipy import integrate

from colossus.cosmology import cosmology as ccosmo
from colossus.lss import mass_function
from colossus.lss import bias

import camb
from camb import get_matter_power_interpolator as mpi

import time

"""
Utility functions for CIBxLIM powerspectra calculations.

List of functions are as follows:
"""

path = "/mount/citadel1/zz1994/codes/CIBxLIM" # Enter full path to CIBxLIM package directory.

def chi(z):
    return np.array(acosmo.comoving_distance(z))

def load_cl(s1, s2, di=-1):
    subpath = path+"/CIBcls/dr3/"
    fname = s1+"x"+s2
    if di != -1:
        fname += "_dp{}".format(di)
    if s1 == "CIB" and s2 != "CIB":
        fr = fname+"_real.txt"
        fi = fname+"_imag.txt"
        return np.loadtxt(subpath+fr) + 1j*np.loadtxt(subpath+fi)
    else:
        return np.loadtxt(subpath+fname+".txt")

def read_CIB_tab(FFT = True, di=-1):
    print("Reading tabulated CIB.")
    if FFT:
        subpath = path + "/tabs/FFT_CIB/1000x2000/"
        xp_arr = np.loadtxt(subpath + "xp_arr.txt")
        kpp_arr = np.loadtxt(subpath + "kpp_arr.txt")
        if di == -1:        
            real = np.loadtxt(subpath + "real.txt")
            imag = np.loadtxt(subpath + "imag.txt")
        else:        
            real = np.loadtxt(subpath + "real_dp{}.txt".format(di))
            imag = np.loadtxt(subpath + "imag_dp{}.txt".format(di))
        res = real + 1j * imag
        return kpp_arr, xp_arr, res
    else:
        subpath = path + "/tabs/Halo_CIB/1000x2000/"
        xp_arr = np.loadtxt(subpath + "xp_arr.txt")
        xpp_arr = np.loadtxt(subpath + "xpp_arr.txt")
        if di == -1:
            res = np.loadtxt(subpath + "b_dI_dz.txt")
        else:
            res = np.loadtxt(subpath + "b_dI_dz_dp{}.txt".format(di))
        return xpp_arr, xp_arr, res

"""
If log is true, karr should be evenly spaced on log scale, such as by using np.logspace().
Gird will be log(P) on axes z and log(k)
"""
def get_camb_mpi(zarr, karr, nonlinear=False, use_log=False):
    # Cosmological parameters setup
    CAMBparams = camb.model.CAMBparams()
    H0 = np.array(acosmo._H0)
    h = acosmo._h
    ombh2 = np.array(acosmo._Ob0) * h**2 # Baryonic density * h^2
    omch2 = np.array(acosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
    CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
   
    PK = mpi(CAMBparams, zmin=zarr.min(), zmax=zarr.max(), nonlinear=nonlinear,
             hubble_units=False, k_hunit=False)

    if use_log:
        logPK = lambda z, logk : np.log10(PK.P(z, 10**logk))
        grid = logPK(zarr, np.log10(karr)) # Both arguments are evenly spaced, k and P are log valued.  
        spline_func = interpolate.RectBivariateSpline(zarr, np.log10(karr), grid)
        return lambda z, k : 10**spline_func(z, np.log10(k), grid=False)
    else:
        grid = PK.P(zarr, karr) # Get tabulated value for power spectrum. 
        spline_func = interpolate.RectBivariateSpline(zarr, karr, grid)    
        return lambda z, k : spline_func(z, k, grid=False)    

def hmf(z, mh):
    ccosmo.setCosmology("planck18")
    return mass_function.massFunction(mh, z=z, q_in="M", q_out="dndlnM", mdef="200c", model="tinker08")

def get_hmf_interpolator():
    subpath = path + "/tabs/HMFdata"
    
    # Load axes and data grid
    zarr = np.loadtxt(subpath+"/z.txt")
    mharr = np.loadtxt(subpath+"/mh.txt")
    data = np.loadtxt(subpath+"/hmf.txt")

    # Create bivariate spline interpolator
    spline_func = interpolate.RectBivariateSpline(zarr, mharr, data.T)
    
    # Return function capable of accepting both redshift and M_halo as arrays.
    return lambda z, mh : spline_func(z, mh, grid=True)

def halo_bias(z, mh):
    ccosmo.setCosmology("planck18")
    return bias.haloBias(mh, model="tinker10", z=z, mdef="200c")

def get_halo_bias_interpolator():
    # Load axes and data grid
    subpath = path + "/tabs/HMFdata"
    
    zarr = np.loadtxt(subpath+"/z.txt")
    mharr = np.loadtxt(subpath+"/mh.txt")
    data = np.loadtxt(subpath+"/bias.txt")

    # Create bivariate spline interpolator
    spline_func = interpolate.RectBivariateSpline(zarr, mharr, data.T)
    
    # Return function capable of accepting both redshift and M_halo as arrays.
    return lambda z, mh : spline_func(z, mh, grid=True)

def subhmf(mh, ms, log10=True):
    res = 0.13 * (ms/mh)**(-0.7)*np.exp(-9.9*(ms/mh)**2.5)
    if log10:
        res *= np.log(10)
    return res

"""
Given Halo mass, return subhalo mass range on a log10 scale.
Assume that the minimum subhalo mass is 10^5 solar masses.
"""
def get_msub(mh, log10msub_min=5):
    log10mh = np.log10(mh)
    log10msub = np.arange(log10msub_min, log10mh, 0.1)
    return 10**log10msub

"""
Specifics of the SFR Halo Model following Maniyar et al. (2020). 
Included are the base best fit parameter values as well as all relevant
analytical formulations. 

See astro-ph.CO:2006.16329 for specifics. 
"""
z_c = 1.5 # redshift below which σ is allowed to evolve with redshift.

def sigma(z, sigma_Mh0, tau):
    res = np.full(z.shape, sigma_Mh0, dtype="float64")
    idx = np.where((z_c - z) > 0)
    res[idx] -= tau * (z_c - z[idx])
    return res

def eta(M_h, z, eta_max=0.42, log10M_max=12.94, sigma_Mh0=1.75, tau=1.17):     
    lnM_max = np.log(10**log10M_max)
    expo = -(np.log(M_h)-lnM_max)**2/2/(sigma(z, sigma_Mh0, tau)**2)                    
    res = eta_max*np.exp(expo)
    return res

# Formulas for the Baryonic Accretion Rate (BAR):

def M_dot(M_h, z): # equation (7)
    res = 46.1 * (M_h/1e12)**(1.1) * (1+1.11*z) * np.sqrt(acosmo.Om0*(1+z)**3 + acosmo.Ode0) # Units [M_h/yr]
    return res

def BAR(M_h, z): # equation (6)
    res = M_dot(M_h, z) * acosmo.Ob(z)/acosmo.Om(z)     
    return res

"""
The Star Formation Rate (SFR) defined as eta x BAR.

Params:
------
M_h : Range of halo masses considered, 1D arrray.

z : Range of redshifts considered. If singular value must be an array or length 1. (e.g. np.array([z])).

di : If a value between 0 and 3, returns the analytical derivative result of SFR with respect 
to the di parameter. Default order is [eta, log_10(M_max), sigma_Mh0, tau]. 
Returns SFR if di=-1, which is the default.

params :  Give specific values in a length 4 array if alternative best fit values are needed.
useful for computing numerical derivatives. This has been done to check the accuracy of the analytical derivatives.

Returns:
------
SFR at given best fit parameter values or analytical derivative. 2D array of shape (z.size, M_h.size)
"""
def SFR(M_h, z, di=-1, params=[0.42, 12.94, 1.75, 1.17]): # equation(9)
    z_grid, M_h_grid = np.meshgrid(z, M_h) # Shape is (M_h, z)
    eta_max = params[0]
    log10M_max = params[1]
    sigma_Mh0 = params[2]
    tau = params[3]
    
    res = eta(M_h_grid, z_grid, eta_max, log10M_max, sigma_Mh0, tau) * BAR(M_h_grid, z_grid)        
    lnM_max = np.log(10**log10M_max)
        
    # Shape will be (z.size, M_h.size)
    if di == 0: # d/d(eta_max)
        factor = 1/eta_max
    elif di == 1: # d/d(log(M_max)
        factor = (np.log(M_h_grid) - lnM_max) / sigma(z_grid, sigma_Mh0, tau)**2 * np.log(10)
    elif di == 2: # d/d(sigma_Mh0)
        factor = (np.log(M_h_grid) - lnM_max)**2 / sigma(z_grid, sigma_Mh0, tau)**3
    elif di == 3: # d/d(tau)
        factor = -(np.log(M_h_grid) - lnM_max)**2 / sigma(z_grid, sigma_Mh0, tau)**3
        idx_zero = np.where((z_c - z) <= 0) # Replace these with 0, the rest stay constant.
        z_grid[:, idx_zero] = 0
        idx_rest = np.where((z_c-z) > 0) # positive values become z_c-z
        z_grid[:, idx_rest] = z_c - z[idx_rest]
        factor *= z_grid
    else: # No derivatives        
        factor = 1
    return res * factor

# Calculate SFR for subhalo, taking smaller value of equations 9 and 10 in Maniyar 2020
def SFR_sub(mheff, msub, z):
    res = np.zeros((msub.size, z.size), dtype="float64")
    SFRI = SFR(msub, z).T
    SFRII = np.outer((msub/mheff), SFR(mheff, z))
    for i in range(msub.size):
        res[i, :] = np.minimum(SFRI[i, :], SFRII[i, :])
    return res

### Other Math Stuff ###

def fft_wrapper(func, x_arr, axis=-1, shift=False):    
    k_arr = np.fft.fftfreq(x_arr.size, d=x_arr[1]-x_arr[0])
    res = np.fft.fft(func, axis=axis)    
    
    # Normalization
    res *= (x_arr.max() - x_arr.min()) / x_arr.size

    if shift:
        k_arr = np.fft.fftshift(k_arr)
        res = np.fft.fftshift(res, axes=(axis,))    
    return k_arr, res


# Computes the expected inverse convergence following the Sherman-Morrison formula,
# and the Miller (1981) recursive relation. Requires that A+B is either rank 1 or 2.
def SMM_converge(A, B):
    rank = np.linalg.matrix_rank(B)
    Ai = np.linalg.inv(A)
    
    if rank == 1:
        # Directly apply the S-M formula.
        factor = 1 / np.trace(Ai @ B)
        return Ai - factor * Ai @ B @ Ai
    elif rank == 2:
        # First find rank 1 bases using SVD, then call S-M on each sub part.
        u, s, vh = np.linalg.svd(B)
        B1 = s[0] * np.outer(u[:, 0], vh[0])
        B2 = s[1] * np.outer(u[:, 1], vh[1])

        # Solve a rank 1 problem recursively using B1.
        Xi = SMM_converge(A, B1)
        
        # Some trace shorthands for computing the correction factor.
        Tr1 = np.trace(Ai @ B1)
        Tr2 = np.trace(Ai @ B2)
        Tr12 = np.trace(Ai @ B1 @ Ai @ B2)
        factor = 1 / (Tr2 - Tr12/Tr1)
        return Xi - factor * Xi @ B2 @ Xi
    else:
        print("Input B matrix has incorrect rank _{}_.".format(rank))
        return 0

