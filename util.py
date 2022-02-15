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

def fft_wrapper(func, x_arr, axis=-1, shift=False):    
    t = time.time()
    k_arr = np.fft.fftfreq(x_arr.size, d=x_arr[1]-x_arr[0])
    #print(time.time() - t)
   
    t = time.time()
    res = np.fft.fft(func, axis=axis)    
    #print(time.time() - t)

    # Normalization
    t = time.time()
    res *= (x_arr.max() - x_arr.min()) / x_arr.size
    #print(time.time() - t)

    t = time.time()
    if shift:
        k_arr = np.fft.fftshift(k_arr)
        res = np.fft.fftshift(res, axes=(axis,))
    #print(time.time() - t)
    
    return k_arr, res

def read_CIB_tab():
    print("Reading tabulated CIB.")
    subpath = path + "/tabs/FFT_CIB/1000x2000/"
    xp_arr = np.loadtxt(subpath + "xp_arr.txt")
    kpp_arr = np.loadtxt(subpath + "kpp_arr.txt")
    real = np.loadtxt(subpath + "real.txt")
    imag = np.loadtxt(subpath + "imag.txt")
    res = real + 1j * imag
    return kpp_arr, xp_arr, res

def get_camb_mpi(zarr, karr, nonlinear=False):
    # Cosmological parameters setup
    CAMBparams = camb.model.CAMBparams()
    H0 = np.array(acosmo._H0)
    h = acosmo._h
    ombh2 = np.array(acosmo._Ob0) * h**2 # Baryonic density * h^2
    omch2 = np.array(acosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
    CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
   
    PK = mpi(CAMBparams, zmin=zarr.min(), zmax=zarr.max(), nonlinear=nonlinear,
             hubble_units=False, k_hunit=False)


    grid = PK.P(zarr, karr) # Get tabulated value for power spectrum. 
    spline_func = interpolate.RectBivariateSpline(zarr, karr, grid)
    return lambda z, k : spline_func(z, k, grid=False)

def hmf(z, mh):
    ccosmo.setCosmology("planck18")
    return mass_function.massFunction(mh, z=z, q_in="M", q_out="dndlnM", mdef="200c", model="tinker08")

def get_hmf_interpolator():
    subpath = path + "/HMFdata"
    
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
    subpath = path+"/HMFdata"
    
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

# SFR Params
eta_max = 0.42 # Yes yes all egregiously without errors.
M_max = 10**12.94 # Halo mass at which SFR produces at efficiency η_max
z_c = 1.5 # redshift below which σ is allowed to evolve with redshift.
sigma_Mh0 = 1.75 # variance of the log normal.
tau = 1.17 # Evolution parameter

def sigma(z):
    res = np.full(z.shape, sigma_Mh0, dtype="float64")
    idx = np.where((z_c - z) > 0)
    res[idx] -= tau * (z_c - z[idx])
    return res

def eta(M_h, z): 
    expo = -(np.log(M_h)-np.log(M_max))**2/2/(sigma(z)**2)
    res = eta_max*np.exp(expo)
    return res

# Now the baryonic accretion rate:
def M_dot(M_h, z): # equation (7)
    res = 46.1 * (M_h/1e12)**(1.1) * (1+1.11*z) * np.sqrt(acosmo.Om0*(1+z)**3 + acosmo.Ode0) # Units [M_h/yr]
    return res

def BAR(M_h, z): # equation (6)
    res = M_dot(M_h, z) * acosmo.Ob(z)/acosmo.Om(z) 
    return res

# Finally, the star formation rate:
def SFR(M_h, z): # equation(9)
    z_grid, M_h_grid = np.meshgrid(z, M_h)
    res = eta(M_h_grid, z) * BAR(M_h_grid, z)
    # Shape will be (z.size, M_h.size)
    return res.T

# Calculate SFR for subhalo, taking smaller value of equations 9 and 10 in Maniyar 2020
def SFR_sub(mheff, msub, z):
    res = np.zeros((msub.size, z.size), dtype="float64")
    SFRI = SFR(msub, z).T
    SFRII = np.outer((msub/mheff), SFR(mheff, z))
    for i in range(msub.size):
        res[i, :] = np.minimum(SFRI[i, :], SFRII[i, :])
    return res
   
