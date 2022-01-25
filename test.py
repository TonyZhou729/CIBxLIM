import numpy as np
import matplotlib.pyplot as plt 
import util
from CII import CII
from CIBxLIM import CIBxLIM
from tqdm import tqdm
from scipy import interpolate
import time
import SFR
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from colossus.cosmology import cosmology as ccosmo
import models

CIB = CIBxLIM()
#ell = np.linspace(2, 1000, 10)
ell = np.arange(3)
kpp = 0.001
#print(abs(CIB.V_tilde(0.01, kpp, 10))**2)
cl = CIB.compute_cl("CIB", "CIB", kpp, ell, 100)
plt.loglog(ell, cl)
plt.show()

"""
CIB = CIBxLIM(want_halo_model=False)
#ell = np.linspace(2, 300, 50)
ell = np.linspace(2, 1000, 10)
#ell_arr = np.array([10, 500, 1000])
#kpp_arr = np.logspace(-4, 0, 100, base=10)
#kpp_arr = np.linspace(0, 0.5, 100)
kpp_arr = np.linspace(0, 0.1, 5)

steps = np.array([1000, 1500, 2000, 2500])
mocksteps = np.array([20, 40, 60, 80])

for step in steps:
    cl_CIB = CIB.CIBxCIB_mp(0.001, ell, step, cores=20, cks=False)
    plt.loglog(ell, cl_CIB, label="{} Steps".format(step))
plt.legend()
plt.show()
    
"""

