import camb
import numpy as np
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from camb import get_matter_power_interpolator as mpi
import matplotlib.pyplot as plt

params = camb.model.CAMBparams()

H0 = np.array(cosmo._H0)
h = cosmo._h # hubble unit
ombh2 = np.array(cosmo._Ob0) * h**2 # Baryonic density * h^2
omch2 = np.array(cosmo._Odm0) * h**2 # Cold Dark Matter density * h^2
params.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)

PK = mpi(params, zmin=0, zmax=11)

z = 0.1
k = np.logspace(-3, 0, 100, base=10)

plt.loglog(k, PK.P(z, k))
plt.show()

