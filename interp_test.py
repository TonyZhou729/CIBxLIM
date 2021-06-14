import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from LinearModel.LinearModel import LinearModel
import camb
from camb import get_matter_power_interpolator as mpi


CIB = LinearModel()
zmin = CIB.redshifts.min()
zmax = CIB.redshifts.max()

CAMBparams = camb.model.CAMBparams()
H0 = np.array(cosmo._H0)
h = cosmo._h
ombh2 = np.array(cosmo._Ob0) * h**2
omch2 = np.array(cosmo._Odm0) * h**2
CAMBparams.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, Alens=1.2)
print("CAMB setup successful")

# This here is the interpolator from CAMB, which forms a grid upon calling. 
# The goal is to then Rectangular spline off of this grid, to form 1D data over pairs of random points.
PK = mpi(CAMBparams,
         zmin=0,
         zmax=11,
         hubble_units=False,
         k_hunit=False)


data_dim = 3500
kmax = 40
k = np.linspace(0, kmax, data_dim) # k mode, In Mpc^-1
z = np.linspace(zmin, zmax, data_dim) # Redshift

fname = "PKdata/k0-{}_{}.txt".format(kmax, data_dim)
#PK_grid = np.loadtxt(fname)
PK_grid = PK.P(z, k)
np.savetxt(fname, PK_grid)

#test_k = np.random.random(2000) * kmax
#test_z = (zmax - zmin) * np.random.random(10) + zmin
test_k = np.linspace(0, kmax, 5000)
test_z = 3.7

f = interpolate.RectBivariateSpline(z, k, PK_grid)

j = f(test_z, test_k, grid=False)
true = PK.P(test_z, test_k)

plt.loglog(test_k, j, label="Interpolated")
plt.loglog(test_k, true, "--", label="CAMB")
plt.legend()
plt.xlabel("k [Mpc]^{-1}")
plt.ylabel("P(z, k)")
plt.title(test_z)
plt.show()



"""
print(j.shape)

print(f(4, 0.25, grid=False))
for i in range(10):
    print("True value: {}".format(PK.P(test_z[i], test_k[i])))
    print("Interpolate: {}".format(j[i]))
    print()
"""
