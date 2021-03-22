import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as acosmo
from colossus.cosmology import cosmology as ccosmo
from colossus.lss import mass_function
from colossus.lss import bias

ccosmo.setCosmology("planck15")
chi = lambda z : acosmo.comoving_distance(z)

################## SFR Component #####################
eta_max = 0.42
M_max = 10**12.94 # Halo mass at which SFR produces at efficiency η_max
z_c = 1.5 # redshift below shich σ is allowed to evolve with redshift.
sigma_Mh0 = 1.75 # variance of the log normal.
tau = 1.17

K = 1e-10 # Kennicutt constant in M_sun/yr/L_sun
# Note: SFR has units M_sun/yr, and SFR/K has units L_sun. 

def sigma(z):
    res = sigma_Mh0
    if z < z_c:
        res -= tau*(z_c - z)
    return res

def eta(M_h, z):
    expo = -(np.log10(M_h)-np.log10(M_max))**2/2/(sigma(z)**2)
    res = eta_max * np.exp(expo)
    return res

def M_dot(M_h, z):
    res = 46.1 * (M_h/1e12)**(1.1) * (1+1.11*z) * np.sqrt(acosmo.Om(z)*(1+z)**3 + acosmo.Ode(z)) # Units [M_h/yr]
    return res

def BAR(M_h, z):
    res = M_dot(M_h, z) * acosmo.Ob(z)/acosmo.Om(z) 
    return res

def SFR(M_h, z):
    res = eta(M_h, z) * BAR(M_h, z)
    return z

################ Halo Mass Function & bias ##################
mfunc = lambda M_h, z : mass_function.massFunction(M_h, z, mdef="200m", model="tinker08") # Use Mass definition of 200 x mean density of universe, model from Tinker 2008.
bias_func = lambda M_h, z : bias.haloBias(M_h, z, mdef="200m", model="tinker10")

################ SED #################
# Unimplemented for now, use place holder S_nu = 1
SED = lambda z, nu : 1

################# Halo Model Integral ##################
nu =1
z_array = np.linspace(0, 20, 50)
M_h_array = np.linspace(8, 15, 1000) # Mass in log10 space.
central_halo = np.zeros((z_array.size, M_h_array.size), dtype="float64")
for i, z in enumerate(z_array): # Construct 2D grid of dj/dlog(M_h) over z and M_h
    central_halo[i, :] = bias_func(10**M_h_array, z) * mfunc(10**M_h_array, z) * np.array(chi(z))**2 * (1+z) * SFR(10**M_h_array, z)/K * SED(z, nu)


#central_halo = lambda M_h, z, nu : mfunc(M_h, z) * np.array(chi(z))**2 * (1+z) * SFR(M_h, z)/K * SED(z, nu) # This is the dj/d(logM_h) for the central halo. 

# Now integrate this w.r.t. log(M_h) to obtain the central halo emissitivity. We'll worry about subhalo later.
j_c = lambda z, nu : np.trapz(central_halo, M_h_array, axis=1)

plt.plot(z_array, j_c(z_array, 1), label="Over logM")
#plt.plot(10**M_h_array,central_halo[10], label="Log space")
################# Alternative Integration test ##################
nu = 1
z_array = np.linspace(0, 20, 50)
M_h_array = 10**np.linspace(8, 15, 1000)
for i, z in enumerate(z_array): # Construct 2D grid of dj/dlog(M_h) over z and M_h
    central_halo[i, :] = bias_func(M_h_array, z) * mfunc(M_h_array, z) * np.array(chi(z))**2 * (1+z) * SFR(M_h_array, z)/K * SED(z, nu)

j_c = lambda z, nu: np.trapz(central_halo/(M_h_array * np.log(10)), M_h_array, axis=1)
plt.plot(z_array, j_c(z_array, 1), label="Over M")
#plt.plot(M_h_array, central_halo[10], label="Linear space")
plt.xlabel("z")
plt.ylabel("Emissitivity j")
plt.legend()
plt.show()

"""
/(M_h_array * np.log10(np.log10(M_h_array)))
"""


