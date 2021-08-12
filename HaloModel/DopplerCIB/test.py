import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from headers_constants import *
from input_var_cibmean import *
from Inu_cib import *
from cosmo_related import *
import time
from scipy.integrate import simps

cc_pl = np.ones(6)
fc_pl = np.ones(len(cc_pl))

deltanu = 800
nucen = 100

nu0min = 50.  # nucen-deltanu/2.
nu0max = 3000.  # nucen+deltanu/2.
steps = nu0max-nu0min+1  # nu0max-nu0min+1  # 200
nu0 = np.linspace(nu0min, nu0max, int(steps))  # nuarray  # np.linspace(nu0min, nu0max, 200)

### LIM specifics:
l_X = 158 # In microns
z_X_min = 2.5
z_X_max = 3.5
l = np.linspace((1+z_X_max)*l_X, (1+z_X_min)*l_X, 3) # Units micrometers
l0 = l/1e6 # Units meters
nu0 = const.c/l0 # Units Hz
nu0 /= 1e9 # Units GHz

nu0 = np.array([100, 143, 217, 353, 545, 857], dtype="float64")
l0 = const.c/(nu0*1e9) # Units meters
l0 *= 1e6 # Units microns

print(nu0)
print(l0)


ell = np.linspace(50, 3000, 15)

custom = {'name': 'custom',
        'do_cibmean': 1,
        'cc': cc_pl,
        'fc': fc_pl,
        'snuaddr': 'data_files/filtered_snu_planck.fits',
        'nu0min': nu0min, 'nu0max': nu0max,
        'nucen': str(int(nucen)),
        'nu0': nu0,
        'ell': ell,
        'cibpar_resfile': 'data_files/one_halo_bestfit_allcomponents_' +
        'lognormal_sigevol_1p5zcutoff_nospire_fcpl_onlyautoshotpar_' +
        'no3000_gaussian600n857n1200_planck_spire_hmflog10.txt'}

exp = custom

redshifts = np.loadtxt('data_files/redshifts.txt')
redshifts = np.linspace(min(redshifts), max(redshifts), 1000)

zsource = 2.
z = np.linspace(min(redshifts), zsource, 10)

logmass = np.arange(6, 15.005, 0.1)
mass = 10**logmass

do_powerspec = 0

driver_uni = cosmo_var_iv(mass, redshifts, do_powerspec)
driver = data_var_iv(exp) 

cibmean = I_nu_cib(driver, driver_uni)
jnu = cibmean.J_nu_iv()
a = lambda z : 1/(1+z)
dchidz = lambda z : (const.c/1000) / np.array(cosmo.H(z))


### Plotting the linear model.
from LinearModel.LinearModel import LinearModel
Linear = LinearModel()

for i, l in enumerate(l0):
    print("{:.2f} GHz".format(nu0[i]))
    print("Linear: {} Jy".format(simps(Linear.CIB_model(l, redshifts)[0], x=redshifts)))
    print("Halo: {} Jy".format(simps(jnu[i]*a(redshifts)*dchidz(redshifts), x=redshifts)))
    print("From Iv(self): {} Jy".format(cibmean.Iv()(nu0[i])))
    print()

"""
for i in range(nu0.size):
    ### Plotting the halo model.
    plt.plot(redshifts, jnu[i]*a(redshifts)*dchidz(redshifts), "--", label="{:.2f} GHz".format(nu0[i]))
    #plt.plot(redshifts, jnu[i], "--", label="{:.2f} GHz".format(nu0[i]))
    plt.plot(redshifts, Linear.CIB_model(l[i], redshifts).T, label="{:.2f} GHz".format(nu0[i]))
    #plt.plot(redshifts, (Linear.j(l[i], redshifts).T[:, 0]), label="{:.2f} GHz".format(nu0[i]))

plt.title("Linear vs. Halo Model")
plt.legend()
plt.xlabel("z")
plt.ylabel("Jy")
plt.show()
"""













