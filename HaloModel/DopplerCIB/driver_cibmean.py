from headers_constants import *
from input_var_cibmean import *
from Inu_cib import *
from cosmo_related import *
import time

time0 = time.time()

# color corrections for 100, 143, 217, 353, 545, 857 and 3000 GHz for Planck
# cc_pl = np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995])  # , 0.960])
cc_pl = np.ones(6)
fc_pl = np.ones(len(cc_pl))

"""
Calculating the observed CIB intensity for halos with a given mass at given
redshifts for different Planck frequencies. The SEDs used for the Planck
channels are bandpassed, so the observed intensities are calculated as
they would be observed with Planck frequency channels at 100, 143, 353, 545,
847 GHz as well as 3000 GHz for IRAS. Intensity is calculated in nW/m^2/sr.
"""
# nuarray = np.array([100., 143., 217., 353., 545., 857.])
deltanu = 800
nucen = 100
"""
CAN SPPED UP THE COMPUTATION BY MAKING NUCEN AS AN ARRAY. IT JUST GIVES THE
CENTRAL FREQUENCY OF THE FILTER. SO WE CAN CALCULATE ALPHA FOR ALL THE
PLANCK FREQUENCIES TOGETHER IN NEED BE.
"""
nu0min = 50.  # nucen-deltanu/2.
nu0max = 3000.  # nucen+deltanu/2.
steps = nu0max-nu0min+1  # nu0max-nu0min+1  # 200
nu0 = np.linspace(nu0min, nu0max, int(steps))  # nuarray  # np.linspace(nu0min, nu0max, 200)

ell = np.linspace(50, 3000, 15)

Planck = {'name': 'Planck_only',
          'do_cibmean': 1,
          'cc_cibmean': np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995, 0.960]),
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
# cc_cibmean = np.array([0.97125, 0.999638, 1.00573, 0.959529, 0.973914, 0.988669, 0.987954, 1., 1.])
# freq_iv = np.array([1875, 3000, 667, 353, 600, 857, 1200, 250, 242])
# freq_iv = np.array([100., 143., 217., 353., 545., 857., 3000.])
# snuaddr: 'data_files/filtered_snu_cib_15_new.fits'

# ell = np.linspace(150., 2000., 20)
redshifts = np.loadtxt('data_files/redshifts.txt')

zsource = 2.
z1 = np.linspace(min(redshifts), zsource, 10)
# z2 = np.linspace(min(redshifts), 1.5, 80)
# z3 = np.linspace(1.51, max(redshifts), 30)
# z11 = np.concatenate((z2, z3))
# zn = np.linspace(min(redshifts), 3., 130)
z = z1  # z1  # redshifts # zn # z11

logmass = np.arange(6, 15.005, 0.1)
mass = 10**logmass

do_powerspec = 0

driver_uni = cosmo_var_iv(mass, z, do_powerspec)
driver = data_var_iv(exp)  # , z)  # , ell)

# cibmean = I_nu_cib(driver)
cibmean = I_nu_cib(driver, driver_uni)
I_nu = cibmean.Iv()(nu0)
# alpha = cibmean.alpha()


# alpha1 = np.gradient(np.log(I_nu), np.log(nu0))
# plot_Inu_freq(nu0, I_nu)
# plot_Inu_freq(nu0, alpha1)

# freq = ['100', '143', '217', '353', '545', '857', '3000']
# Iv_cen = np.array([13.63, 12.61, 1.64, 0.46, 2.8, 6.6, 10.1, 0.08, 0.05])
# for i in range(len(I_nu)):
#     print "Intensity is %f nW/m^2/sr at %s GHz" % (I_nu[i], freq[i])


def Tdust(z):
    T0 = 24.4  # Planck CIB 2013 paper
    alpha = 0.36
    result = T0*(1.+z)**alpha
    return result


def B_nu(z, nu):
    Td = Tdust(z)
    res = 2.*h_p*nu**3/(c_light*1.e3)**2
    x = h_p*nu/k_B/Td
    # print (min(x), max(x))
    res /= (np.exp(x) - 1)
    return res


def mod_blackbody(z, nu):
    beta = 1.75
    Bnu = B_nu(z, nu)
    result = Bnu
    result *= nu**beta
    # result *= w_jy  # Watt to Jy
    return result


def alpha_modblackbod(z, nu):
    # gamma = 1.7
    # nu_0 = 1000.
    Inu = mod_blackbody(z, nu)
    grad = np.gradient(np.log(Inu), np.log(nu))
    alpha = grad  # *nu/Inu
    # alpha = np.where(nu < nu_0, alpha, -gamma)
    return alpha


def plot_Inu_freq(exp, mass, zs):
    # driver_uni = cosmo_var_iv(mass, z)
    nuarray = exp['nu0']
    driver = data_var_iv(exp)  # , z)  # , ell)
    nz = len(zs)

    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111)

    for i_z in range(nz):
        z1 = np.linspace(min(redshifts), zs[i_z], 10)
        z = z1  # z1  # redshifts # zn # z11
        driver_uni = cosmo_var_iv(mass, z, do_powerspec)
        cibmean = I_nu_cib(driver, driver_uni)
        I_nu = cibmean.Iv()(nuarray)
        # alpha = cibmean.alpha()
        col = plt.cm.rainbow(i_z/float(nz))
        ax.plot(nuarray, I_nu, c=col, label=r'$z_s = %s$' % (zs[i_z]))
        # ax.plot(np.log(nuarray), np.log(I_nu), c=col, label=r'$z_s = %s$' % (zs[i_z]))

        Inu_mod = mod_blackbody(zs[i_z], ghz*nuarray)
        ax.plot(nuarray, Inu_mod, c=col, ls='--')

    ax.legend(fontsize='18', loc='lower left', frameon=False)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'$\nu_{\rm obs}$ [GHz]', fontsize=24)
    ax.set_ylabel(r'$I_\nu$ [Jy]', fontsize=24)
    # ax.set_ylim((1e-9))  # , 4.e-6))
    # ax.set_xlim((5., 4.e3))
    ax.set_title(r'$z_{\rm source} = %s$' % (zsource))
    ax.tick_params(axis='both', labelsize=20)
    plt.show()


zs = np.array([0.5, 1.5, 3., 5.])
# plot_Inu_freq(exp, mass, zs)


def plot_alpha_freq_z(exp, mass, zs):
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111)
    if exp['name'] == 'Planck_only':
        freq = ['100', '143', '217', '353', '545', '857']  # , '3000']
        nuarray = np.array([100., 143., 217., 353., 545., 857.])  # , 3000.])
        nf = len(freq)
        nz = len(zs)
        alpha = np.zeros((nf, nz))
    
        # lines = ["-", "--", "-."]
        # cl = ["g", "b", "r", "k"]
        for i_z in range(nz):
            z1 = np.linspace(min(redshifts), zs[i_z], 10)
            z = z1  # z1  # redshifts # zn # z11
            driver_uni = cosmo_var_iv(mass, z, do_powerspec)
            for i_f in range(nf):
                Planck['nucen'] = freq[i_f]
                exp_n = Planck
                driver = data_var_iv(exp_n)  # , mass)
                cibmean = I_nu_cib(driver, driver_uni)
                alpha[i_f, i_z] = cibmean.alpha()
                # alpha[:, i_z] = cibmean.alpha()
    
            # ax.plot(nuarray, alpha[:, i_z], cl[i_z], label='z = %s' % (zs[i_z]))
            col = plt.cm.rainbow(i_z/float(nz))
            ax.plot(nuarray, alpha[:, i_z], c=col, label=r'$z_s = %s$' % (zs[i_z]))
    else:
        driver = data_var_iv(exp)  # , mass)
        nuarray = exp['nu0']
        nz = len(zs)
        for i_z in range(nz):
            z1 = np.linspace(min(redshifts), zs[i_z], 10)
            z = z1  # z1  # redshifts # zn # z11
            driver_uni = cosmo_var_iv(mass, z, do_powerspec)
            cibmean = I_nu_cib(driver, driver_uni)

            # alpha = cibmean.alpha()
            alpha = cibmean.alpha()

            col = plt.cm.rainbow(i_z/float(nz))
            ax.plot(nuarray, alpha, c=col, label=r'$z_s = %s$' % (zs[i_z]))

            alpha_mod_black = alpha_modblackbod(zs[i_z], ghz*nuarray)
            ax.plot(nuarray, alpha_mod_black, c=col, ls='--')
            # print (alpha_mod_black[40], alpha_mod_black[100], alpha_mod_black[150])
        ax.plot([], [], c='k', ls='--', label="Modified blackbody")
    # ax.plot(nuarray, alpha, 'r')
    ax.legend(fontsize='18', loc='lower left', frameon=False)  # , labelspacing=0.1)
    # ax.set_xscale('log')
    # ax.set_yscale('log', nonposy='mask')
    ax.set_xlabel(r'$\nu_{\rm obs}$ [GHz]', fontsize=24)
    ax.set_ylabel(r'$\alpha$', fontsize=24)
    # ax.set_ylim((1.0))  # , 4.e-6))
    ax.set_xlim((50.))  # , 4.e3))
    ax.tick_params(axis='both', labelsize=20)
    plt.show()
    # plt.savefig('output/Figures/alpha-nu-zs-nonfilteredInugradient-general.pdf', bbox_inches="tight")


zs = np.array([0.02, 0.5, 1.5, 3., 5.])
# zs = np.array([0.5])  # , 1.5, 3., 5.])
# plot_alpha_freq_z(exp, mass, zs)

print(time.time()-time0)
