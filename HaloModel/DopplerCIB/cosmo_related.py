# cosmo relared inputs

from headers_constants import *


class cosmo_var_iv(object):

    def __init__(self, mass, z, do_powerspec):  # , ell):
        self.mass = mass
        self.z = z
        # self.ell = ell

        nm = len(self.mass)
        nz = len(self.z)

        deltah_cib = 200.
        # ########## reading in the matter power spectra #############
        redshifts = np.loadtxt('data_files/redshifts.txt')
        #redshifts = np.loadtxt("/mount/citadel1/zz1994/codes/CIBxLIM/HaloModel/DopplerCIB/data_files/redshifts.txt")
        nr = len(redshifts)
        self.zpk = redshifts

        if min(self.z) < min(redshifts) or max(self.z) > max(redshifts):
            print ("If the redshift range is outside of [%s to %s], then " +
                   "values of the matter power spectrum and effective " +
                   "CIB SEDs are extrapolated and might be incorrect.") % (min(redshifts), max(redshifts))
        ll = [str(x) for x in range(1, 211)]
        addr = 'data_files/matter_power_spectra'
        pkarray = np.loadtxt('%s/test_highk_lin_matterpower_210.dat' % (addr))
        self.k = pkarray[:, 0]*cosmo.h
        self.Pk = np.zeros((len(self.k), len(redshifts)))
        for i in range(len(redshifts)):
            pkarray = np.loadtxt("%s/test_highk_lin_matterpower_%s.dat" % (addr, ll[209-i]))
            self.Pk[:, i] = pkarray[:, 1]/cosmo.h**3

        # pkinterp2d = RectBivariateSpline(k, redshifts, Pk)
        # pkinterpk = interp1d(k, Pk.T, kind='linear', bounds_error=False, fill_value=0.)
        # pkinterpz = interp1d(redshifts, Pk, kind='linear', bounds_error=False, fill_value=0.)
        self.pkinterpz = interp1d(redshifts, self.Pk, kind='linear', bounds_error=False, fill_value="extrapolate")
        # self.pkinterpk = interp1d(k, Pk.T, kind='linear', bounds_error=False, fill_value="extrapolate")

        """
        self.k_array = np.zeros((len(self.ell), len(self.z)))
        self.Pk_int = np.zeros(self.k_array.shape)
        chiz = cosmo.comoving_distance(self.z).value
        for i in range(len(self.ell)):
            self.k_array[i, :] = self.ell[i]/chiz
            for j in range(len(self.z)):
                pkz = pkinterpz(self.z[j])
                self.Pk_int[i, j] = np.interp(self.k_array[i, j], k, pkz)
        """

        # ######## hmf, bias, nfw ###########
        print ("Calculating the halo mass function " +
               "for given mass and redshift for CIB mean calculations.")

        self.hmf = np.zeros((nm, nz))
        delta_h = deltah_cib

        for r in range(nz):
            pkz = self.pkinterpz(self.z[r])
            instance = hmf_unfw_bias.h_u_b(self.k, pkz, self.z[r],
                                           cosmo, delta_h, self.mass)
            self.hmf[:, r] = instance.dn_dlogm()
        """
        for r in range(nr):
            instance = hmf_unfw_bias.h_u_b(k, Pk[:, r], redshifts[r],
                                           cosmo, delta_h, self.mass)
            hmfr[:, r] = instance.dn_dlogm()
        self.hmf_r = interp1d(redshifts, hmfr, kind='linear',
                              bounds_error=False, fill_value=0.)
        """

        if do_powerspec == 1:
            self.nfw_u = np.zeros((nm, len(self.k), nr))
            self.bias_m_z = np.zeros((nm, nr))
            for r in range(nr):
                instance = hmf_unfw_bias.h_u_b(self.k, self.Pk[:, r],
                                               redshifts[r],
                                               cosmo, delta_h, self.mass)
                self.nfw_u[:, :, r] = instance.nfwfourier_u()
                self.bias_m_z[:, r] = instance.b_nu()

            """
            self.nfw = RegularGridInterpolator((mass, k, redshifts), nfw_u)
            self.biasmz = interp1d(redshifts, bias_m_z, kind='linear',
                                   bounds_error=False, fill_value=0.)
            """

    def dchi_dz(self, z):
        a = c_light/(cosmo.H0*np.sqrt(cosmo.Om0*(1.+z)**3 + cosmo.Ode0))
        return a.value

    def chi(self, z):
        return cosmo.comoving_distance(z).value

    def karray(self, ell, z):
        nl = len(ell)
        nz = len(z)
        k_array = np.zeros((nl, nz))

        for i in range(nl):
            k_array[i, :] = ell[i]/self.chi(z)

        return k_array

    def interp_bias(self, z):
        nm, nz = len(self.mass), len(z)
        bias = np.zeros((nm, nz))
        for m in range(nm):
            bias[m, :] = np.interp(z, self.zpk, self.bias_m_z[m, :])
        return bias

    def interp_nfw(self, ell, z):
        nm, nl, nz = len(self.mass), len(ell), len(z)
        nfw_ureds = np.zeros((nm, len(self.k), nz))
        for i in range(len(self.k)):
            for m in range(nm):
                nfw_ureds[m, i, :] = np.interp(z, self.zpk, self.nfw_u[m, i, :])

        u_nfw = np.zeros((nm, nl, nz))
        k_array = self.karray(ell, z)
        for m in range(nm):
            for j in range(nz):
                u_nfw[m, :, j] = np.interp(k_array[:, j], self.k,
                                           nfw_ureds[m, :, j])
        return u_nfw

    def Pk_array(self, ell, z):
        nl = len(ell)
        nz = len(z)
        nreds = len(self.zpk)
        pk1 = np.zeros((nl, nreds))
        Pk_int = np.zeros((nl, nz))

        for i in range(nreds):
            ell_chi = ell/self.chi(self.zpk[i])
            pk1[:, i] = np.interp(ell_chi, self.k, self.Pk[:, i])

        for i in range(nl):
            Pk_int[i, :] = np.interp(z, self.zpk, pk1[i, :])

        return Pk_int

    def beta2(self, z):
        gamma = 0.55
        H_z = cosmo.H(z).value
        f = (cosmo.Om(z))**gamma
        a = 1./(1.+z)
        fact = (a*f*H_z/c_light)**2
        fact *= 1./3
        pk = self.pkinterpz(z)
        integrand = pk/(2*np.pi**2)
        # integrand *= self.k[:, None]
        res = intg.simps(integrand, x=self.k, axis=0, even='avg')
        res *= fact
        return res
