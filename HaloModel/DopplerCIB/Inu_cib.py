from headers_constants import *


class I_nu_cib(object):

    def __init__(self, data_var_iv, cosmo_var_iv):  # ,
        self.dv = data_var_iv
        self.uni = cosmo_var_iv
        self.z = self.uni.z  # self.dv.z
        self.z_c = self.dv.z_c
        self.mh = self.uni.mass
        self.filt = self.dv.filt
        self.filtgrad = self.dv.filtgrad
        self.nu0min = self.dv.nu0min
        self.nu0max = self.dv.nu0max
        # self.nu0 = np.linspace(self.nu0min, self.nu0max, 200)
        self.nu0 = self.dv.nu0
        self.nucen = self.dv.nucen
        # self.snu_eff = self.dv.snu
        self.snu_unfilt = self.dv.unfiltered_snu(self.nu0, self.z)
        self.cosmo = cosmo
        # self.deltah = deltah
        self.Meffmax = self.dv.Meffmax
        self.etamax = self.dv.etamax
        self.sigmaMh = self.dv.sigmaMh
        self.tau = self.dv.tau
        self.hmfmz = self.uni.hmf  # self.dv.hmf
        self.sig_z = np.array([max(self.z_c - r, 0.) for r in self.z])
        self.sigpow = self.sigmaMh - self.tau*self.sig_z
        # self.cc_cibmean = self.dv.cc_cibmean
        # self.freq_Iv = self.dv.freq_Iv

    def sfr_mhdot(self, mhalo):
        """ SFR/Mhdot lognormal distribution wrt halomass """
        if hasattr(mhalo, "__len__"):
            a = np.zeros((len(mhalo), len(self.z)))
            for i in range(len(mhalo)):
                if mhalo[i] < self.Meffmax:
                    a[i, :] = self.etamax * np.exp(-(np.log(mhalo[i]) - np.log(self.Meffmax))**2 / (2 * self.sigmaMh**2))
                else:
                    a[i, :] = self.etamax * np.exp(-(np.log(mhalo[i]) - np.log(self.Meffmax))**2 / (2 * self.sigpow**2))
        else:
            if mhalo < self.Meffmax:
                a = self.etamax * np.exp(-(log(mhalo) - log(self.Meffmax))**2 / (2 * self.sigmaMh**2))
            else:
                a = self.etamax * np.exp(-(log(mhalo) - log(self.Meffmax))**2 / (2 * self.sigpow**2))
        return a

    def Mdot(self, mhalo):
        use_mean = True
        if use_mean:
            a = 46.1*(1 + 1.11*self.z) * \
                np.sqrt(self.cosmo.Om0 * (1 + self.z)**3 + self.cosmo.Ode0)
            b = (mhalo / 1.0e12)**1.1
            return np.outer(b, a)
        else:
            a = 25.3*(1 + 1.65*self.z) * \
                np.sqrt(self.cosmo.Om0*(1 + self.z)**3 + self.cosmo.Ode0)
            b = (mhalo / 1.0e12)**1.1
            return np.outer(b, a)

    def sfr(self, mhalo):
        sfrmhdot = self.sfr_mhdot(mhalo)
        mhdot = self.Mdot(mhalo)
        f_b = self.cosmo.Ob(self.z)/self.cosmo.Om(self.z)
        return mhdot * f_b * sfrmhdot

    def djc_dlnMh(self):
        fsub = 0.134
        """fraction of the mass of the halo that is in form of
        sub-halos. We have to take this into account while calculating the
        star formation rate of the central halos. It should be calulated by
        accounting for this fraction of the subhalo mass in the halo mass
        central halo mass in this case is (1-f_sub)*mh where mh is the total
        mass of the halo.
        for a given halo mass, f_sub is calculated by taking the first moment
        of the sub-halo mf and and integrating it over all the subhalo masses
        and dividing it by the total halo mass.
        """
        # a = np.zeros((len(self.snu_eff[:, 0]), len(self.mh), len(self.z)))
        snu = self.snu_unfilt
        a = np.zeros((len(snu[:, 0]), len(self.mh), len(self.z)))
        rest = self.sfr(self.mh*(1-fsub))*(1 + self.z) *\
            self.cosmo.comoving_distance(self.z).value**2/KC
        for f in range(len(snu[:, 0])):
            a[f, :, :] = rest*snu[f, :]
        return a

    def subhmf(self, mhalo, ms):
        # subhalo mass function from (https://arxiv.org/pdf/0909.1325.pdf)
        return 0.13*(ms/mhalo)**(-0.7)*np.exp(-9.9*(ms/mhalo)**2.5)*np.log(10)

    def msub(self, mhalo):
        """
        for a given halo mass mh, the subhalo masses would range from
        m_min to mh. For now, m_min has been taken as 10^5 solar masses
        """
        log10msub_min = 5
        """
        if np.log10(mhalo) <= log10msub_min:
            #raise ValueError, "halo mass %d should be greater than subhalo mass %d." % (np.log10(mhalo), log10msub_min)
        else:
            logmh = np.log10(mhalo)
            logmsub = np.arange(log10msub_min, logmh, 0.1)
            return 10**logmsub
        """
        logmh = np.log10(mhalo)
        logmsub = np.arange(log10msub_min, logmh, 0.1)
        return 10**logmsub


    def djsub_dlnMh(self):
        """
        for subhalos, the SFR is calculated in two ways and the minimum of the
        two is assumed.
        """
        fsub = 0.134
        # a = np.zeros((len(self.snu_eff[:, 0]), len(self.mh), len(self.z)))
        snu = self.snu_unfilt
        a = np.zeros((len(snu[:, 0]), len(self.mh), len(self.z)))
        # sfrmh = self.sfr(mh)
        for i in range(len(self.mh)):
            ms = self.msub(self.mh[i]*(1-fsub))
            dlnmsub = np.log10(ms[1] / ms[0])
            sfrI = self.sfr(ms)  # dim(len(ms), len(z))
            sfrII = self.sfr(self.mh[i]*(1-fsub))*ms[:, None]/(self.mh[i]*(1-fsub))
            # sfrII = sfrmh[i] * ms / mh[i]
            sfrsub = np.zeros((len(ms), len(self.z)))
            for j in range(len(ms)):
                sfrsub[j, :] = np.minimum(sfrI[j, :], sfrII[j, :])
            integral = self.subhmf(self.mh[i], ms)[:, None]*sfrsub / KC
            intgn = intg.simps(integral, dx=dlnmsub, axis=0)
            a[:, i, :] = snu*(1 + self.z)*intgn *\
                self.cosmo.comoving_distance(self.z).value**2
        return a

    def J_nu_iv(self):  # , Meffmax, etamax, sigmaMh, alpha):
        # integrated differential emissivity over all the masses
        dj_cen, dj_sub = self.djc_dlnMh(), self.djsub_dlnMh()
        intgral1 = dj_cen+dj_sub
        intgral1 *= self.hmfmz
        # dm = np.log10(self.mh[1] / self.mh[0])
        # return intg.simps(intgral1, dx=dm, axis=1, even='avg')
        return intg.simps(intgral1, x=np.log10(self.mh), axis=1, even='avg')

    def Iv(self):  # , Meffmax, etamax, sigmaMh, alpha):
        jnu = self.J_nu_iv()
        dchi_dz = c_light/(self.cosmo.H0*np.sqrt((self.cosmo.Om0)*(1+self.z)**3 + self.cosmo.Ode0)).value
        intgral2 = dchi_dz*jnu/(1+self.z)
        # result = self.cc_cibmean*self.freq_Iv*intg.simps(intgral2, x=self.z,
        #                                                  axis=-1, even='avg')
        result = intg.simps(intgral2, x=self.z, axis=-1, even='avg')
        Ivint = interp1d(self.nu0, result, kind='linear', bounds_error=False,
                         fill_value="extrapolate")
        # result *= ghz*nW/w_jy  # nWm^2/sr
        return Ivint

    def alpha(self):
        """
        Doing integration by parts in the numerator. That gives first term as
        I_obs * nu0 * W(nu0) evaluated at nu0_min and nu0_max. If we assume
        that the filter at both nu0_min and nu0_max is zero, then this first
        term disappears. So I am solving only the second part of integration
        by parts.
        """
        if self.dv.name == 'Planck_only':
            """
            first = self.Iv()(self.nu0max)*self.nu0max*self.filt[self.nucen](self.nu0max)
            second = self.Iv()(self.nu0min)*self.nu0min*self.filt[self.nucen](self.nu0min)
            num1 = first-second
            numint = self.Iv()(self.nu0)
            numint *= self.filt[self.nucen](self.nu0) + self.nu0*self.filtgrad[self.nucen](self.nu0)
            num2 = intg.simps(numint, x=self.nu0, axis=0, even='avg')
            num = num1-num2
            """
            Inugrad = np.gradient(self.Iv()(self.nu0), self.nu0)
            numint = Inugrad*self.nu0*self.filt[self.nucen](self.nu0)
            num = intg.simps(numint, x=self.nu0, axis=0, even='avg')
            # """
            denomint = self.Iv()(self.nu0)*self.filt[self.nucen](self.nu0)
            denom = intg.simps(denomint, x=self.nu0, axis=0, even='avg')
            res = num/denom
        else:
            # print ("here")
            a = np.gradient(np.log(self.Iv()(self.nu0)), np.log(self.nu0))
            # res *= self.nu0/self.Iv()(self.nu0)
            res = interp1d(self.nu0, a, kind='linear', bounds_error=False, fill_value="extrapolate")
        return res
