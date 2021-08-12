from headers_constants import *


class data_var_iv(object):

    def __init__(self, exp):  # , mass):  # , z):  # , ell):
        # ############### cib data #########################
        self.exp = exp
        self.name = self.exp['name']
        # self.gal_exp = self.exp['name_gal']

        self.z_c = 1.5

        # self.cc_cibmean = self.exp['cc_cibmean']
        self.nu0min = self.exp['nu0min']
        self.nu0max = self.exp['nu0max']
        self.nucen = self.exp['nucen']
        self.nu0 = self.exp['nu0']
        self.ell = self.exp['ell']

        # self.mass = mass
        # self.z = z

        self.cc = self.exp['cc']
        self.fc = self.exp['fc']
        # if self.exp['do_cibmean'] == 1:
        self.z_c = 1.5
        # ######### reading and interpolating the SEDs
        snuaddr = self.exp['snuaddr']
        hdulist = fits.open(snuaddr)
        """
        The effective SEDs for the CIB for Planck (100, 143, 217, 353, 545,
        857) and
        IRAS (3000) GHz frequencies.
        Here we are shwoing the CIB power spectra corressponding to the
        Planck
        frequency channels. If you want to calculate the Hershel/Spire
        power spectra, use corresponding files in the data folder.
        """
        redshifts = hdulist[1].data
        snu_eff = hdulist[0].data  # in Jy/Lsun
        hdulist.close()

        self.snufilt = interp1d(redshifts, snu_eff, kind='linear',
                                bounds_error=False, fill_value=0.)
        # ######### unfiltered SEDs ###########################
        # """
        list_of_files = sorted(glob.glob('data_files/TXT_TABLES_2015/./*.txt'))
        a = list_of_files[95]
        b = list_of_files[96]
        for i in range(95, 208):
            list_of_files[i] = list_of_files[i+2]
        list_of_files[208] = a
        list_of_files[209] = b

        wavelengths = np.loadtxt('data_files/TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt')[:, [0]]
        # the above wavelengths are in microns
        freq = c_light/wavelengths
        # c_light is in Km/s, wavelength is in microns and we would like to
        # have frequency in GHz. So gotta multiply by the following
        # numerical factor which comes out to be 1
        # numerical_fac = 1e3*1e6/1e9
        numerical_fac = 1.
        freqhz = freq*1e3*1e6
        freq *= numerical_fac
        freq_rest = freqhz*(1+redshifts)

        n = np.size(wavelengths)

        snu_unfiltered = np.zeros([n, len(redshifts)])
        for i in range(len(list_of_files)):
            snu_unfiltered[:, i] = np.loadtxt(list_of_files[i])[:, 1]
        L_IR15 = self.L_IR(snu_unfiltered, freq_rest, redshifts)
        # print (L_IR15)

        for i in range(len(list_of_files)):
            snu_unfiltered[:, i] = snu_unfiltered[:, i]*L_sun/L_IR15[i]

        # Currently unfiltered snus are ordered in increasing wavelengths,
        # we re-arrange them in increasing frequencies i.e. invert it

        freq = freq[::-1]
        snu_unfiltered = snu_unfiltered[::-1]
        """
        snu_unfiltered = snu_eff
        freq = np.array([100., 143., 217., 353., 545., 857., 3000.])
        # """
        self.unfiltered_snu = RectBivariateSpline(freq, redshifts,
                                                  snu_unfiltered)
        # snuinterp = interp1d(redshifts, snu_eff, kind='linear',
        #                      bounds_error=False, fill_value=0.)
        # snuinterp = interp1d(redshifts, snu_eff, kind='linear',
        #                      bounds_error=False, fill_value="extrapolate")
        # self.snu = snuinterp(z)

        # ############## Planck filter at x GHz ####################
        freqar = ['100', '143', '217', '353', '545', '857']
        # addr_f = 'data_files/filters/HFI__avg_545_CMB_noise_avg_Apod5_Sfull_v302_HNETnorm.dat'
        # filt = np.loadtxt(addr_f)
        # filt_freq = np.zeros((len(filt[:, 1]), len(freqar)))
        # filt_trans = np.zeros((len(filt[:, 1]), len(freqar)))
        self.filt = {}
        self.filtgrad = {}

        # plt.figure()
        for i in range(len(freqar)):
            adf = 'data_files/filters/HFI__avg_%s_CMB_noise_avg_Apod5_Sfull_v302_HNETnorm.dat' % (freqar[i])
            filt = np.loadtxt(adf)
            filt_freq = filt[:, 1]  # GHz
            filt_trans = filt[:, 2]
            area = np.trapz(filt_trans, filt_freq)
            # print (max(filt_trans))
            # plt.plot(filt_freq, filt_trans, label='%s GHz' % (freqar[i]))
            filt_trans /= area
            # print (max(filt_trans))
            self.filt[freqar[i]] = interp1d(filt_freq, filt_trans,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value="extrapolate")
            filt_grad = np.gradient(filt_trans, filt_freq)
            self.filtgrad[freqar[i]] = interp1d(filt_freq, filt_grad,
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value="extrapolate")
        # plt.xlim(0., 1100.)
        # plt.ylim(0., 1.05)
        # plt.xlabel(r'$\nu_{\rm obs}$ [GHz]', fontsize=16)
        # plt.legend(fontsize='12')
        # plt.legend()
        # plt.show()
        # ######### CIB halo model parameters ###################
        cibparresaddr = self.exp['cibpar_resfile']
        self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibparresaddr)[:4, 0]
        # self.Meffmax, self.etamax, self.sigmaMh, self.tau = 8753289339381.791, 0.4028353504978569, 1.807080723258688, 1.2040244128818796

        # if name == 'Planck_only':
            # self.fc[-4:] = np.loadtxt(cibparresaddr)[-4:, 0]

    def L_IR(self, snu_eff, freq_rest, redshifts):
        # freq_rest *= ghz  # GHz to Hz
        fmax = 3.7474057250000e13  # 8 micros in Hz
        fmin = 2.99792458000e11  # 1000 microns in Hz
        no = 10000
        fint = np.linspace(np.log10(fmin), np.log10(fmax), no)
        L_IR_eff = np.zeros((len(redshifts)))
        dfeq = np.array([0.]*no, dtype=float)
        for i in range(len(redshifts)):
            L_feq = snu_eff[:, i]*4*np.pi*(Mpc_to_m*cosmo.luminosity_distance(redshifts[i]).value)**2/(w_jy*(1+redshifts[i]))
            Lint = np.interp(fint, np.log10(np.sort(freq_rest[:, i])),
                             L_feq[::-1])
            dfeq = 10**(fint)
            L_IR_eff[i] = np.trapz(Lint, dfeq)
        return L_IR_eff
