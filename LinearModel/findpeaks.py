import numpy as np
import matplotlib.pyplot as plt
from LinearModel import LinearModel
from astropy.cosmology import Planck18_arXiv_v2 as cosmo

m = LinearModel()

zpeaks = np.zeros(m.wavelengths.size, dtype="float64") # Stores the peak redshift at each wavelength.
for i, l in enumerate(m.wavelengths):
    _max = m.SED[i].max()
    pos = np.where(m.SED[i] == _max)
    zpeak = m.redshifts[pos]
    zpeaks[i] = zpeak

chi_peaks = np.array(cosmo.comoving_distance(zpeaks))
np.savetxt("../SEDdata/chi_peaks.txt", chi_peaks)
