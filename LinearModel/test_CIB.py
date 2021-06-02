from LinearModel import LinearModel
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import astropy.units as u

LinearModel = LinearModel()

c = const.c * u.m/u.s
freqs = np.array([100, 143, 217, 353, 545, 857, 3000, 127.5, 392.4, 188.8], dtype="float64")
redshifts=LinearModel.redshifts
"""
for l in wavelengths:
    plt.plot(redshifts, LinearModel.CIB_model(l, redshifts)/np.trapz(LinearModel.CIB_model(l, redshifts),x=redshifts), label="{:.3f} micron".format(l))
    #plt.plot(redshifts, LinearModel.j(l, redshifts), label="{:4f} micron".format(l))
#plt.yscale("log")
plt.xlabel("z")
plt.ylabel(r"$dI_{\lambda}/dz$")
plt.xlim((0, 5))
plt.legend()
plt.show()
"""

LinearModel.plot_CIB_model(redshifts, freqs=freqs)


