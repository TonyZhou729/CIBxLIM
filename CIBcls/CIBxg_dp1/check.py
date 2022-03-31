import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

CIBxg = np.loadtxt("CIBxg_real.txt") + 1j*np.loadtxt("CIBxg_imag.txt")
kpps = np.loadtxt("kpps.txt")
ells = np.loadtxt("ells.txt")

plt.plot(kpps, CIBxg[70].real)
plt.plot(kpps, CIBxg[70].imag)


plt.xscale("log")
plt.show()



