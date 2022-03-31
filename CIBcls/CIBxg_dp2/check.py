import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

CIBxg = np.loadtxt("CIBxg_real.txt") + 1j*np.loadtxt("CIBxg_imag.txt")
kpps = np.loadtxt("kpps.txt")
ells = np.loadtxt("ells.txt")

plt.plot(ells, CIBxg[:, 30].real)
plt.plot(ells, CIBxg[:, 30].imag)
plt.plot(ells, abs(CIBxg[:, 30]), "--")

plt.xscale("log")
plt.yscale("log")
plt.show()



