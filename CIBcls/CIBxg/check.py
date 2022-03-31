import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

CIBxg = np.loadtxt("CIBxg_real.txt") + 1j*np.loadtxt("CIBxg_imag.txt")
kpps = np.loadtxt("kpps.txt")
ells = np.loadtxt("ells.txt")

real = interp1d(ells, CIBxg[:, 10].real)
imag = interp1d(ells, CIBxg[:, 10].imag)

newells = np.arange(10, 1000)
#plt.plot(ells, CIBxg[:,10].real)
#plt.plot(ells, CIBxg[:,10].imag)
plt.plot(newells, real(newells))
plt.plot(newells, imag(newells))

#plt.xscale("log")
plt.show()



