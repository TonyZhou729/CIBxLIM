import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM

CIB = CIBxLIM()
larr = np.linspace(158*3.5, 158*4.5, 100)
zarr = np.linspace(CIB.z_min, CIB.z_max, 101)
kp = np.linspace(0, 1, 50)
ell = np.arange(103)

ll, zz, kpkp, ellell = np.meshgrid(larr, zarr, kp, ell)
func = CIB.fourier_func(ll, zz, kpkp, ellell)
print(func.shape)

"""
func = CIB.CIBmodel.CIB_model(ll, zz) / CIB.chi_p(ll)
print(func.shape)
func_k = np.fft.fft(func, axis=0)
freq = np.fft.fftfreq(larr.size, d=larr[1]-larr[0])
plt.plot(freq, func_k[3])
plt.show()
"""
