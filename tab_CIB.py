import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM
from models import HaloModel
import util

CIB = CIBxLIM()
model = HaloModel()

kpp_max = 1
kp_max = 0.5
xpp_max = util.chi(3.5)
xpp_min = util.chi(2.5)
xp_max = util.chi(5)
xp_min = util.chi(0.03)

pp = 2000
p = 1000

xpp_arr = np.linspace(xpp_min, xpp_max, pp)
xp_arr = np.linspace(xp_min, xp_max, p)

kp_arr = np.fft.fftfreq(xp_arr.size, d=xp_arr[1] - xp_arr[0])
print(kp_arr.max())

zarr = CIB.z_from_chi(xp_arr)
larr = 158 * (1 + CIB.z_from_chi(xpp_arr))

print("Calculating CIB...")
b_dI_dz = model.b_dI_dz(larr, zarr) # shape is l, z

print("Performing FFT")
# Fourier Transform b_dI_dz only along the l axis
kpp_arr, res = util.fft_wrapper(b_dI_dz, xpp_arr, axis=0, shift=False)
print(kpp_arr.max())
path = "/mount/citadel1/zz1994/codes/CIBxLIM/tabs/FFT_CIB/1000x2000/"
np.savetxt(path+"xp_arr.txt", xp_arr)
np.savetxt(path+"kpp_arr.txt", kpp_arr)
np.savetxt(path+"real.txt", res.real)
np.savetxt(path+"imag.txt", res.imag)


