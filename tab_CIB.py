import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM
from models import HaloModel
import util

"""
Code for tabulating CIB intensity using HaloModel.
"""

CIB = CIBxLIM(want_halo_model=True)
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

#kp_arr = np.fft.fftfreq(xp_arr.size, d=xp_arr[1] - xp_arr[0])
#print(kp_arr.max())

zarr = CIB.z_from_chi(xp_arr)
larr = 158 * (1 + CIB.z_from_chi(xpp_arr))

path = "/mount/citadel1/zz1994/codes/CIBxLIM/tabs/Halo_CIB/1000x2000/"

b_dI_dz = model.b_dI_dz(larr, zarr)
#b_dI_dz = np.ones((xp_arr.size, xpp_arr.size))
print(b_dI_dz.shape)
np.savetxt(path+"xp_arr.txt", xp_arr)
np.savetxt(path+"xpp_arr.txt", xpp_arr)
np.savetxt(path+"b_dI_dz.txt", b_dI_dz)

"""
for di in range(4):
    print("Calculating CIB...")
    b_dI_dz = model.b_dI_dz(larr, zarr, di) # shape is l, z

    print("Performing FFT")
    # Fourier Transform b_dI_dz only along the l axis
    kpp_arr, res = util.fft_wrapper(b_dI_dz, xpp_arr, axis=0, shift=False)
    print(kpp_arr.max())
    path = "/mount/citadel1/zz1994/codes/CIBxLIM/tabs/FFT_CIB/1000x2000/"
    #np.savetxt(path+"xp_arr.txt", xp_arr)
    #np.savetxt(path+"kpp_arr.txt", kpp_arr)
    np.savetxt(path+"real_dp{}.txt".format(di), res.real)
    np.savetxt(path+"imag_dp{}.txt".format(di), res.imag)
"""

