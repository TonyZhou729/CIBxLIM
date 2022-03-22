import numpy as np
from scipy import interpolate

z = np.loadtxt("SEDredshifts.txt")
l = np.loadtxt("SEDwavelengths.txt")
data = np.loadtxt("SEDtable.txt")

f = interpolate.interp2d(l, z, data)

new_z = np.linspace(z.min(), z.max(), 1000)
new_l = np.linspace(l.min(), l.max(), 1000)
new_data = f(new_l, new_z)
print(new_data.shape)

