import numpy as np
import os

# Read all files in this directory and compiler into 

path = "/mount/citadel1/zz1994/codes/CIBxLIM/TXT_TABLES_2015"

counter = -1 # x dimension, redshifts

filenames = {} # Empty dictionary for redshift ---> filename
z_array = []

for f in os.listdir(path):
    if f.endswith(".txt"):
        head = os.path.splitext(f)[0]
        redshift_wz = head.split("_")[2]
        z = float(redshift_wz[1:])
        z_array.append(z)
        filenames[z] = os.path.join(path, f)

z_array = np.array(z_array)
z_array = z_array[np.argsort(z_array)] # Sort z_array by redshift value
l_array = np.loadtxt(filenames[z_array[0]])[:, 0] # Wavelengths in microns

np.savetxt("SEDredshifts.txt", z_array)
np.savetxt("SEDwavelengths.txt", l_array)

"""
# Check that all files have the same wavelength array.

all_close = True
for z in z_array:
    this_l_array = np.loadtxt(filenames[z])[:, 0]
    if np.allclose(l_array, this_l_array) is False:
        all_close = False

print(all_close)
"""

data = np.zeros((z_array.size, l_array.size), dtype="float64")
for i, z in enumerate(z_array):
    data_at_z = np.loadtxt(filenames[z])[:, 1]
    data[i] = data_at_z

#np.savetxt("SEDtable.txt", data)
