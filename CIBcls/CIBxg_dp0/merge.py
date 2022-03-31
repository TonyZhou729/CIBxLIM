import numpy as np
import matplotlib.pyplot as plt

kpps = np.loadtxt("dr1/kpps.txt")

ell1 = np.loadtxt("dr1/ells.txt")
ell2 = np.loadtxt("dr2/ells.txt")

real1 = np.loadtxt("dr1/CIBxg_real.txt")
real2 = np.loadtxt("dr2/CIBxg_real.txt")

imag1 = np.loadtxt("dr1/CIBxg_imag.txt")
imag2 = np.loadtxt("dr2/CIBxg_imag.txt")

print(ell1)
print(ell2)

ell_merged = np.zeros(ell1.size+ell2.size)
real = np.zeros((ell_merged.size, kpps.size), dtype="float64")
imag = np.zeros((ell_merged.size, kpps.size), dtype="float64")


"""
for i in range(0, ell_merged.size, 2):
    ell_merged[i] = ell1[i]
    ell_merged[i+1] = ell2[i]
print(ell_merged)
"""
for i in range(ell1.size):
    ell_merged[i*2] = ell1[i]
    real[i*2] = real1[i]
    imag[i*2] = imag1[i]
for i in range(ell2.size):
    ell_merged[i*2 + 1] = ell2[i]
    real[i*2 + 1] = real2[i]
    imag[i*2 + 1] = imag2[i]
print(ell_merged)
print(real1.shape)
print(real2.shape)
print(real.shape)

np.savetxt("ells.txt", ell_merged)
np.savetxt("CIBxg_real.txt", real)
np.savetxt("CIBxg_imag.txt", imag)



