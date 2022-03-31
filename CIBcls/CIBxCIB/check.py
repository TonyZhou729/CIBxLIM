import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


ells = np.loadtxt("ells.txt")
kpps = np.loadtxt("kpps.txt")
CIBxCIB = np.loadtxt("CIBxCIB.txt")
#print(simps(CIBxCIB[0], x=kpps))


plt.loglog(ells, CIBxCIB[:, 0], label="first")
plt.loglog(ells, CIBxCIB[:, -1], label="last")
plt.legend()
#plt.xscale("log")
plt.show()


