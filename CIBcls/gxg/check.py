import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


ells = np.loadtxt("ells.txt")
kpps = np.loadtxt("kpps.txt")
gxg = np.loadtxt("gxg.txt")
#print(simps(CIBxCIB[0], x=kpps))


plt.loglog(kpps, gxg[0], label="first")
plt.loglog(kpps, gxg[-1], label="last")
plt.legend()
#plt.xscale("log")
plt.show()


