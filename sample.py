import numpy as np
import matplotlib.pyplot as plt
from CIBxLIM import CIBxLIM

"""
Sample script for calculation Cls over k_||' axis, at specific values of ells. 
"""


path = "" # Enter output destination.
plot = True # Will plot upon completion, otherwise save to path above.
cross_gal = True # True for cross galaxy, otherwise auto.

CL = CIBxLIM()
kpp_arr = np.logspace(-4, 0, 30, base=10)
#ell_arr = [100, 300, 700, 1000]
ell_arr = [100]

if plot is False:
    np.savetxt(path+"kpp_arr.txt", kpp_arr)
    np.savetxt(path+"ell.txt", ell_arr)

for ell in ell_arr:    
    print("Computing for ell = {}".format(ell))
    if cross_gal:        
        CIB = CL.compute_cl("CIB", "g", kpp_arr, ell, 500, 0, 0.025, over_k=True)
        CII = CL.compute_cl("CII", "g", kpp_arr, ell, 1000, 0, 1, over_k=True)
    else:
        CIB = CL.compute_cl("CIB", "CIB", kpp_arr, ell, 500, 0, 0.025, over_k=True)    
        CII = CL.compute_cl("CII", "CII", kpp_arr, ell, 1000, 0, 1, over_k=True)
    
    if plot:
        if cross_gal:
            plt.plot(kpp_arr, CIB.real, label="Re[CIBxG]")
            plt.plot(kpp_arr, CIB.imag, label="Im[CIBxG]")
            plt.plot(kpp_arr, CII, label="CIIxG")
            plt.plot(kpp_arr, abs(CIB), "--", label="|CIBxG|")
            plt.ylabel(r"$[Jy\ Mpc]$")
            plt.xscale("log")
        else:
            plt.plot(kpp_arr, CIB, label="CIBxCIB")        
            plt.plot(kpp_arr, CII, label="CIIxCII")                    
            plt.ylabel(r"$[Jy^2\ Mpc]$")            
        plt.legend()
        plt.xlabel(r"$k_{\parallel}'$")
        plt.title(ell)
        plt.show()
    else:
        if cross_gal:
            np.savetxt(path+"CIBxg_{}_real.txt".format(ell), CIBxg.real)
            np.savetxt(path+"CIBxg_{}_imag.txt".format(ell), CIBxg.imag)
            np.savetxt(path+"CIIxg_{}.txt".format(ell), CIIxg.value)
        else:
            np.savetxt(path+"CIBxCIB_{}.txt".format(ell), CIBxCIB)
            np.savetxt(path+"CIIxCII_{}.txt".format(ell), CIIxCII)

    print()



