import numpy as np
import vegas
import matplotlib.pyplot as plt
import time

N=2e5

xmax = np.pi
xmin = 0

t = 0

@vegas.batchintegrand
def f_batch(x):
    return np.sin(x[:, 0]-x[:, 1])

def VG_batch(f_batch):
    integ = vegas.Integrator([[xmin, xmax], [xmin, xmax]])
    integ(f_batch, nitn=10, neval=N)
    ret = integ(f_batch, nitn=10, neval=N)
    return ret

print(VG_batch(f_batch).summary())
