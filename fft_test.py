import numpy as np
import matplotlib.pyplot as plt

a = 1
gaussian = lambda x : np.exp(-a * x**2) # FFT test Gaussian Function.
decay = lambda x : np.exp(-a * abs(x))
xmin = -5
xmax = 5

# Trial griding sizes, each value corresponds to number of steps within a xmax-xmin sized window. 
spacings = [300, 200, 100]

for width in spacings:
    x = np.linspace(xmin, xmax, width)
    func = decay(x)        
    # Obtain and 0 center wave array and fft results. 
    k = np.fft.fftshift(np.fft.fftfreq(x.size, d=x[1]-x[0]))        
    res = np.fft.fftshift(np.fft.fft(func)) / width * (xmax-xmin)
    
    # Plot magnitude.
    plt.plot(k, abs(res), label=width)    

# Analytical function of Gaussian Fourier Transform:
#true_res = lambda k : np.sqrt(np.pi/a) * np.exp(-1/a * np.pi**2 * k**2)
true_res = lambda k : 2*a / (a**2 + 4*np.pi**2*k**2)
k = np.linspace(-4, 4, 1000)
plt.plot(k, true_res(k), "--", label="Analytical")

#plt.yscale("log")
plt.legend()
plt.show()


