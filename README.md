# CIBxLIM

Code for computing the Cosmic Infrared Background Powerspectrum coupled with Line Intensity signal. 

### Dependencies
- Numpy
- Matplotlib
- Astropy (Fits io and Cosmology)
- Colossus (Halo mass function and Halo Bias) https://bdiemer.bitbucket.io/colossus/index.html
- halomod (Alternative HMF & Halo Bias) https://github.com/halomod/hmf

### Models
- Under LinearModel, the linear, non halo mass dependent model for the CIB emissitivity. Accurate at large scales for the 2-halo terms, conveniently where the CIB signal dominates over the Line Intensity signal. Derivations and fitted parameter found in Maniyar 2018.
- Under HaloModel, the Halo Mass corrected model accuate to small scales within the one halo term according to Maniyar 2020. Also incorporates the subhalo mass function according to Tinker 2018

### Data files
- SED.fits, redshift dependent Spectral Energy Distribution for the luminous star forming galaxy accross the Planck survey frequencies \[100, 143, 217, 353, 545, 867, 3000]
