###################################################################################################
##                     Parameters: https://github.com/cchandre/KH   (TDSE)                       ##
###################################################################################################
import numpy as xp

Method = 'wavefunction'

laser_intensity = 1e15
laser_wavelength = 780
laser_envelope = 'const'
laser_E = lambda phi: -xp.sin(phi)
te = [1, 2, 1]

a = 5
V = lambda r: -1 / xp.sqrt(r**2 + a**2)
InitialState = [(0, 1), 'VKH2']
InitialCoeffs = (1, -1)
DisplayCoord = 'VKH2'

L = 200
N = 2**12
delta = 30
Lg = 200

nsteps_per_period = 1e3
scale = 'linear'

SaveWaveFunction = False
PlotData = True
SaveData = False
dpi = 300
refresh = 50

darkmode = True