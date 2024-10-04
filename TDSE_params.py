###################################################################################################
##                     Parameters: https://github.com/cchandre/KH   (TDSE)                       ##
###################################################################################################
import numpy as xp

Method = 'wavefunction'

laser_intensity = 7.5e13
laser_wavelength = 780
laser_envelope = 'const'
laser_E = lambda phi: [xp.sin(phi), 0]
te = [1, 2, 1]

a = 5
V = lambda r: -1 / xp.sqrt(r**2 + a**2)
InitialState = [(0, 1), 'VKH2']
InitialCoeffs = (1, -1)
DisplayCoord = 'KH2'

L = [100, 50]
N = [2**10, 2**8]
delta = [10, 5]

nsteps_per_period = 1e3
scale = 'log'

SaveWaveFunction = True
PlotData = True
SaveData = False
dpi = 300
refresh = 50

darkmode = False