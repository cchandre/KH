###################################################################################################
##                     Parameters: https://github.com/cchandre/KH   (TDSE)                       ##
###################################################################################################
import numpy as xp

Method = 'wavefunction'

laser_intensity = 1e15
laser_wavelength = 780
laser_E = lambda phi: xp.sin(phi)
te = 3

a = 1
V = lambda r: -1 / xp.sqrt(r**2 + a**2)
InitialState = [(0, 1), 'VKH2']
InitialCoeffs = (1, -1)
DisplayCoord = 'KH2'

L = 200
N = 2**10
