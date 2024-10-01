###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/KH   (TDSE)               ##
###################################################################################################
import numpy as xp

Method = 'wavefunction'

laser_intensity = 1e15
laser_wavelength = 780
laser_envelope = 'const'
laser_E = lambda phi: xp.sin(phi)
te = [1, 2, 1]

a = 5
V = lambda r: -1 / xp.sqrt(r**2 + a**2)
InitialState = [(0, 1), 'VKH2']
InitialCoeffs = (1, 1)
DisplayCoord = 'KH2'

L = 200
N = 2**10
delta = 10
Lg = 200

nsteps_per_period = 1e3
scale = 'linear'

SaveWaveFunction = False
PlotData = True
SaveData = False
dpi = 300
refresh = 50

darkmode = False

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
L = xp.atleast_1d(L)
N = xp.atleast_1d(N)
delta = xp.atleast_1d(delta)
Lg = xp.atleast_1d(Lg)
laser_E_ = lambda phi: xp.atleast_1d(laser_E(phi))
if not len(L) == len(N) == len(laser_E_(0)):
    raise ValueError('Dimension of variables in dictionary not compatible')
if not len(delta) == len(L):
    delta = delta[0] * xp.ones_like(L)
if not len(Lg) == len(L):
    Lg = Lg[0] * xp.ones_like(L)
if isinstance(InitialState, (int, tuple)):
    InitialState = [InitialState, 'V']
if 'InitialCoeffs' not in locals():
    InitialCoeffs = xp.ones_like(InitialState[0])
dict_ = {
        'Method': Method,
        'laser_intensity': laser_intensity,
        'laser_wavelength': laser_wavelength,
        'envelope': laser_envelope,
        'laser_E': laser_E_,
        'te': xp.asarray(te),
        'nsteps_per_period': nsteps_per_period,
        'dim': len(L),
        'ncycles': xp.asarray(te).sum(),
        'scale': scale,
        'V': V,
        'InitialState': InitialState,
        'InitialCoeffs': InitialCoeffs,
        'DisplayCoord': DisplayCoord,
        'Lg': Lg,
        'L': L,
        'N': N,
        'delta': delta,
        'PlotData': PlotData,
        'SaveWaveFunction': SaveWaveFunction,
        'refresh': refresh,
        'SaveData': SaveData,
        'dpi': dpi}
dict_.setdefault('Nkh', 2**12)
dict_.setdefault('ode_solver', 'BM4')
dict_.setdefault('tol', 1e-10)
dict_.setdefault('maxiter', 1000)
dict_.setdefault('ncv', 100)
###################################################################################################
