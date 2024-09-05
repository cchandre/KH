###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/KH   (TDSE)               ##
###################################################################################################
import numpy as xp

Method = 'wavefunction'

laser_intensity = 1e15
laser_wavelength = 780
laser_envelope = 'const'
laser_field = lambda phi: [xp.sin(phi), 0]
te = [1, 8, 1]

a = 5
V = lambda r: -1 / xp.sqrt(r**2 + a**2)
InitialState = [(0, 1), 'VKH2']
DisplayCoord = 'KH2'

L = [200, 100]
N = [2**10, 2**9]
delta = [10, 5]
Lg = [200, 100]

nsteps_per_period = 1e3
scale = 'log'

SaveWaveFunction = True
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
if not len(L) == len(N) == len(xp.atleast_1d(laser_field(0))):
    raise ValueError('Dimension of variables in dictionary not compatible')
if not len(delta) == len(L):
    delta = delta[0] * xp.ones_like(L)
if not len(Lg) == len(L):
    Lg = Lg[0] * xp.ones_like(L)
if isinstance(InitialState, int) or isinstance(InitialState, tuple):
    InitialState = [InitialState, 'V']
dict_ = {
        'Method': Method,
        'laser_intensity': laser_intensity,
        'laser_wavelength': laser_wavelength,
        'envelope': laser_envelope,
        'field': laser_field,
        'te': xp.asarray(te),
        'nsteps_per_period': nsteps_per_period,
        'dim': len(L),
        'ncycles': xp.asarray(te).sum(),
        'scale': scale,
        'V': V,
        'InitialState': InitialState,
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
