#
# BSD 2-Clause License
#
# Copyright (c) 2024, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as xp
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq, rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
import scipy.sparse.linalg as la
from typing import Tuple, List

def generate_dict(self) -> dict:
    dict_ = dict()
    compulsory_params = ['Method', 'laser_intensity', 'laser_wavelength', 'laser_E', 'V', 'InitialState', 'L', 'N', 'te']
    extra_params = ['legend', 'xlim', 'ylim', 'figsize', 'InitialCoeffs']
    default_params = ['envelope', 'nsteps_per_period', 'refresh', 'scale', 'dpi', 'DisplayCoord', 'Nkh', 'ode_solver', 'tol', 'maxiter', 'ncv', 'PlotData', 'SaveData', 'SaveWaveFunction']
    default_vals = ['const', 1e3, 50, 'linear', 300, 'lab', 2**12, 'BM4', 1e-10, 1000, 100, True, False, False]
    for param in compulsory_params:
        if hasattr(self, param):
            dict_.update({param: getattr(self, param)})
        else:
            raise ValueError(f'{param} is underfined')
    dict_['L'], dict_['N'] = xp.atleast_1d(self.L), xp.atleast_1d(self.N)
    dict_['te'] = xp.atleast_1d(self.te)
    dict_.update({'ncycles': dict_['te'].sum()}) 
    Lg = self.Lg if hasattr(self, 'Lg') else dict_['L'].copy()
    delta = self.delta if hasattr(self, 'delta') else dict_['L'] // 10
    delta, Lg = xp.atleast_1d(delta), xp.atleast_1d(Lg)
    dim = len(xp.atleast_1d(self.laser_E(0)))
    if not len(delta) == len(dict_['L']):
        delta = delta[0] * xp.ones_like(dict_['L'])
    if not len(Lg) == len(dict_['L']):
        Lg = Lg[0] * xp.ones_like(dict_['L'])
    dict_.update({'Lg': Lg, 'delta': delta, 'dim': dim})
    if isinstance(self.InitialState, (int, tuple, type(lambda:0))):
        dict_['InitialState'] = [self.InitialState, 'V']
    if not hasattr(self, 'InitialCoeffs') and isinstance(dict_['InitialState'][0], (int, tuple)):
        dict_.update({'InitialCoeffs': xp.ones_like(dict_['InitialState'][0])})
    for param in extra_params:
        if hasattr(self, param):
            dict_.update({param: getattr(self, param)})
    for param, val in zip(default_params, default_vals):
        dict_.update({param: val if not hasattr(self, param) else getattr(self, param)})
    if len(dict_['te']) != 3 and dict_['envelope'] != 'const':
        raise ValueError('The field envelope requires te to have 3 values (ramp-up, plateau, ramp-down)')
    if dict_['Method'] == 'Husimi':
        if not hasattr(self, 'p_husimi') or not hasattr(self, 'sigma_husimi'):
            raise ValueError('The parameters p_husimi and/or sigma_husimi are not defined')
        else:
            dict_.update({'p_husimi': self.p_husimi, 'sigma_husimi': self.sigma_husimi})
    return dict_

class TDSE:
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.DictParams})'

    def __str__(self) -> str:
        return f'Time-dependent Schrödinger equation ({self.__class__.__name__})'

    def __init__(self, params) -> None:
        dict_ = generate_dict(params)
        for key in dict_:
            setattr(self, key, dict_[key])
        self.DictParams = dict_
        self.E0 = xp.sqrt(self.laser_intensity) * 5.33803e-9
        self.omega = 2 * xp.pi * 299792458 / 41341000 / self.laser_wavelength
        self.T = 2 * xp.pi / self.omega
        self.Up = self.E0**2 / (4 * self.omega**2)
        self.q0 = self.E0 / self.omega**2
        self.E = xp.vectorize(lambda t: self.E0 * self.env(t) * xp.atleast_1d(self.laser_E(self.omega * t)), signature='()->(n)')
        self.te = self.te * self.T
        self.final_time = xp.sum(self.te)
        self.step = self.T / self.nsteps_per_period
        self.vecx = tuple([xp.linspace(-L, L, N, endpoint=False, dtype=xp.float64) for L, N in zip(self.L, self.N)])
        self.xgrid = xp.asarray(xp.meshgrid(*self.vecx, indexing='ij'))
        absfunc = lambda x, L, delta: xp.where(xp.abs(x)>=L-delta, xp.abs(xp.cos(xp.pi/2 * (xp.abs(x)-L+delta)/delta))**(1/8), 1)
        self.Abs = xp.prod(xp.asarray([absfunc(x, L, delta) for x, L, delta in zip(self.xgrid, self.L, self.delta)]), axis=0)
        self.dx = xp.asarray([2 * L / N for L, N in zip(self.L, self.N)])
        veck = tuple([xp.pi / L * fftfreq(N, d=1/N) for L, N in zip(self.L, self.N)])
        self.kgrid = xp.asarray(xp.meshgrid(*veck, indexing='ij'))
        self.Lap = xp.sum(self.kgrid**2, axis=0) / 2
        self.Vgrid = self.V(xp.sqrt((self.xgrid**2).sum(axis=0)))
        self.xshape = self.Vgrid.shape
        self.dim_ext = (self.dim,) + self.dim * (1,)
        self.Tavg = self.T if self.envelope=='const' else self.final_time
        if 'KH' in self.DisplayCoord + self.InitialState[1]:
            self.t_, self.A_, self.q_, self.phib_ = self.compute_stflds()

    def eigenstates(self, V:xp.ndarray, k:int, output:str='last'):
        indx = [[xp.abs(self.vecx[_] + self.Lg[_]).argmin(), xp.abs(self.vecx[_] - self.Lg[_]).argmin()] for _ in range(self.dim)]
        rgx = tuple([xp.arange(*indx[_]) for _ in range(self.dim)])
        Lg = [self.vecx[_][indx[_][1]] for _ in range(self.dim)]
        Ng = [len(rgx[_]) for _ in range(self.dim)]
        ixgrid = xp.meshgrid(*rgx, indexing='ij')
        veck = tuple([xp.pi / L * fftfreq(N, d=1/N) for L, N in zip(Lg, Ng)])
        kg = xp.asarray(xp.meshgrid(*veck, indexing='ij'))
        Lap = xp.sum(kg**2, axis=0) / 2
        Vg = V[tuple(ixgrid)]
        Nt = xp.prod(Ng)
        H = lambda psi: xp.real(ifftn(Lap * fftn(psi.reshape(Ng))) + Vg * psi.reshape(Ng)).flatten()
        lam, v = la.eigsh(la.LinearOperator((Nt, Nt), matvec=H), which='SA', k=k, tol=self.tol, maxiter=self.maxiter, ncv=self.ncv)
        if output == 'last':
            psi = xp.zeros(self.xshape, dtype=xp.float64)
            psi[tuple(ixgrid)] = v[:, -1].reshape(Ng) / self.norm(v[:, -1].reshape(Ng))
            err = xp.abs(xp.sum(psi * (ifftn(self.Lap * fftn(psi)) + V * psi - lam[-1] * psi)) * xp.prod(self.dx))
            return lam[-1], psi, err
        elif output == 'all':
            psi = xp.zeros((k,) + self.xshape, dtype=xp.float64)
            err = xp.zeros(k)
            for _ in range(k):
                psi[_][tuple(ixgrid)] = v[:, _].reshape(Ng) / self.norm(v[:, _].reshape(Ng))
                err[_] = xp.abs(xp.sum(psi[_].conj() * (ifftn(self.Lap * fftn(psi[_])) + V * psi[_] - lam[_] * psi[_])) * xp.prod(self.dx))
            return lam, psi, err
        
    def compute_Ip(self):
        psi = self.eigenstates(self.Vgrid, 1)[1]
        Ip = xp.sum(psi.conj() * (ifftn(self.Lap * fftn(psi)) + self.Vgrid * psi)) * xp.prod(self.dx)
        return -Ip.real

    def quantum_numbers(self, psi:xp.ndarray) -> List[float]:
        axis = tuple(range(1, self.dim + 1))
        dim_cross = 2 * self.dim - 3
        Ppsi = ifftn(self.kgrid * fftn(psi[xp.newaxis], axes=axis), axes=axis)
        L = xp.real(xp.sum(xp.conj(psi[xp.newaxis]) * xp.cross(self.xgrid, Ppsi, axis=0).reshape((dim_cross,) + self.xshape), axis=axis) * xp.prod(self.dx))
        if self.dim == 2:
            return L[0]
        elif self.dim == 3:
            return -0.5 + xp.sqrt(0.25 + (L**2).sum()), L[-1]

    def antiderivative(self, vec:xp.ndarray) -> xp.ndarray:
        nu = 2 * xp.pi / self.Tavg * rfftfreq(self.Nkh, d=1/self.Nkh)
        div = xp.divide(1, 1j * nu, where=nu!=0)
        div[0] = 0
        dim = (-1,) + (vec.ndim - 1) * (1,)
        return irfft(div.reshape(dim) * rfft(vec, axis=0), axis=0).reshape(vec.shape)

    def compute_stflds(self) -> Tuple[float, xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
        t = xp.linspace(0, self.Tavg, self.Nkh, endpoint=False)
        A = -self.antiderivative(self.E(t))
        q = self.antiderivative(A)
        if '3' in self.InitialState[1] + self.DisplayCoord or self.Method == 'plot_potentials':
            phib = -self.antiderivative(self.V(xp.sqrt(((self.xgrid[xp.newaxis] + q.reshape((-1,) + self.dim_ext))**2).sum(axis=1))))
            return t, A, q, phib
        else:
            return t, A, q, []

    def kh_potential(self, order:int=2) -> xp.ndarray:
        q = self.q_.reshape((-1,) + self.dim_ext)
        V2 = self.V(xp.sqrt(((self.xgrid[xp.newaxis] + q)**2).sum(axis=1))).mean(axis=0)
        if order == 2:
            return V2
        elif order == 3:
            xaxis = tuple(range(2, self.dim + 2))
            phib = self.phib_[:, xp.newaxis, ...]
            Dphib = ifftn(1j * self.kgrid[xp.newaxis] * fftn(phib, axes=xaxis), axes=xaxis)
            f = (Dphib**2).sum(axis=1).real / 2
            return V2 + f.mean(axis=0).reshape(self.xshape)

    def lab2kh(self, t:float, psi:xp.ndarray, order:int=2, dir:str='lab2kh') -> xp.ndarray:
        if self.envelope == 'const':
            t = t % self.T 
        q = interp1d(self.t_, self.q_, axis=0, kind='quadratic', bounds_error=False, fill_value='extrapolate')(t)
        A = interp1d(self.t_, self.A_, axis=0, kind='quadratic', bounds_error=False, fill_value='extrapolate')(t)
        if order == 3:
            phib = interp1d(self.t_, self.phib_, axis=0, kind='quadratic', bounds_error=False, fill_value='extrapolate')(t).reshape(self.xshape)
        expq = xp.exp(1j * xp.einsum('i...,...i->...', self.kgrid, q))
        if dir == 'lab2kh':
            psi_ = ifftn(expq * fftn(psi))
            phase = -xp.einsum('i...,...i->...', self.xgrid + q.reshape(self.dim_ext), A)
            if order == 3:
                phase -= phib
        elif dir == 'kh2lab':
            psi_ = ifftn(xp.conj(expq) * fftn(psi))
            phase = xp.einsum('i...,...i->...', self.xgrid, A)
            if order == 3:
                phase += (ifftn(xp.conj(expq) * fftn(phib))).real
        return (psi_ * xp.exp(1j * phase)).reshape(self.xshape)
    
    def change_frame(self, t:float, psi:xp.ndarray) -> xp.ndarray:
        if 'KH' in self.DisplayCoord:
            return self.lab2kh(t, psi, order=int(self.DisplayCoord[-1]))
        return psi

    def env(self, t:float) -> xp.ndarray:
        te = xp.cumsum(self.te)
        if self.envelope == 'const':
            return 1
        elif self.envelope == 'sinus':
            return xp.where(t<=0, 0, xp.where(t<=te[0], xp.sin(xp.pi * t / (2 * te[0]))**2, xp.where(t<=te[1], 1, xp.where(t<=te[2], xp.sin(xp.pi * (te[2] - t) / (2 * self.te[2]))**2, 0))))
        elif self.envelope == 'trapez':
            return xp.where(t<=0, 0, xp.where(t<=te[0], t / te[0], xp.where(t<=te[1], 1, xp.where(t<=te[2], (te[2] - t) / self.te[2], 0))))
        else:
            raise ValueError(f'The field envelope {self.envelope} is not defined')

    def norm(self, psi:xp.ndarray) -> float:
        return xp.sqrt(xp.sum(xp.abs(psi)**2) * xp.prod(self.dx))

    def dipole(self, t:float, psi:xp.ndarray) -> xp.ndarray:
        dd = xp.squeeze(self.xgrid)
        DVgrid = ifftn(1j * self.kgrid * fftn(self.Vgrid))
        da = -DVgrid.real - self.E(t)
        return (xp.sum(xp.abs(psi)**2 * dd) * xp.prod(self.dx)).flatten(), (xp.sum(xp.abs(psi)**2 * da) * xp.prod(self.dx)).flatten()

    def compute_spectrum(self, t:float, vec:xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        npoints = len(vec[0])
        vec_ = vec * hann(npoints)[xp.newaxis]
        f_hhg = 2 * xp.pi / t * rfftfreq(npoints, d=1/npoints)
        return f_hhg / self.omega, [xp.abs(-rfft(vec_[0]) * f_hhg**2)**2, xp.abs(rfft(vec_[1]))**2]
    
    def compute_husimi(self, psi:xp.ndarray, p:xp.ndarray, sigma:float) -> xp.ndarray:
        if self.dim != 1:
            raise ValueError(f'Computation of Husimi representation not implemented for dimension {self.dim}')
        xgrid, pgrid = xp.meshgrid(xp.squeeze(self.xgrid), p)
        xgrid, pgrid = xgrid[xp.newaxis], pgrid[xp.newaxis]
        psig = lambda x: xp.exp(-(x - xgrid)**2 / (2 * sigma**2) - 1j * x * pgrid) / (xp.pi * sigma**2)**0.25
        return (xp.abs(xp.sum(psig(self.xgrid.reshape((-1, 1, 1))) * psi, axis=0) * xp.prod(self.dx)))**2 

    def plot(self, ax, h, t:float, vec:xp.ndarray) -> None:
        if self.Method in ['wavefunction', 'ionization']:
            psi = self.change_frame(t, vec)
            if self.dim == 1:
                h.set_ydata(xp.abs(psi)**2)
                if hasattr(self, 'ylim') and self.ylim == 'auto':
                    ax.set_ylim((0, 1.1 * max(xp.abs(psi)**2)))
            elif self.dim == 2:
                h.set_data(xp.abs(psi).T**2)
                if self.scale == 'linear':
                    h.set_clim(0, xp.abs(psi).max()**2)
        elif self.Method == 'HHG':
            if t > 0:
                freq, spectrum = self.compute_spectrum(t, vec)
                h[0].set_data((freq[1:], spectrum[0][1:]))
                h[1].set_data((freq[1:], spectrum[1][1:]))
                ax.set_xlim((1, max(freq)))
                if xp.any(spectrum[0]):
                    ax.set_ylim((min(spectrum[0][1:]), max(spectrum[0][1:])))
        elif self.Method == 'Husimi':
            psi = self.change_frame(t, vec)
            h.set_data(self.compute_husimi(psi, self.p_husimi, self.sigma_husimi))
        ax.set_title(f'$t / T = {{{t / self.T:.2f}}}$', loc='right', pad=20)
        plt.pause(1e-4)           

    def save(self, t:float, psi:xp.ndarray, t_vec, psi_vec:xp.ndarray):
        if self.SaveWaveFunction or self.SaveData:
            if t_vec is None:
                t_vec = [t]
            else:
                t_vec.append(t)
            psi_ = self.change_frame(t, psi)
            psi_vec = psi_[..., xp.newaxis] if psi_vec is None else xp.concatenate((psi_vec, psi_[..., xp.newaxis]), axis=-1)
        return t_vec, psi_vec
    
    def chi(self, h:float, t:float, psi:xp.ndarray) -> xp.ndarray:
        psi = ifftn(xp.exp(-1j * self.Lap * h) * fftn(psi))
        Vgrid = self.Vgrid + xp.einsum('i...,...i->...', self.xgrid, self.E(t))
        return xp.exp(-1j * Vgrid * h) * psi * self.Abs
    
    def chi_star(self, h:float, t:float, psi:xp.ndarray) -> xp.ndarray:
        Vgrid = self.Vgrid + xp.einsum('i...,...i->...', self.xgrid, self.E(t))
        psi = xp.exp(-1j * Vgrid * h) * psi
        return ifftn(xp.exp(-1j * self.Lap * h) * fftn(psi)) * self.Abs
    
    def initcond(self) -> Tuple[xp.ndarray, float, float]:
        if isinstance(self.InitialState[0], (int, tuple)):
            if self.InitialState[1] == 'V':
                Vgrid_ = self.Vgrid.copy()
            elif 'KH' in self.InitialState[1]:
                Vgrid_ = self.kh_potential(int(self.InitialState[1][-1]))
            num_init = len(xp.atleast_1d(self.InitialState[0]))
            max_init = max(xp.atleast_1d(self.InitialState[0])) + 1
            lam, psi, err = self.eigenstates(Vgrid_, max_init, output='all' if num_init >= 2 else 'last')
            psi_ = xp.sum(xp.asarray([self.InitialCoeffs[_] * psi[_] for _ in range(num_init)]), axis=0) / xp.sqrt(num_init) if num_init >= 2 else psi
            lam_ = float(lam) if num_init == 1 else lam[0]
            err_ = max(err) if num_init >= 2 else float(err)
        elif isinstance(self.InitialState[0], type(lambda:0)):
            psi_ = self.InitialState[0](xp.squeeze(self.xgrid))
            psi_ /= self.norm(psi_)
            lam_, err_ = [], []
        if 'KH' in self.InitialState[1]:
            psi_ = self.lab2kh(0, psi_, order=int(self.InitialState[1][-1]), dir='kh2lab')  
        return psi_, lam_, err_
