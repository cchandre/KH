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
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from pyhamsys import solve_ivp_symp
import os
import warnings
import time
from datetime import date
from TDSE_classes import TDSE
import TDSE_params 

darkmode = TDSE_params.darkmode if hasattr(TDSE_params, 'darkmode') else False
if darkmode:
	cs = ['k', 'w', 'c', 'm', 'r']
else:
	cs = ['w', 'k', 'c', 'm', 'r']
cmap_psi = 'bwr'
	
plt.rc('figure', facecolor=cs[0], titlesize=30)
plt.rc('text', usetex=True, color=cs[1])
plt.rc('font', family='sans-serif', size=20)
plt.rc('axes', facecolor=cs[0], edgecolor=cs[1], labelsize=26, labelcolor=cs[1], titlecolor=cs[1])
plt.rc('xtick', color=cs[1], labelcolor=cs[1])
plt.rc('ytick', color=cs[1], labelcolor=cs[1])
plt.rc('lines', linewidth=3)
plt.rc('image', cmap='bwr')

def main() -> None:
	self = TDSE(TDSE_params)
	print(f'\033[92m  {self} \033[00m')
	filestr = type(self).__name__ + '_' + time.strftime('%Y%m%d_%H%M')

	if self.Method == 'eigenstates':
		start = time.time()
		if self.InitialState[1] == 'V':
			Vgrid = self.Vgrid.copy()
		elif 'KH' in self.InitialState[1]:
			Vgrid = self.kh_potential(int(self.InitialState[1][-1]))
		lam, psi, err = self.eigenstates(Vgrid, max(xp.atleast_1d(self.InitialState[0])) + 1, output='all')
		fig, ax = display_axes(self, [lam, psi, err], type='eigenstates')
		print(f'\033[90m        Computation of the following eigenstates finished in {int(time.time() - start)} seconds \033[00m')
		for _ in range(max(xp.atleast_1d(self.InitialState[0])) + 1):
			message = f'\033[90m        '
			if self.dim == 1:
				message += f'Eigenstate {_} : E0 = {lam[_]:.6f} (err={err[_]:.1e})'
			elif self.dim == 2:
				L = self.quantum_numbers(psi[_])
				message += f' with Lz = {L:.2f}'
			elif self.dim == 3:
				L, Lz = self.quantum_numbers(psi[_])
				message += f' with L = {L:.2f} and Lz = {Lz:.2f} \033[00m'
			print(message + ' \033[00m')
	elif self.Method in ['wavefunction', 'HHG', 'ionization', 'Husimi']:
		plt.ion()
		start = time.time()
		psi0, lam0, err = self.initcond()
		if isinstance(self.InitialState[0], (int, tuple)):
			print(f'\033[90m        Computation of the initial state finished in {int(time.time() - start)} seconds: E0 = {lam0:.6f} (with err = {err:.2e}) \033[00m')
			if self.dim >= 2:
				print(f'\033[90m                   with quantum number(s):  L = {self.quantum_numbers(psi0)} \033[00m')
		
		if self.PlotData:
			fig, ax, h = display_axes(self, self.change_frame(0, psi0), type=self.Method)

		start = time.time()

		def plot_command(t:float, psi:xp.ndarray) -> None:
			if self.Method == 'HHG':
				dipole = xp.asarray(self.dipole(t, psi))
				self.hhg = xp.concatenate((self.hhg, dipole), axis=1) if hasattr(self, 'hhg') else dipole
			n = int(t / self.step)
			if (n+1)%self.refresh == 0 and self.PlotData:
				if self.Method in ['wavefunction', 'Husimi', 'ionization']:
					vec = psi.copy()
				elif self.Method == 'HHG':
					vec = self.hhg.copy()
				self.plot(ax, h, t, vec)
		
		tspan = xp.linspace(0, self.final_time, int(self.ncycles * self.nsteps_per_period // self.refresh))
		sol = solve_ivp_symp(self.chi, self.chi_star, (0, self.final_time), psi0, step=self.step, t_eval=tspan, method=self.ode_solver, command=lambda t, psi:plot_command(t, psi))
		print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
		save_data(self, xp.array([sol.t, sol.y], dtype=xp.object_), filestr)	

		if self.SaveWaveFunction:
			fig, ax, h = display_axes(self, self.change_frame(0, psi0), type=self.Method)
			def animate(_):
				vec = sol.y[..., _] if self.Method in ['wavefunction', 'ionization'] else (self.hhg[:, :_] if self.Method == 'HHG' else [])
				self.plot(ax, h, sol.t[_], vec)
				return h
			FuncAnimation(fig, animate, frames=len(sol.t), interval=200).save(filestr + '.gif', writer=PillowWriter(fps=5), dpi=self.dpi)
			print(f'\033[90m        Animation saved in {filestr}.gif \033[00m')

		if self.Method == 'ionization':
			proba = 1 - self.norm(sol.y[..., -1])**2 / self.norm(psi0)**2
			print(f'\033[96m          for E0 = {self.E0:.3e}, ionization probability = {proba:.2e} \033[00m')
			vec_data = [self.E0, proba]
			file = open('TDSE_' + self.Method + '.txt', 'a')
			if os.path.getsize(file.name) == 0:
				file.writelines('%   E0           proba \n')
			file.writelines(' '.join([f'{data:.6e}' for data in vec_data]) + '\n')
			file.close()
	plt.ioff()
	plt.show()

def display_axes(self, data, type:str='wavefunction'):
	if type == 'eigenstates':
		lam, psi, err = data[0], data[1], data[2]
		if self.dim == 1:
			fig = plt.figure(figsize=(8, 4) if not hasattr(self, 'figsize') else self.figsize)
			fig.canvas.manager.set_window_title(f'TDSE simulation: {type} of {self.InitialState[1]}')
			ax = plt.gca()
			for _ in range(max(xp.atleast_1d(self.InitialState[0])) + 1):
				ax.plot(self.xgrid[0] / self.q0, psi[_], label=f'{lam[_]:.6f} (err={err[_]:.2e})')
			ax.set_xlabel('$x$')
			ax.set_xlim((-self.L[0] / self.q0, self.L[0] / self.q0))
			ax.legend(loc='upper right', labelcolor='linecolor')
		elif self.dim ==2:
			for _ in range(max(xp.atleast_1d(self.InitialState[0])) + 1):
				vmin = min(psi[_].min(), -1e-3)
				vmax = max(psi[_].max(), 1e-3)
				divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
				fig, ax = plt.subplots(1, 1)
				fig.canvas.manager.set_window_title(f'TDSE simulation: {_}th eigenstate of {self.InitialState[1]}')
				extent = (-self.L[0], self.L[0], -self.L[1], self.L[1])
				im = ax.imshow(psi[_].T, origin='lower', extent=extent, norm=divnorm)
				ax.set_title(f'$E$ = {lam[_]:.3f}  (err = {err[_]:.2e})')
				ax.set_xlabel('$x$')
				ax.set_ylabel('$y$')
				plt.colorbar(im)
		return fig, ax
	elif type in ['wavefunction', 'ionization']:
		fig, ax = plt.subplots(figsize=(8, 4) if not hasattr(self, 'figsize') else self.figsize)
		fig.canvas.manager.set_window_title(f'TDSE simulation: {type}')
		if self.dim == 1:
			ax.plot(self.xgrid[0] / self.q0, xp.abs(data)**2, cs[2], linewidth=1, label=r'$\vert\psi (x,0)\vert^2$')
			h, = ax.plot(self.xgrid[0] / self.q0, xp.abs(data)**2, cs[1], linewidth=2, label=r'$\vert\psi (x,t)\vert^2$')
			ax.set_yscale(self.scale)
			ax.legend(loc=self.legend if hasattr(self, 'legend') else 'best', labelcolor='linecolor')
			if hasattr(self, 'ylim') and (self.ylim != 'auto'):
				ax.set_ylim(self.ylim)
			if hasattr(self, 'xlim'):
				ax.set_xlim((self.xlim[0] / self.q0, self.xlim[1]/self.q0))
			else:
				ax.set_xlim((-self.L[0] / self.q0, self.L[0]/self.q0))
			ax.set_aspect('auto')
			plt.tight_layout(pad=2)
		elif self.dim == 2:
			norm = LogNorm(vmin=1e-4, vmax=xp.abs(data).max()**2, clip=True) if self.scale=='log' else None
			h = ax.imshow(xp.abs(data).T**2, extent=(-self.L[0] / self.q0, self.L[0] / self.q0, -self.L[1] / self.q0, self.L[1] / self.q0), cmap=cmap_psi, norm=norm, interpolation='nearest')
			if hasattr(self, 'xlim'):
				ax.set_xlim((self.xlim[0] / self.q0, self.xlim[1] / self.q0))
			if hasattr(self, 'ylim'):
				ax.set_ylim((self.ylim[0] / self.q0, self.ylim[1] / self.q0))
			fig.colorbar(h, ax=ax, shrink=0.5)
			ax.set_ylabel('$y/q$')
		ax.set_title('$t / T = 0 $', loc='right', pad=20)
		ax.set_xlabel('$x/q$')
	elif type == 'HHG':
		fig, ax = plt.subplots(figsize=(8, 4) if not hasattr(self, 'figsize') else self.figsize)
		fig.canvas.manager.set_window_title(f'TDSE simulation: HHG spectrum')
		h = ax.plot([], [], cs[2], linewidth=2, label=r'dipole')[0], ax.plot([], [], cs[3], linewidth=2, label=r'acceleration')[0]
		ax.axvline(x= (3.17 * self.Up + self.compute_Ip()) / self.omega, color='k', linewidth=2, label=r'$3.17 U_p + I_p$')
		ax.set_xlabel('$\omega /\omega_\mathrm{field}$')
		ax.set_yscale('log')
		ax.legend(loc='upper right', labelcolor='linecolor')
	elif type == 'Husimi':
		p0 = self.omega / self.E0
		hrepr = self.compute_husimi(data, self.p_husimi, self.sigma_husimi)
		fig, ax = plt.subplots(figsize=(8, 4) if not hasattr(self, 'figsize') else self.figsize)
		fig.canvas.manager.set_window_title(f'TDSE simulation: Husimi representation')
		h = ax.imshow(hrepr.T, extent=(-self.L[0] / self.q0, self.L[0] / self.q0, self.p_husimi.min() / p0, self.p_husimi.max() / p0), cmap=cmap_psi, interpolation='nearest')
		fig.colorbar(h, ax=ax, shrink=0.5)
		ax.set_xlabel('$x/q$')
		ax.set_ylabel('$\omega p / E_0$')
	return fig, ax, h

def save_data(self, data:xp.ndarray, filestr:str, info=[]) -> None:
	if self.SaveData:
		mdic = self.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(' %B %d, %Y'), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')

if __name__ == '__main__':
    main()
