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
from TDSE_dict import dict_, darkmode

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
    self = TDSE(dict_)
    print(f'\033[92m  {self} \033[00m')
    run_method(self)

def run_method(self):
	filestr = type(self).__name__ + '_' + time.strftime('%Y%m%d_%H%M')
	if self.Method == 'plot_potentials':
		fig, ax = display_axes(self, [], type='Potentials')
	elif self.Method == 'plot_eigenstates':
		start = time.time()
		lam, psi, err = self.eigenstates(self.Vgrid_, self.InitialState[0] + 1, output='all')
		fig, ax = display_axes(self, [lam, psi, err], type='eigenstates')
		print(f'\033[90m        Computation of the following eigenstates finished in {int(time.time() - start)} seconds \033[00m')
		for _ in range(self.InitialState[0] + 1):
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
	elif self.Method in ['wavefunction', 'ionization']:
		plt.ion()
		start = time.time()
		lam0, psi0, err = self.initcond()
		print(f'\033[90m        Computation of the initial state finished in {int(time.time() - start)} seconds: E0 = {lam0:.6f} (with err = {err:.2e}) \033[00m')
		if self.dim >= 2:
			print(f'\033[90m                   with quantum number(s):  L = {self.quantum_numbers(psi0)} \033[00m')
		init_density = self.norm(psi0)**2
		if self.PlotData:
			fig, ax, h = display_axes(self, self.change_frame(0, psi0), type=self.Method)

		start = time.time()

		def plot_command(t:float, psi:xp.ndarray) -> None:
			n = int(t / self.step)
			if (n+1)%self.refresh == 0:
				self.plot(ax, h, t, psi)
		
		tspan = xp.linspace(0, self.final_time, int(self.ncycles * self.nsteps_per_period // self.refresh))
		sol = solve_ivp_symp(self.chi, self.chi_star, (0, self.final_time), psi0, step=self.step, t_eval=tspan, method=self.ode_solver, command=lambda t, psi:plot_command(t, psi))
		print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
		save_data(self, xp.array([sol.t, sol.y], dtype=xp.object_), filestr)

		if self.SaveWaveFunction:
			def animate(_):
				self.plot(ax, h, sol.t[_], sol.y[..., _])
				return h
			FuncAnimation(fig, animate, frames=len(sol.t), interval=200).save(filestr + '.gif', writer=PillowWriter(fps=5), dpi=self.dpi)
			print(f'\033[90m        Animation saved in {filestr}.gif \033[00m')
		if self.Method == 'ionization':
			proba = 1 - self.norm(sol.y[..., -1])**2 / init_density
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
	if type == 'wavefunction':
		fig, ax = plt.subplots(figsize=(8, 4))
		fig.canvas.manager.set_window_title(f'TDSE simulation: {type}')
		if self.dim == 1:
			ax.plot(self.xgrid[0] / self.q0, xp.abs(data)**2, cs[2], linewidth=1, label=r'$\vert\psi (x,0)\vert^2$')
			h, = ax.plot(self.xgrid[0] / self.q0, xp.abs(data)**2, cs[1], linewidth=2, label=r'$\vert\psi (x,t)\vert^2$')
			ax.set_yscale(self.scale)
			ax.legend(loc='upper right', labelcolor='linecolor')
			#ax.set_ylim((-0.003, 0.07))
			ax.set_xlim((-self.L[0] / self.q0, self.L[0]/self.q0))
			ax.set_aspect('auto')
			plt.tight_layout(pad=2)
		elif self.dim == 2:
			norm = LogNorm(vmin=1e-4, vmax=(xp.abs(data)**2).max(), clip=True) if self.scale=='log' else None
			h = ax.imshow(xp.abs(data).transpose()**2, extent=(-self.L[0] / self.q0, self.L[0] / self.q0, -self.L[1] / self.q0, self.L[1] / self.q0), cmap=cmap_psi, norm=norm, interpolation='nearest')
			fig.colorbar(h, ax=ax, shrink=0.5)
			ax.set_ylabel('$y/q$')
		ax.set_title('$t / T = 0 $', loc='right', pad=20)
		ax.set_xlabel('$x/q$')
		return fig, ax, h
	elif type == 'eigenstates':
		lam, psi, err = data[0], data[1], data[2]
		if self.dim == 1:
			fig = plt.figure(figsize=(8, 8))
			fig.canvas.manager.set_window_title(f'TDSE simulation: {type} of {self.InitialState[1]}')
			ax = plt.gca()
			for _ in range(self.InitialState[0] + 1):
				ax.plot(self.xgrid[0] / self.q0, psi[_], label=f'{lam[_]:.6f} (err={err[_]:.2e})')
			ax.set_xlabel('$x$')
			ax.set_xlim((-self.L[0] / self.q0, self.L[0] / self.q0))
			ax.legend(loc='upper right', labelcolor='linecolor')
		elif self.dim ==2:
			for _ in range(self.InitialState[0] + 1):
				vmin = min(psi[_].min(), -1e-3)
				vmax = max(psi[_].max(), 1e-3)
				divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
				fig, ax = plt.subplots(1, 1)
				fig.canvas.manager.set_window_title(f'TDSE simulation: {_}th eigenstate of {self.InitialState[1]}')
				extent = (-self.L[0], self.L[0], -self.L[1], self.L[1])
				im = ax.imshow(psi[_].transpose(), origin='lower', extent=extent, norm=divnorm)
				ax.set_title(f'$E$ = {lam[_]:.3f}  (err = {err[_]:.2e})')
				ax.set_xlabel('$x$')
				ax.set_ylabel('$y$')
				plt.colorbar(im)
	elif type == 'Potentials':
		if self.dim == 1:
			fig = plt.figure(figsize=(8, 8))
			fig.canvas.manager.set_window_title('TDSE simulation: ' + type)
			ax = plt.gca()
			ax.plot(self.xgrid[0] / self.q0, self.Vgrid, cs[2], label=r'$V(x)$')
			ax.plot(self.xgrid[0] / self.q0, self.kramers_henneberger(2), cs[3], label=r'$V_\mathrm{KH, 2}(x)$')
			ax.plot(self.xgrid[0] / self.q0, self.kramers_henneberger(3), cs[4], label=r'$V_\mathrm{KH, 3}(x)$')
			ax.set_xlabel('$x/q$')
			ax.set_xlim((-self.L[0] / self.q0, self.L[0] / self.q0))
			ax.legend(loc='upper right', labelcolor='linecolor')
		elif self.dim == 2:
			plt.rcParams.update({'figure.figsize': [21, 7]})
			fig, ax = plt.subplots(1, 3)
			extent = (-self.L[0] / self.q0, self.L[0] / self.q0, -self.L[1] / self.q0, self.L[1] / self.q0)
			v = [self.Vgrid.min(), self.Vgrid.max()]
			ax[0].imshow(self.Vgrid.transpose(), origin='lower', extent=extent, cmap='gist_yarg', vmin=v[0], vmax=v[1])
			ax[1].imshow(self.kramers_henneberger(2).transpose(), origin='lower', extent=extent, cmap='gist_yarg', vmin=v[0], vmax=v[1])
			ax[2].imshow(self.kramers_henneberger(3).transpose(), origin='lower', extent=extent, cmap='gist_yarg', vmin=v[0], vmax=v[1])
			ax[1].set_title('$V_\mathrm{KH, 2}$')
			ax[2].set_title('$V_\mathrm{KH, 3}$')
			for _ in ax:
				_.set_xlabel('$x/q$')
				_.set_ylabel('$y/q$')
		else:
			fig, ax = [], []
			warnings.warn('Dimension not compatible for display')
	return fig, ax

def save_data(self, data:xp.ndarray, filestr:str, info=[]) -> None:
	if self.SaveData:
		mdic = self.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(' %B %d, %Y'), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')

if __name__ == '__main__':
    main()
