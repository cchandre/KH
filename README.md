# Bogolyubov’s averaging applied to the Kramers-Henneberger Hamiltonian
* [**KHBogolyubov.mlx**](https://github.com/cchandre/KH/blob/main/KHBogolyubov.mlx): Matlab livescript for the manuscript *Bogolyubov’s averaging theorem applied to the Kramers-Henneberger Hamiltonian* by E. Floriani, J. Dubois, C. Chandre


<ins>Reference:</ins> E. Floriani, J. Dubois, C. Chandre, *Bogolyubov's averaging theorem applied to the Kramers-Henneberger Hamiltonian*, [Physica D](https://doi.org/10.1016/j.physd.2021.133124) 431, 133124 (2022); [arxiv:2107.01946](https://arxiv.org/abs/2107.01946)

```bibtex
@article{floriani2021,
         title = {Bogolyubov's averaging theorem applied to the Kramers-Henneberger Hamiltonian}, 
         author = {Floriani, E. and Dubois, J. and Chandre, C.},
         journal = {Physica D},
         volume = {431},
         pages = {133124},
         year = {2022},
         doi = {10.1016/j.physd.2021.133124},
         URL = {https://doi.org/10.1016/j.physd.2021.133124}
}
```

___
# Time-dependent Schrödinger equation in the dipole approximation (TDSE)

Numerical integration of the following Schrödinger equation (in the dipole approximation and in atomic units)
```math
i \frac{\partial \psi}{\partial t} = \left( -\frac{\Delta}{2} + V(x) + x E(t) \right) \psi(x,t),
```
where $E(t)=E_0 f(t) \Phi(\omega t)$ with $f(t)$ the laser envelope, and $\Phi$ a $2\pi$-periodic function. The frequency $\omega$ is defined by the laser wavelength, and the amplitude of the electric field $E_0$ is defined by the laser intensity. Here $V$ is the ionic potential. 

- [`TDSE_params.py`](https://github.com/cchandre/KH/blob/main/TDSE_params.py): to be edited to change the parameters of the TDSE computation (see below for a list of parameters)

- [`TDSE_classes.py`](https://github.com/cchandre/KH/blob/main/TDSE_classes.py): contains the TDSE class and main functions

- [`TDSE.py`](https://github.com/cchandre/KH/blob/main/TDSE.py): contains the methods to execute TDSE

Once [`TDSE_params.py`](https://github.com/cchandre/KH/blob/main/TDSE_params.py) has been edited with the relevant parameters, run the file as 
```sh
python3 TDSE.py
```
or 
```sh
nohup python3 -u TDSE.py &>TDSE.out < /dev/null &
```
The list of Python packages and their version are specified in [`requirements.txt`](https://github.com/cchandre/KH/blob/main/requirements.txt)

##  Parameters

The file [`TDSE_params.py`](https://github.com/cchandre/KH/blob/main/TDSE_params.py) should contain the following parameters:

- *Method*: string; 'eigenstates', 'wavefunction', 'HHG', 'ionization'; choice of method
  - 'eigenstates': plot the first *k* eigenstates and eigenvalues of the potential specified in *InitialState[1]*, where *k* is equal to *InitialState[0]*+1
  - 'wavefunction': displays the wavefunction as a function of time obtained by integrating the TDSE equation
  - 'HHG': compute the high-harmonic generation (HHG) spectrum as a function of time 
  - 'ionization': computes the ionization probability as well as displaying the wavefunction as a function of time
  - 'Husimi': computes the Husimi representation of the wavefunction as a function of time; 'p_husimi' and 'sigma_husimi' need to be defined
- *laser_intensity*: float; intensity of the laser field in W cm<sup>-2</sup>
- *laser_wavelength*: float; wavelength of the laser field in nm
- *laser_E*: lambda function returning an array of *n* floats; *n* components (where *n* is the dimension of configuration space) of the electric field (dipole approximation); the electric field is then given by *E0* * *laser_envelope*(t) * *laser_E*(&omega; t) where *E0* = sqrt(*laser_intensity*)
- *te*: array of 3 floats; duration of the ramp-up, plateau and ramp-down in laser cycles
- *V*: lambda function; ionic potential
- *InitialState*: integer or array [integer or tuple of integers or lambda function, string]; integer = index of the initial eigenstate (0 corresponds to the ground state, 1 is the first excited state...); string = potential with which the initial state is computed ('V', 'VKH2' or 'VKH3'); in case a tuple of integers is entered, the initial state is a linear combination of the various states in the tuple with the coefficients given in *InitialCoeffs*; if *InitialState* or *InitialState*[0] is a lambda function, the initial state is computed on the grid using this function
- *L*: array of *n* floats; size of the box in each direction
- *N*: array of *n* integers; number of points in each direction

Some additional (optional) parameters could be defined in [`TDSE_params.py`](https://github.com/cchandre/KH/blob/main/TDSE_params.py):

- *laser_envelope*: string; 'trapez', 'sinus', 'const'; envelope of the laser field during ramp-up and ramp-down
- *InitialCoeffs*: array of floats; the initial state is a linear combination of eigenstates $\Psi_k(x)$ of the potential defined in *InitialState*[1], i.e., $\psi(x,0)=\sum_k c_k \Psi_k(x)$ where $k$ belongs to *InitialState*[0]; if not specified, the coefficients are equal to 1
- *DisplayCoord*: string; 'lab', 'KH2' or 'KH3'; if KH (Kramers-Henneberger), the wave function is moved to the KH frame (for display and for saving) of order 2 or 3; if not specified, 'lab' is the default
- *delta*: float or array of *n* floats; size of the absorbing boundary in each direction (if float, the size is taken equal in all dimensions)
- *Lg*: float or array of *n* floats; size of the box for the initial computation of the initial state along each dimension; if float, [-*Lg*, *Lg*] in each dimension; if not specified, *Lg*=*L*
- *nsteps_per_period*: integer; number of steps per laser period for the integration; the time-step is then defined as 2&pi; /&omega; / *nsteps_per_period*
- *scale*: string; 'linear' or 'log'; the axis scale type to apply for the representation of the wavefunction (if *Method*='wavefunction')
- *legend*: string; location of the legend; for more details, see [matplotlib legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
- *xlim*: tuple of floats; x-axis view limits (in atomic units)
- *ylim*: tuple of floats or string; y-axis view limits (in atomic units); if 'auto', let the y-axis scale automatically
- *figsize*: tuple of floats; width and height in inches of the figure
- *SaveWaveFunction*: boolean; if True, saves the animation of the wavefunction  as an animated `.gif` image
- *PlotData*: boolean; if True, displays the wavefunction on the screen as time increases (only for 1D and 2D)
- *SaveData*: boolean; if True, the time evolution of the wave function are saved in a `.mat` file
- *dpi*: integer; number of dots per inch for the movie frames (if *SaveWaveFunction* is True)
- *refresh*: integer; the wavefunction is displayed every *refresh* time steps
- *darkmode*: boolean; if True, plots are done in dark mode
- *tol*: relative accuracy for eigenvalues (stopping criterion) (default=10<sup>-10</sup>, 0 implies machine precision); see [eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- *maxiter*: maximum number of Arnoldi update iterations allowed (default=1000); see [eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- *ncv*: number of Lanczos vectors generated (default=100); see [eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- *Nkh*: integer; number of points in one period to compute the Kramers-Henneberger potentiel *V*<sub>KH</sub>(x) (default=2<sup>12</sup>)
- *ode_solver*: string; choice of splitting symplectic integrator; for a list see [pyHamSys](https://pypi.org/project/pyhamsys/) (default='BM4')

<ins>Reference:</ins> E. Floriani, J. Dubois, C. Chandre, *Scars of Kramers-Henneberger atoms*, [arxiv:2407.18575](https://arxiv.org/abs/2407.18575)

```bibtex
@misc{floriani2024,
      title={Scars of Kramers-Henneberger atoms}, 
      author={Elena Floriani and Jonathan Dubois and Cristel Chandre},
      year={2024},
      eprint={2407.18575},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2407.18575}, 
}
```

---

For more information: <cristel.chandre@cnrs.fr>
