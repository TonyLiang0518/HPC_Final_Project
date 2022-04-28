import numpy as np
import matplotlib.pyplot as plt

N = 2**np.arange(1,7)
recursion_seq  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
recursion_par  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
recursion_task = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
iteration_seq  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
iteration_par  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
fftw = "0.233 0.121 0.0032 0.002 0.0005 0.0003"

recursion_PDE  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
iteration_PDE  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
fftw_PDE = "0.233 0.121 0.0032 0.002 0.0005 0.0003"

recursion_soln  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
iteration_soln  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
fftw_soln = "0.233 0.121 0.0032 0.002 0.0005 0.0003"


recursion_seq = np.array([float(x) for x in recursion_seq.split(' ')])
recursion_par = np.array([float(x) for x in recursion_par.split(' ')])
recursion_task = np.array([float(x) for x in recursion_task.split(' ')])
iteration_seq = np.array([float(x) for x in iteration_seq.split(' ')])
iteration_par = np.array([float(x) for x in iteration_par.split(' ')])
fftw = np.array([float(x) for x in fftw.split(' ')])

recursion_PDE = np.array([float(x) for x in recursion_PDE.split(' ')])
iteration_PDE = np.array([float(x) for x in iteration_PDE.split(' ')])
fftw_PDE = np.array([float(x) for x in fftw_PDE.split(' ')])

recursion_soln = np.array([float(x) for x in recursion_soln.split(' ')])
iteration_soln = np.array([float(x) for x in iteration_soln.split(' ')])
fftw_soln = np.array([float(x) for x in fftw_soln.split(' ')])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, recursion_seq, '*-', label = 'SEQ')
ax.loglog(N, recursion_par, '*-', label = 'PAR')
ax.loglog(N, recursion_task, '*-', label = 'TASK')
ax.set_title('Recursion: Forward+Inverse tramsform')
ax.legend()
#plt.savefig('recursion_fwd+inv.pdf')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, iteration_seq, '*-', label = 'SEQ')
ax.loglog(N, iteration_par, '*-', label = 'PAR')
ax.set_title('Iteration: Forward+Inverse tramsform')
ax.legend()
#plt.savefig('iteration_fwd+inv.pdf')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, recursion_par, '*-', label = 'Recursion')
ax.loglog(N, iteration_par, '*-', label = 'Iteration')
ax.loglog(N, fftw, '*-', label = 'FFTW')
ax.set_title('Comparison: Forward+Inverse tramsform')
ax.legend()
#plt.savefig('3in1_fwd+inv.pdf')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, recursion_PDE, '*-', label = 'Recursion')
ax.loglog(N, iteration_PDE, '*-', label = 'Iteration')
ax.loglog(N, fftw_PDE, '*-', label = 'FFTW')
ax.set_title('Comparison: PDE solve')
ax.legend()
#plt.savefig('3in1_PDE.pdf')

xs = np.linspace(0, 2*np.pi, 6)
ext_soln = np.sin(np.cos(xs))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, recursion_soln, '*-', label = 'Recursion')
ax.plot(N, iteration_soln, '*-', label = 'Iteration')
ax.plot(N, fftw_soln, '*-', label = 'FFTW')
ax.plot(N, ext_soln, '*-', label = 'Exact')
ax.set_title('Comparison: PDE solution')
ax.legend()
#plt.savefig('PDE_soln.pdf')



plt.show()
