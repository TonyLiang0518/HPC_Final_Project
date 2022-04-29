import numpy as np
import matplotlib.pyplot as plt

N = 2**np.arange(1,17)
recursion_seq  = "2.1234500000000003962e-06 3.7447999999999992808e-06 7.0703999999999975608e-06 1.752560000000000039e-05 3.7780750000000005843e-05 8.5291200000000007655e-05 0.00018637400000000002858 0.00040999440000000003312 0.00089964064999999994975 0.0019619848500000000233 0.0042363620000000004606 0.0090604712499999986303 0.019315103199999999334 0.040998073099999997326 0.087022246950000020638 0.18525902965000001776"
recursion_par  = "8.8515000000000011416e-07 2.6420500000000003945e-06 6.3562000000000002314e-06 1.5370099999999997057e-05 3.6206649999999996638e-05 8.2205149999999986609e-05 0.00018298249999999999845 0.00040688580000000006088 0.00089195360000000000443 0.0019489381000000003535 0.006840686599999999673 0.011527895450000000613 0.021528137149999997807 0.042578516599999997883 0.087906690500000023158 0.18233290000000001996"
recursion_task = "0.0015307564499999998069 0.0012086439500000002376 0.0012271406500000002666 0.0012076555999999999576 0.0012114931999999999698 0.0012173139000000001222 0.0012671960500000001813 0.0012668447000000000648 0.00147219490000000011 0.0022780184499999996464 0.0026965899000000004108 0.0036507145000000003864 0.0061054353000000007382 0.010026139499999999621 0.019711291550000002715 0.038045621299999998755"
iteration_seq  = "7.877000000000000985e-07 2.688500000000000602e-06 3.3392000000000001487e-06 4.5442000000000013171e-06 6.3239500000000006672e-06 9.6768500000000003629e-06 1.726875000000000301e-05 3.3085199999999999614e-05 8.5617100000000024304e-05 0.00020990145000000003995 0.00057416075000000003047 0.0014308277500000000058 0.0036989040000000002542 0.0092559119000000002198 0.022950810600000003114 0.055466071099999993133"
iteration_par  = "1.030450000000000019e-06 2.0266000000000006374e-06 2.7721999999999999023e-06 3.8873999999999994063e-06 5.6311500000000011339e-06 9.0491500000000012736e-06 1.6237099999999997396e-05 3.1808549999999995498e-05 8.542515000000002693e-05 0.0002080718499999999855 0.00057165665000000003525 0.0014270144500000000402 0.0037759016999999997152 0.0093918477999999985806 0.022896144100000005023 0.055250949350000001792"
fftw = "3.8860299999999997904e-05 1.0046950000000000428e-05 1.0055450000000002125e-05 1.1743650000000000632e-05 1.3702450000000002952e-05 2.8175100000000007441e-05 3.2173599999999999736e-05 3.8633800000000009475e-05 6.23683000000000039e-05 9.6647700000000006412e-05 0.00019661589999999998213 0.00038906800000000006353 0.0012050558500000000881 0.0030252105500000003695 0.0062614925000000019178 0.014789753999999998707"



recursion_soln  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
iteration_soln  = "0.233 0.121 0.0032 0.002 0.0005 0.0003"
fftw_soln = "0.233 0.121 0.0032 0.002 0.0005 0.0003"


recursion_seq = np.array([float(x) for x in recursion_seq.split(' ')])
recursion_par = np.array([float(x) for x in recursion_par.split(' ')])
recursion_task = np.array([float(x) for x in recursion_task.split(' ')])
iteration_seq = np.array([float(x) for x in iteration_seq.split(' ')])
iteration_par = np.array([float(x) for x in iteration_par.split(' ')])
fftw = np.array([float(x) for x in fftw.split(' ')])


recursion_soln = np.array([float(x) for x in recursion_soln.split(' ')])
iteration_soln = np.array([float(x) for x in iteration_soln.split(' ')])
fftw_soln = np.array([float(x) for x in fftw_soln.split(' ')])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, recursion_seq, '*-', label = 'SEQ')
ax.loglog(N, recursion_par, '*-', label = 'PAR')
ax.loglog(N, recursion_task, '*-', label = 'TASK')
ax.set_title('Recursion: Forward+Inverse tramsform')
ax.set_xlabel('N')
ax.set_ylabel('Runtime')
ax.legend()
plt.savefig('recursion_fwd+inv.jpg')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, iteration_seq, '*-', label = 'SEQ')
ax.loglog(N, iteration_par, '*-', label = 'PAR')
ax.set_title('Iteration: Forward+Inverse tramsform')
ax.set_xlabel('N')
ax.set_ylabel('Runtime')
ax.legend()
plt.savefig('iteration_fwd+inv.jpg')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(N, recursion_par, '*-', label = 'Recursion')
ax.loglog(N, iteration_par, '*-', label = 'Iteration')
ax.loglog(N, fftw, '*-', label = 'FFTW')
ax.set_title('Comparison: Forward+Inverse tramsform')
ax.set_xlabel('N')
ax.set_ylabel('Runtime')
ax.legend()
plt.savefig('3in1_fwd+inv.jpg')

'''
xs = np.linspace(0, 2*np.pi, 6)
ext_soln = np.sin(np.cos(xs))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xs, recursion_soln, '*-', label = 'Recursion')
ax.plot(xs, iteration_soln, '*-', label = 'Iteration')
ax.plot(xs, fftw_soln, '*-', label = 'FFTW')
ax.plot(xs, ext_soln, '*-', label = 'Exact')
ax.set_title('Comparison: PDE solution')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()
plt.savefig('PDE_soln.jpg')
'''


plt.show()
