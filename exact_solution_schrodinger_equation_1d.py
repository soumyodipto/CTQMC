#solving the schrodinger equation for a gaussian potential in 1d

#!/usr/bin/env python

import numpy as np
#import itertools as itertools
import matplotlib.pyplot as plt
#import collections
#from scipy.integrate import quad, dblquad, nquad
#import gmpy2
#gmpy2.get_context().precision=200
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

#T = 25   # in Kelvin
#beta = 1./(1.38065e-23*2.2937e+17*T)
#mass = 3672*20    #ph2 mass in au
#mass = 145536.41    #Ar2 mass in au
mass = 1.
hbar = 1.
def A(alpha):
    return np.sqrt(mass/(2*np.pi*hbar*alpha))

    
def B(alpha):
    return 0.5*mass/(alpha*hbar)


def potential_grid(x):
    #num_gaus = 4
    #C = [5.5049, -0.00209]   #in a.u.
    #alpha = [0.21926, 0.029683]    #in a.u.
    #return C[0]*np.exp(-alpha[0]*x**2)+C[1]*np.exp(-alpha[1]*x**2)
    
    print 'Lambda = %f \n' % lam
    return -np.exp(-lam*x**2)
    #return -np.exp(np.log(-0.5*x*x - C))
    #epsilon = 0.3833e-03
    #sigma = 6.4251
    #return epsilon*((sigma/x)**12 - (sigma/x)**6)
    

#plt.plot(np.arange(0., 10., 0.001), potential_grid(np.arange(0., 10., 0.001))/(3.16683e-06), 'k')
#plt.show()
#print min(potential_grid(np.arange(0., 10., 0.001))/(3.16683e-06))


L = 10000.
lam = 1.0
n_grid = 20000
#fd=file("Eavg_vs_T_L=100_ngrid="+str(n_grid)+"_lam="+str(lam),"w")
for T in [0.5]:#np.arange(0.01, 1.1, 0.01):
        print 'Temperature = %f \n' % T
	#beta = 1./(1.38065e-23*2.2937e+17*T)
	beta = 1./T
	#L = 150
        xmin = -0.5*L
        xmax = 0.5*L
	   #choose an even number
	dx = (xmax - xmin)/n_grid
	x_grid = np.arange(xmin, xmax+dx, dx)	
	x_grid = x_grid[0:n_grid+1]
	V_grid = potential_grid(x_grid)
	V_mat = np.diag(V_grid[1:-1])
	H_mat = 1./(dx**2*mass)*np.eye(n_grid-1) + V_mat
        for i in xrange(n_grid-2):
	    H_mat[i, i+1] = -0.5/(dx**2*mass)
	    H_mat[i+1, i] = -0.5/(dx**2*mass)
	    

	#e,psi=scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(H_mat),k=20,which='SA',maxiter=100000,tol=1e-5)
	e, v = np.linalg.eigh(H_mat)
	#e-=C
	'''
	#Counting the number of boun states
	c = 0.
	for en in e:
	    if en < 0:
	        c += 1
	print c
	'''
	#print sum(e)
	Z = sum(np.exp(-beta*e))
	#print 'Z = %f \n' % Z
	E_avg = sum(e*np.exp(-beta*e))/Z
	print 'En_avg = %f \n' % E_avg
	
	#print >>fd, T, E_avg

