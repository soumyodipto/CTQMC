#solving the schrodinger equation in 1d

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


mass = 1.
hbar = 1.
def A(alpha):
    return np.sqrt(mass/(2*np.pi*hbar*alpha))

    
def B(alpha):
    return 0.5*mass/(alpha*hbar)


#1-d Gaussian well
def potential_grid(x):    
    #print 'Lambda = %f \n' % lam
    return -np.exp(-lam*x**2)



L = 100.
lam = 1.0
n_grid = 20000

temp_list = [0.5] #temperatures to loop over

for T in temp_list:
	print 'Temperature = %f \n' % T
	
	beta = 1./T
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
	    

	e, v = np.linalg.eigh(H_mat)
	
	Z = sum(np.exp(-beta*e))
	#print 'Z = %f \n' % Z
	E_avg = sum(e*np.exp(-beta*e))/Z
	#print 'En_avg = %f \n' % E_avg
	
	#print >>fd, T, E_avg

