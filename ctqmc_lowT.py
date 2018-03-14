#!/usr/bin/env python
import numpy as np
import scipy.linalg
import CalcStatistics
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad, nquad
from itertools import *
#import gmpy2
#gmpy2.get_context().precision=400
import time
import math
#start = time.time()
from os import sys



def Derv(fn, x):
    return (fn(x+1e-04*x)-fn(x))/(1e-04*x)


def A(alpha):
    return np.sqrt(mass/(2*np.pi*hbar*alpha))

    
def B(alpha):
    return 0.5*mass/(alpha*hbar)




def GetAll_Psi_ts(all_taus, args):
    (beta, order) = args
    all_ts = np.zeros(order)
    all_ts[0] = beta - all_taus[0] + all_taus[-1]
    for i in range(1,order):
        all_ts[i] = all_taus[i-1] - all_taus[i]
    return all_ts


def Make_M_matrix(all_Bs, gamma):
    dim = len(all_Bs)
    M = np.zeros([dim,dim],float)
    if dim == 1:
        M[0,0] = 2*gamma
    elif dim == 2:
        M[0,0] = M[1,1] = 2*(all_Bs[0] + all_Bs[1] + gamma)
        M[0,1] = M[1,0] = -2*(all_Bs[0]+all_Bs[1])
    else:
        for i in range(dim-1):
            M[i,i] = 2*(all_Bs[i] + all_Bs[i+1] + gamma)
            M[i,i+1] = M[i+1,i] = -2*all_Bs[i+1]
        M[-1,-1] = 2*(all_Bs[0] + all_Bs[-1] + gamma)
        M[0,-1] = M[-1,0] = -2*all_Bs[0]

    return M


def PhiConfigWeight(all_taus, all_ts, e, order):
    all_taus_phi = all_taus.copy()
    all_taus_phi[0] = beta
    args1 = (beta, order)
    all_ts_phi = GetAll_Psi_ts(all_taus_phi, args1)
    all_Bs_phi = B(all_ts_phi)
    M_phi = Make_M_matrix(all_Bs_phi, lam)
    e_phi = scipy.linalg.eigvalsh(M_phi)    
    Aratio = all_ts[0]*all_ts[1]/(all_ts_phi[0]*all_ts_phi[1])
    det_ratio = np.product(e/e_phi)    
    return math.sqrt(Aratio*det_ratio)*order/beta
    


def PsiConfigWeight(all_ts, all_Bs, M):
    Minv = np.linalg.inv(M)
    fac1 = (-0.5 + all_Bs[0]*(Minv[0,0]+Minv[-1,-1]-2*Minv[0,-1]))
    return fac1/all_ts[0]


def TotalEnergyEstimator(all_taus, order, args):
    (all_ts, all_Bs, M, e) = args
    if order == 0:
        return NegDervZ0/Z0       
    else:
        phi = -PhiConfigWeight(all_taus, all_ts, e, order)
        psi = -PsiConfigWeight(all_ts, all_Bs, M)
        #return -(phi + psi)
        return phi+psi



def InsertMoveDecision(timeOrder_old, timeOrder_new, ts_old, ts_new, e_old, e_new):
    accept_decision = False
    
    if timeOrder_old == 0:
        R_n_nplus1 = (0.5/lam)/l*math.sqrt(PI/lam)
        
    else:
        Aratio = math.sqrt(np.product(ts_old/ts_new[0:timeOrder_old]))*A(ts_new[-1])
        det_ratio = math.sqrt(2*PI*np.product(e_old/e_new[0:timeOrder_old])/e_new[-1])
        R_n_nplus1 = (0.5/lam)*Aratio*det_ratio*beta/timeOrder_new
    if R_n_nplus1 >= 1.:
        accept_decision = True
    elif R_n_nplus1 < 1. and R_n_nplus1 > np.random.rand():
        accept_decision = True
    else:
        accept_decision = False
    return accept_decision



def RemovalMoveDecision(timeOrder_old, timeOrder_new, ts_old, ts_new, e_old, e_new):

    accept_decision = False
    if timeOrder_new == 0:
        R_n_nplus1 = (0.5/lam)/l*math.sqrt(PI/lam)
    else:
        Aratio = math.sqrt(np.product(ts_new/ts_old[0:timeOrder_new]))*A(ts_old[-1])
        det_ratio = math.sqrt(2*PI*np.product(e_new/e_old[0:timeOrder_new])/e_old[-1])
        R_n_nplus1 = (0.5/lam)*Aratio*det_ratio*beta/timeOrder_old
        
    R_nplus1_n = 1/R_n_nplus1
    if R_nplus1_n > 1:
        accept_decision = True
    elif R_nplus1_n < 1 and R_nplus1_n > np.random.rand():
        accept_decision = True
    else:
        accept_decision = False
    return accept_decision




def StayMoveDecision(timeOrder_old, timeOrder_new, ts_old, ts_new, e_old, e_new):
    accept_decision = False
    if timeOrder_old == 0 or timeOrder_old == 1:
        R = 1.
    else:
        Aratio = math.sqrt(np.product(ts_old/ts_new))
        det_ratio = math.sqrt(np.product(e_old/e_new))
        R = Aratio*det_ratio
        
    if R >= 1.:
        accept_decision = True
    elif R < 1. and R > np.random.rand():
        accept_decision = True
    else:
        accept_decision = False
    return accept_decision





#execfile("input.py")


mass = 1.
hbar = 1.
k_b = 1.
lam = 0.05
T = 0.5
beta = 1/(k_b*T)
l = 10000.
PI = np.pi


print 'Temperature = %f \n' % T
print 'Lambda = %f \n' % lam


def Z_0(gamma):
    return A(gamma)*l

Z0 = Z_0(beta)
NegDervZ0 = 0.5/beta*Z0


numMCsteps = 1000000
numEquil = numMCsteps/5
numObsInterval = 5
timeOrder_start = 1 #Pick an initial time order not equal to 0
moves = ['insert','remove','stay','stay', 'remove','remove','remove','remove','stay', 'stay', 'stay', 'stay', 'stay', 'stay', 'stay', 'stay']


En_trace = []
Enavg_trace = []
#fd = file("job_percent_done","w")


order_trace = []
ins_accepted = 0.
rem_accepted = 0.
stay_accepted = 0.
ins_attempted = 0.
rem_attempted = 0.
stay_attempted = 0.

count = 0.
enavg = 0.
timeOrder_old = timeOrder_start


taus_old = beta*np.random.rand(timeOrder_old)
taus_old[::-1].sort()
args1 = (beta, timeOrder_old)    
all_ts_old = GetAll_Psi_ts(taus_old, args1)
all_Bs_old = B(all_ts_old)
M_old = Make_M_matrix(all_Bs_old, lam)
e_old = scipy.linalg.eigvalsh(M_old)

        
for step in range(numMCsteps):
    
    print step, timeOrder_old
    #print e_old    
    if timeOrder_old == 0:
	moves = ['insert','stay','stay', 'stay','stay','stay','stay','stay','stay','stay','stay']
	MCmove = moves[np.random.randint(0,2)]    
    #elif timeOrder_old == 900:
	#moves = ['insert','remove','stay','stay','stay','stay','stay','stay','stay','stay','stay','stay']
	#MCmove = moves[np.random.randint(1,3)]
    else:
        moves = ['insert','remove','stay','stay','stay','stay','stay','stay','stay','stay','stay','stay']
	MCmove = moves[np.random.randint(0,3)]
	    
    if MCmove == 'insert':
	ins_attempted += 1
	timeOrder_new = timeOrder_old + 1
	tau_add = beta*np.random.rand()
	taus_new = np.append(taus_old, tau_add)
	taus_new[::-1].sort()
	args2 = (beta, timeOrder_new)
	all_ts_new = GetAll_Psi_ts(taus_new, args2)	    
	all_Bs_new = B(all_ts_new)
    	M_new = Make_M_matrix(all_Bs_new, lam)
	e_new = scipy.linalg.eigvalsh(M_new)
	   
	if InsertMoveDecision(timeOrder_old, timeOrder_new, all_ts_old, all_ts_new, e_old, e_new):
	    ins_accepted += 1
	    timeOrder_upd = timeOrder_new
	    taus_upd = taus_new
	    all_ts_upd = all_ts_new
	    all_Bs_upd = all_Bs_new
	    M_upd = M_new
	    e_upd = e_new
	else:
	    timeOrder_upd = timeOrder_old
	    taus_upd = taus_old
	    all_ts_upd = all_ts_old
	    all_Bs_upd = all_Bs_old
	    M_upd = M_old
	    e_upd = e_old
	        
    if MCmove == 'remove':
	rem_attempted += 1
	timeOrder_new = timeOrder_old - 1
	#print timeOrder_new
	if timeOrder_new == 0:
	    taus_new = np.array([])  
	    all_ts_new = np.array([])
	    all_Bs_new =np.array([])
	    M_new = np.array([[]])
	    e_new = np.array([])
       	else:
	    tau_remove_index = np.random.randint(0,timeOrder_old)
	    taus_new = np.delete(taus_old, tau_remove_index)
	    args2 = (beta, timeOrder_new)
	    all_ts_new = GetAll_Psi_ts(taus_new, args2)	    
	    all_Bs_new =B(all_ts_new)
    	    M_new = Make_M_matrix(all_Bs_new, lam)
	    e_new = scipy.linalg.eigvalsh(M_new)
	   
	   
    	if RemovalMoveDecision(timeOrder_old, timeOrder_new, all_ts_old, all_ts_new, e_old, e_new):
    	    rem_accepted += 1
	    #print 'insertion move accepted'
	    timeOrder_upd = timeOrder_new
       	    taus_upd = taus_new
       	    all_ts_upd = all_ts_new
	    all_Bs_upd = all_Bs_new
	    M_upd = M_new
	    e_upd = e_new
	else:
	    timeOrder_upd = timeOrder_old
	    taus_upd = taus_old
	    all_ts_upd = all_ts_old
	    all_Bs_upd = all_Bs_old
	    M_upd = M_old
	    e_upd = e_old
	
	
    if MCmove == 'stay':
	stay_attempted += 1
	timeOrder_new = timeOrder_old	    
	#print timeOrder_new
	if timeOrder_new == 0:
	    taus_new = np.array([])  
	    all_ts_new = np.array([])
	    all_Bs_new = np.array([])
	    M_new = np.array([[]])
	    e_new = np.array([])
        else:
	    taus_new = beta*np.random.rand(timeOrder_new)
	    taus_new[::-1].sort()
	    args2 = (beta, timeOrder_new)
	    all_ts_new = GetAll_Psi_ts(taus_new, args2)
	    all_Bs_new = B(all_ts_new)
    	    M_new = Make_M_matrix(all_Bs_new, lam)
    	    e_new = scipy.linalg.eigvalsh(M_new)    	    
	    
	if StayMoveDecision(timeOrder_old, timeOrder_new, all_ts_old, all_ts_new, e_old, e_new):
	    stay_accepted += 1
	    timeOrder_upd = timeOrder_new
	    taus_upd = taus_new
	    all_ts_upd = all_ts_new
       	    all_Bs_upd = all_Bs_new
	    M_upd = M_new
	    e_upd = e_new
	else:
	    timeOrder_upd = timeOrder_old
	    taus_upd = taus_old
	    all_ts_upd = all_ts_old
	    all_Bs_upd = all_Bs_old
	    M_upd = M_old
	    e_upd = e_old
	
	
	        
    if step > numEquil and step % numObsInterval == 0:
        count += 1
        order_trace.append(timeOrder_upd)
	args3 = (all_ts_upd, all_Bs_upd, M_upd, e_upd)
	en_tot = TotalEnergyEstimator(taus_upd, timeOrder_upd, args3)
        #enavg += en_tot
	#Enavg_trace.append(enavg/count)
	En_trace.append(en_tot)
        #Enavg_trace.append(np.mean(En_trace))	    

    #if step % numMCsteps/10 == 0:
    #    open("job_percent_done","w")
    #    print >>fd, count*10
    #    fd.close()
    #    count += 1
         
        
    
    timeOrder_old = timeOrder_upd
    taus_old = taus_upd
    all_ts_old = all_ts_upd
    all_Bs_old = all_Bs_upd
    M_old = M_upd
    e_old = e_upd
    



#count_list, order_list = np.histogram(order_trace,bins=max(order_trace)+1)
#plt.hist(order_trace, bins=max(order_trace)+1)
#plt.show()
E_stats = CalcStatistics.Stats(np.array(En_trace))
E_MC = E_stats[0] + 0.5/lam
E_err = E_stats[1] #- num_stats[0]*den_stats[1]/(den_stats[0]**2)
E_low = E_MC - E_err
E_high = E_MC + E_err
order_max = max(order_trace)



print 'E_avg = %f \n' % E_MC 
print 'std error = %f \n' % E_err
         
#print 'It took', time.time()-start, 'seconds.'	        
	    
fd = file("CTQMC_qho_gaussian_pot_Eavg_traj","w")	    
print >>fd, E_MC, E_err, order_max	    
fd.close()
   
sys.exit()	    
	    
    
    
