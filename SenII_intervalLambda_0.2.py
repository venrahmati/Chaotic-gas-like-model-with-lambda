##################################
#sen II _interval lambda
#random trade
#chaotic agent
##################################
import numpy as np
import math
import numba
import matplotlib.pyplot as plt
#import random
import time
from numba import *


 
start_time = time.time()
@numba.njit(parallel=True, fastmath = True)
def saving(N,z,lowband):
    N2 = N*N
    m = np.ones(N)
    lan = 0.2*np.random.random(size=N)+lowband
    epsi = np.random.random(N2)
    for i in range(N2):
        dm = epsi[i]*(1-lan[z[i,1]])*m[z[i,1]]  - (1-epsi[i])*(1-lan[z[i,0]])*m[z[i,0]] 
        m[z[i,1]] = m[z[i,1]] - dm 
        m[z[i,0]] = m[z[i,0]] + dm 
    return m


#setting values and doing simulation
N = 5000

N2 = N*N
z= np.ones(2*N2).reshape(N2,2)#np.array([0.4913 for i in prange(N2)])

z[0,0] = np.random.random()
z[0,1] = np.random.random()
    
za = 1.032
zb = 1.08429

for i in range(N2-1):
    z[i+1,0] = za * (3*z[i,1]+1) * z[i,0] * (1-z[i,0])
    z[i+1,1] = zb * (3*z[i,0]+1) * z[i,1] * (1-z[i,1])
    
z= N*z
z = z.astype(np.int64)

lam = [0.2, 0.75, 0.8]
q= [np.array([]) for i in range(len(lam))]

for j in range(len(lam)):
      for i in range(1000):
          z= np.random.permutation(z)
          m = saving(N,z,lam[j])
          q[j] = np.hstack((q[j], m))

 print("--- %s seconds ---" % (time.time() - start_time))
#####################################################################
mark = ['^','*','o','x','+']
col = ['k','k','k','k','k']
plt.rcParams['figure.dpi'] = 300
for j in np.arange(len(lam)):
    m1 = [value for value in list(q[j]) if value != 1.0]
    n,bins = np.histogram(m1,bins= np.logspace(np.log10(0.01), np.log10(100.0), 50),density=1)
    pos= 0.5*(bins[1:]+bins[:-1])
    plt.scatter(pos, n , label = str(lam[j])+ ' < ' + '$\lambda $'' < ' +str(lam[j]+0.2),marker=mark[j], color=col[j])


x = np.arange(3,50)
y = 0.5/x**2
plt.plot(x,y, label = "$x^{-2}$", color='k')

plt.title('Money distribution with random trade, chaotic agent selection & interval for saving')
plt.xlabel('Money')
plt.ylabel('Probability Density')
plt.xlim(10**-1)
plt.xscale('log')
plt.yscale('log')
plt.legend()
     
