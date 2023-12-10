#####fix lambda Fig1 a
#sen I
# random agent
# chaotic trade
#####################################
import numpy as np
import math
import numba
import matplotlib.pyplot as plt
import time
from numba import *


start_time = time.time()
@numba.njit(parallel=True, fastmath = True)
def nonsaving(N,z,z1,lan):
    N2 = N*N
    m = np.ones(N)
    for i in range(N2):
        z = [np.random.randint(0, high=N-1),np.random.randint(0, high=N-1)]
        dm =  z1[i]*(1-lan)*m[z[1]] - (1-z1[i])*(1-lan)*m[z[0]]
        m[z[1]] = m[z[1]] - dm 
        m[z[0]] = m[z[0]] + dm 
    return m

#setting values and doing simulation
N = 5000

N2 = N*N
z= np.ones(2*N2).reshape(N2,2)#np.array([0.4913 for i in prange(N2)])

z[0,0] = np.random.random()
z[0,1] = np.random.random()
    
za = 1.032
zb = 1.032
for i in range(N2-1):
    z[i+1,0] = za * (3*z[i,1]+1) * z[i,0] * (1-z[i,0])
    z[i+1,1] = zb * (3*z[i,0]+1) * z[i,1] * (1-z[i,1])

vi=np.hstack((z[:,0],z[:,1]))
z= N*z
z = z.astype(np.int64)

lam = [0,0.5,0.75,0.9]
q= [np.array([]) for i in range(len(lam)) ]

for j in range(len(lam)): 
    for i in range(1000):
        z= np.random.permutation(z)
        vi = np.random.permutation(vi)
        m = nonsaving(N,z,vi,lam[j])
        q[j] = np.hstack((q[j], m))

        
print("--- %s seconds ---" % (time.time() - start_time))
#####################################################################
mark=['^','*','o','x','+']
col =['k','k','k','k','k']
plt.rcParams['figure.dpi'] = 300
for j in np.arange(0,len(lam)):
    m1 = [value for value in list(q[j]) if value != 1.0]
    n, bins = np.histogram(m1, bins = 400, density = 1)
    pos = 0.5* (bins[1:]+bins[:-1])
    plt.scatter(pos, n, color = col[j], label= '$\lambda = $' +str(lam[j]), marker = mark[j])
    
plt.xlim(0,4.5)
plt.ylim(0)
plt.xlabel('Money')
plt.ylabel('Probability Density')
plt.title('Money Distribution with random agent selection,chaotic trade & fix saving')
plt.legend()

