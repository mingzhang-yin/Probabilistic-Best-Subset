from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import scipy

#%%
P = 1024; N = 500; T = 10
np.random.seed(3)
X = np.random.normal(size = [N, P])  
X = X / np.linalg.norm(X, axis=1, keepdims = True)
sigma = 0.005 
noise = np.random.normal(scale = sigma, size = N)
beta_true = np.zeros([P])
label = np.random.permutation(np.arange(P))[:T]
beta_true[label] = np.sign(np.random.normal(size = [T]))

Y = np.matmul(X,beta_true) + noise

z_true = (np.abs(beta_true)>0)

Y = Y.astype(np.float32) 
X = X.astype(np.float32) 
#%%
def sigmoid(z):
    return(1/(1+np.exp(-z)))


def F(E,X,Y, lmbd = 5):
    if np.sum(E)>0:
        a1 = np.squeeze(np.where(E != 0))
        X1 = np.take(X, a1, axis=1)
        
        params = scipy.linalg.lstsq(X1, Y,cond=1e-10, lapack_driver= 'gelsy')[0]
        y_hat = X1.dot(params)
        f_value = 0.5 * np.sum(np.square(Y - y_hat)) + lmbd * np.sum(E)
    else:
        f_value = 0.5 * np.sum(np.square(Y)) + lmbd * np.sum(E)
    return f_value, params
        
    
phi = np.zeros(P)
loss = []
lmbd = 0.08 #increase for low SNR; 5--T1 sigma1.0, 4--T2, 20--T3 sigma3.0 
niter = 6000
K = 5  
lr = (0.02/lmbd)
entropy_r = []
entropy2_r = []

import time
start = time.time()
current = start
for i in range(1,niter):
    u_noise = np.random.uniform(0,1,[K,P])
    E1 = (u_noise>sigmoid(-phi)).astype(np.float32)
    E2 = (u_noise<sigmoid(phi)).astype(np.float32)
    Fs = []; fs = []
    for j in range(len(E1)):
        f1, _ = F(E1[j],X,Y, lmbd) 
        f2, _ = F(E2[j],X,Y, lmbd)
        Fs.append(f1 - f2)
        fs.append((f1+f2)/2)

    G = np.array(Fs)[:,None] * (E1 - E2) * \
        np.maximum(sigmoid(phi),1-sigmoid(phi))[None,:] / 2.0
    grad = np.mean(G,axis=0)
    
    
    phi = phi - lr * grad
    
    probz = sigmoid(phi)
    entropy = np.sort(- probz * np.log(probz + 1e-8))
    entropy_r.append(np.mean(entropy))
    entropy2_r.append(np.mean(entropy[-10:]))
    if i%100 == 0 and i<1000:
        print(i)
        print(time.time()-current)
        current = time.time()

print('iteration is', i, 'time is', time.time()-start)   
#%%
probability = sigmoid(phi)  
z_hat = np.array(probability>0.5).astype(np.int32) 

print(np.where(np.abs(beta_true)>0))
print(np.where(z_hat>0))

_, beta_hat0 = F(z_hat,X,Y)
beta_hat = np.zeros(P)
beta_hat[np.where(z_hat>0)] = beta_hat0


#%%
import pickle as cPickle
import os
directory = os.getcwd()+'/out/'
cPickle.dump([beta_true, beta_hat], open(directory+'fq0005', 'wb'))


