#Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import statsmodels.api as sm
import pickle
import os
import scipy
import time
import argparse


#%%
def sigmoid(z):
    return(1/(1+np.exp(-z)))

def F(E,X,Y, lmbd = 5):
    if np.sum(E)>0:
        a1 = np.squeeze(np.where(E != 0))
        X1 = np.take(X, a1, axis=1)
        if len(np.shape(X1))<2:
            X1 = X1[:,None]
        params = scipy.linalg.lstsq(X1, Y,cond=1e-10, lapack_driver='gelsy')[0]
        y_hat = X1.dot(params)
        f_value = 0.5 * np.sum(np.square(Y - y_hat)) + lmbd * np.sum(E)
    else:
        f_value = 0.5 * np.sum(np.square(Y)) + lmbd * np.sum(E)
        params = np.array(E) * 0.
    return f_value, params


def data_gen(N ,P, snr, seed=1):
    if args.name[:4] == 'exp1':
        beta_true = np.array([3,1.5,0,0,2,0,0,0]+[0]*(P-8))
    elif args.name[:4] == 'exp2':
        beta_true = np.array([1.]*10+[0]*(P-10))  
    
    def cov(P, rho):
        V = np.empty([P,P])
        for i in range(P):
            for j in range(i):
                V[i,j] = np.power(rho, np.abs(i-j))
                V[j,i] = V[i,j]
        for i in range(P):
            V[i,i] = 1    
        return V
    V = cov(P, args.rho)
    M = np.array([0]*P)
    ############
    np.random.seed(seed)
    ############
    X = np.random.multivariate_normal(mean=M, cov=V, size = N)
    sigma = np.sqrt(np.matmul(np.matmul(beta_true, V),beta_true) / snr) 
    noise = np.random.normal(scale = sigma, size = N)
    Y = np.matmul(X,beta_true) + noise
    Y = Y.astype(np.float32) 
    X = X.astype(np.float32) 
    return X, Y, V, beta_true
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # General info
    parser.add_argument('--name', '-n', default='exp1_u2g', help='exp,model name')
    parser.add_argument('--grad', '-g', type=str, default='u2g', \
                        help='reinforce, arm, u2g')
    parser.add_argument('--path', default='SNR/', help='The path results to be saved.')
        
    # Algorithm parameters
    parser.add_argument('--lmbd', type=float, default=20.0, \
                        help='Lagrangian parameter, increase with low SNR ')
    parser.add_argument('--K', '-k', type=int, default=20, help='number of MC sample')
    parser.add_argument('--lr', '-l', type=float, default=0.05, help='base lr')
    parser.add_argument('--iter',  type=int, default=10000, help='maximal iter')
    
    # Model setting
    #parser.add_argument('--snr',  type=float, default=3.0)
    parser.add_argument('--N', type=int, default=60, help='number of data')
    parser.add_argument('--P', '-b', type=int, default=200, help='dimension size')
    parser.add_argument('--rho',  type=float, default=0.5)
    
    args = parser.parse_args()
    lr = (args.lr/args.lmbd)
    eps = 1e-8
    
    for snr in np.exp(np.linspace(0,2.303,10)):
        path = os.path.join(args.path, f'{args.name}',f'{np.round(snr,2)}', f'{args.lmbd}')
        if not os.path.exists(path):
            os.makedirs(path) 
        for count, seed in enumerate(np.arange(2000,2020,1)):  
            #Generate data
            X,Y,COV,beta_true = data_gen(args.N,args.P,snr,seed=seed)   
            z_true = (beta_true>0)  
            phi = np.zeros(args.P) - 2.0
            start = time.time()
            for i in range(1,args.iter):
                u_noise = np.random.uniform(0,1,[args.K,args.P])
                E1 = (u_noise>sigmoid(-phi)).astype(np.float32)
                E2 = (u_noise<sigmoid(phi)).astype(np.float32)
                Fs = []; 
                for j in range(args.K):
                    if args.grad in {'arm','u2g'}:
                        f1, _ = F(E1[j],X,Y, args.lmbd) 
                        f2, _ = F(E2[j],X,Y, args.lmbd)
                        Fs.append(f1 - f2)
                    elif args.grad == 'reinforce' :
                        f2, _ = F(E2[j],X,Y, args.lmbd)
                        Fs.append(f2)
                    else:
                        raise ValueError('No grad defined')
                if args.grad == 'arm':       
                    G = np.array(Fs)[:,None] * (u_noise - 0.5)
                    mask = np.abs(E1-E2)
                    G = G * mask
                    grad = np.mean(G,axis=0)
                elif args.grad == 'u2g':
                    G = np.array(Fs)[:,None] * (E1 - E2) * \
                        np.maximum(sigmoid(phi),1-sigmoid(phi))[None,:]/ 2.0
                    grad = np.mean(G,axis=0)
                elif args.grad == 'reinforce':
                    G = np.array(Fs)[:,None] * (E2-sigmoid(phi)[None,:]) # [K,1], 
                    grad = np.mean(G,axis=0)
                else:
                    raise ValueError('No grad defined')
                    
                phi = phi - lr * grad
                
                # Stopping Criterion
                probz = sigmoid(phi)
                entropy = np.sort(- probz * np.log(probz + eps))
                if np.mean(entropy[-10:])<0.15:
                    break
            duration = time.time()-start
            
            n_fail = 1 if i == (args.iter-1) else 0
        
            z_hat = (sigmoid(phi) > 0.5)   #MLE
            _, beta_hat0 = F(z_hat,X,Y)
            beta_hat = np.zeros(args.P)
            if sum(z_hat)>eps:
                beta_hat[np.where(z_hat>0)] = beta_hat0 
            nonzero = sum(z_hat)
            
            sigma = np.sqrt(np.matmul(np.matmul(beta_true, COV),beta_true) / snr) 
            
            #Calculate metrics
            mse = np.mean(np.square(Y - np.matmul(X,beta_hat)))
            ols_results = sm.OLS(Y, X).fit()
            beta_ols = ols_results.params
            
            TP = np.sum(z_true * z_hat) 
            FP = np.sum((1-z_true) * z_hat) #true is 0, predict 1
            FN = np.sum(z_true * (1-z_hat)) #true is 1, predict 0
            
            prec = TP/(TP + FP + eps)   
            rec = TP/(TP + FN + eps)
            F1 = 2 * prec * rec / (prec + rec + eps)   
            
            nonzero = sum(z_hat)
               
            RTE_g = 1 + np.squeeze(np.matmul(np.matmul((beta_hat-beta_true), COV),\
                                    (beta_hat-beta_true).T) / (sigma**2))
                
            RR = (RTE_g-1)/snr    
            PVE = 1 - RTE_g/(snr+1)
            
            RTE_ols = 1 + np.matmul(np.matmul((beta_true-beta_ols), COV),\
                                (beta_true-beta_ols)) / (sigma**2) 
            corr = np.sum((1-z_true) * (1-z_hat))
            incorr = FN
            
                   
            print(args.grad,': prec =', round(prec,2), 'rec =', round(rec,2), 'Iter =', i)
            print('Time =', round(duration,2))
            
        
            all_ = [prec, rec, F1, nonzero, RR, RTE_g, PVE, corr, incorr, n_fail, RTE_ols, mse, duration, args]
            pickle.dump(all_,open(os.path.join(path, f'{count}.pkl'),'wb'))



