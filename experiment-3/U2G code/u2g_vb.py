#Python 2.7/3.6 pip install tensorflow==1.13.1 pip install tensorflow-probability==0.5.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import statsmodels.api as sm
import pickle
import os
import scipy
import time
import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import pyreadr
tfd = tfp.distributions
Normal = tfd.MultivariateNormalFullCovariance
slim=tf.contrib.slim

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * \
                        (-log_alpha - tf.nn.softplus(-log_alpha))


def log_p_yz(E, x, y):
    def ff(E, x, y, sigma, sigma_beta):
        idx = tf.squeeze(tf.where(tf.not_equal(E,0)))       
        X1 = tf.gather(x,idx,axis=1)
        X1 = tf.reshape(X1,[args.N,-1])
        V0 = tf.matmul(X1, tf.transpose(X1))
        cov = (sigma_beta**2)*V0 + (sigma**2) * tf.eye(args.N)
        return cov
    cov = tf.cond(tf.greater(tf.reduce_sum(E), 1e-2), lambda:ff(E, x, y,  \
                  args.sigma, args.sigma_beta),\
                  lambda:(args.sigma**2) * tf.eye(args.N))
    mean = tf.zeros(args.N)
    mvn = Normal(mean, cov)
    return mvn.log_prob(y)


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # General info
    parser.add_argument('--name', '-n', default='exp3_u2gvb', help='exp,model name')
    parser.add_argument('--grad', '-g', type=str, default='u2g', \
                        help='reinforce, arm, u2g')
    parser.add_argument('--path', default='SNR/', help='The path results to be saved.')
        
    # Algorithm parameters
    parser.add_argument('--lmbd', type=float, default=8.0, \
                        help='Lagrangian parameter, increase with low SNR ')
    parser.add_argument('--K', '-k', type=int, default=20, help='number of MC sample')
    parser.add_argument('--lr', '-l', type=float, default=0.1, help='base lr')
    parser.add_argument('--iter',  type=int, default=10000, help='maximal iter')
    parser.add_argument('--sigma_beta',  type=float, default=1.0, \
                        help='variance of slab prior')
    parser.add_argument('--sigma',  type=float, default=1.0, help='likelihood variance')    
    
    # Model setting
    parser.add_argument('--N', type=int, default=102, help='number of data')
    parser.add_argument('--P', '-b', type=int, default=1000, help='dimension size')
    parser.add_argument('--corr', type=str, default='medium')
    
    
    args = parser.parse_args()
    lr = (args.lr/args.lmbd)
    eps = 1e-8
    
    for snr in np.exp(np.linspace(0,2.302585092994046,10)):
        path = os.path.join(args.path, f'{args.name}',f'{np.round(snr,2)}', f'{args.lmbd}')
        if not os.path.exists(path):
            os.makedirs(path) 
        for count, seed in enumerate(np.arange(2010,2020,1)): 
            result = pyreadr.read_r('prostate.RData') 
            X = result["X"].values
            COV = np.cov(X.T) 
            start = time.time()
            beta_true = np.zeros(args.P)
            # [1,3,5,7,10] for mod_corr, [0,1,2,3,4] for high_corr
            indexs = [0,1,2,3,4] if args.corr == 'high' else [1,3,5,7,10] 
            beta_true[indexs] = 1
            sigma = np.sqrt(np.matmul(np.matmul(beta_true, COV),beta_true) / snr) 
            noise = np.random.normal(scale = sigma, size = args.N)
            Y = np.matmul(X,beta_true) + noise 
            z_true = (beta_true>0)  
            
            
            start = time.time()
            tf.reset_default_graph(); 
             
            x = tf.placeholder(tf.float32,[args.N, args.P],name='data_x')  
            y = tf.placeholder(tf.float32,[args.N],name='data_y')   
            
            phi = tf.get_variable("phi", dtype=tf.float32, \
                                  initializer=tf.zeros([args.P])-2.0)
            prob = tf.sigmoid(phi)
            
            def fun(E):
                logit_pi = - args.lmbd/(2 * args.sigma**2)    
                log_p_y_given_z = log_p_yz(E, x, y)
                log_p_z = tf.reduce_sum(bernoulli_loglikelihood(E, logit_pi))
                log_q_z = tf.reduce_sum(bernoulli_loglikelihood(E, phi))
                return log_p_z+log_p_y_given_z-log_q_z
                
            u_noise = tf.random_uniform(shape=[args.K,args.P],maxval=1.0)  #K*P
            E1 = tf.cast(u_noise>tf.sigmoid(-phi),tf.float32)
            E2 = tf.cast(u_noise<tf.sigmoid(phi),tf.float32)   
            if args.grad in {'arm','u2g'}:
                Fun1 = tf.map_fn(fun, E1)       
                Fun2 = tf.map_fn(fun, E2)
                F_term = Fun1 - Fun2
                elbo = tf.reduce_mean(Fun2)
            elif args.grad == 'reinforce' :
                F_term = tf.map_fn(fun, E2)
                elbo = tf.reduce_mean(Fun2)
            else:
                raise ValueError('No grad defined')
            if len(np.shape(F_term))<2:
                F_term = F_term[:,None]  #K*1
            
            if args.grad == 'arm':
                G = F_term*(u_noise - 0.5)
                phi_tile = tf.tile(phi[None,:], [args.K,1]) #K*P
                mask = tf.to_float(tf.abs(E1 - E2))
                G_mask = G * mask
                grad = tf.reduce_mean(G_mask,axis=0) #P,
            elif args.grad == 'u2g':
                phi_tile = tf.tile(phi[None,:], [args.K,1]) #K*P
                current_prob = tf.sigmoid(phi_tile)
                G = F_term * tf.to_float(E1 - E2) * \
                                tf.maximum(current_prob,1-current_prob)/ 2.0
                grad = tf.reduce_mean(G,axis=0) #P,
            elif args.grad == 'reinforce':
                phi_tile = tf.tile(phi[None,:], [args.K,1]) #K*P
                current_prob = tf.sigmoid(phi_tile)
                G = F_term * (tf.to_float(E2)-current_prob)
                grad = tf.reduce_mean(G,axis=0) #P,
                            
            train_opt = tf.train.GradientDescentOptimizer(lr)
            
            gradvars = zip([-grad], [phi])
            train_op = train_opt.apply_gradients(gradvars)
          
            init_op=tf.global_variables_initializer()
            
            sess=tf.InteractiveSession()
            sess.run(init_op)    
            cost_r = []; entropy_r=[]; 
            
            start = time.time()
            for i in range(args.iter):   
                _,cost = sess.run([train_op, elbo],{x:X, y:Y})
                cost_r.append(cost)  
                #stopping criterion: 
                probz = np.squeeze(sess.run(prob))           
                entropy = np.sort(- probz * np.log(probz + eps))
                entropy_r.append(np.mean(entropy))
                if np.mean(entropy[-10:])<0.2:
                    break  
            duration = time.time()-start
            
            n_fail = 1 if i == (args.iter-1) else 0
            
            probability = np.squeeze(sess.run(prob))
            z_hat = np.array(probability>0.5).astype(np.int32)  #posterior q(z) MAP    
            beta_hat = np.zeros([args.P])
            if sum(np.abs(z_hat))>eps:
                X1 = np.take(X, np.squeeze(np.where(z_hat != 0)), axis=1)
                if sum(np.abs(z_hat)) == 1:
                    X1 = X1[:,None]
                beta_hat0 = scipy.linalg.lstsq(X1, Y,cond=eps, lapack_driver= 'gelsy')[0]
                beta_hat[np.where(z_hat != 0)] = beta_hat0    
            
            #Calculate metric
            mse = np.mean(np.square(Y - np.matmul(X,beta_hat)))
            sigma = np.sqrt(np.matmul(np.matmul(beta_true, COV),beta_true) / snr) 
            
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
            
            sess.close()
            
            print(args.grad,': prec =', round(prec,2), 'rec =', round(rec,2), 'Iter =', i)
            print('Time =', round(duration,2))
            
            
            all_ = [prec, rec, F1, nonzero, RR, RTE_g, PVE, corr, incorr, n_fail, RTE_ols, mse, duration, args]
            pickle.dump(all_,open(os.path.join(path, f'{count}.pkl'),'wb'))
        
            
