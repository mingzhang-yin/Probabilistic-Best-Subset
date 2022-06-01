from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from matplotlib import pyplot as plt
plt.style.use("ggplot")
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
slim=tf.contrib.slim

Normal = tfd.MultivariateNormalFullCovariance
    
#%%
if 1:
    P = 1024; N = 500; T = 10; 
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
    
    
    COV = np.eye(P)
    SNR = np.matmul(np.matmul(beta_true, COV),beta_true) / (sigma**2) 
    print('SNR=', SNR)

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def lognormal(log_tau,h1,h2):
    logpdf = - 0.5 * ((log_tau-h1)**2)/ tf.exp(2*h2) - log_tau - h2
    return logpdf


def log_p_yz(E):
    def ff(E, x, y, sigma, sigma_beta):
        idx = tf.squeeze(tf.where(tf.not_equal(E,0)))       
        X1 = tf.gather(x,idx,axis=1)
        X1 = tf.reshape(X1,[N,-1])
        V0 = tf.matmul(X1, tf.transpose(X1))
        cov = (sigma_beta**2)*V0 + (sigma**2) * tf.eye(N)
        return cov
    cov = tf.cond(tf.greater(tf.reduce_sum(E), 1e-2), lambda:ff(E, x, y,  sigma, sigma_beta),\
                  lambda:(sigma**2) * tf.eye(N))
    mean = tf.zeros(N)
    mvn = Normal(mean, cov)
    return mvn.log_prob(y)

def fun(E):
    logit_pi = - lmbd0
    theta = lmbd0 # >0
    log_tau = -2*tf.log(sigma_beta)
    
    log_p_y_given_z = log_p_yz(E)
    log_p_z = tf.reduce_sum(bernoulli_loglikelihood(E, logit_pi))
    log_q_z = tf.reduce_sum(bernoulli_loglikelihood(E, phi))
    log_p_theta = (a2 - 1) * tf.log(theta) - b2 * theta
    log_q_theta = lognormal(tf.log(theta),s1,s2)
    log_p_tau = (a1 - 1) * log_tau - b1 / (sigma_beta**2)
    log_q_tau = lognormal(log_tau,h1,h2)
    return log_p_z+log_p_y_given_z+log_p_theta+log_p_tau-log_q_z-log_q_theta-log_q_tau

def sample_post(z, X, Y, sigma, sigma_beta):
    #sample from p(alpha|z,y)
    beta_post = np.zeros(P)
    beta_map = np.zeros(P)
    if np.sum(z)>1e-2:
        mu0 = tf.zeros(N)
        a1 = np.squeeze(np.where(z != 0))
        X1 = np.take(X, a1, axis=1)
        mu0 = np.take(mu0, a1)
        cov_posterior = np.linalg.inv(np.matmul(X1.T,X1)/(sigma**2) + np.eye(np.sum(z))/(sigma_beta**2))
        mu = np.matmul(cov_posterior, mu0/(sigma_beta**2)+np.matmul(X1.T,Y)/(sigma**2))  
        beta_post0 = np.random.multivariate_normal(mu, cov_posterior, size = 1)   
        beta_post[np.where(z>0)] = beta_post0
        beta_map[np.where(z>0)] = mu

    return beta_post, beta_map

#%%
K = 5
tf.reset_default_graph();

train_hyper = 0; train_lmbd = 0;
if train_hyper:
    beta_star = tf.get_variable("beta", dtype=tf.float32,initializer=tf.zeros([P]),trainable=False)
    a1 = b1 = 0.01; a2 = 15.; b2 = 5.;
    s1 = tf.get_variable("loggauss_m1", dtype=tf.float32, initializer=0.);
    s2 = tf.get_variable("loggauss_logstd1", dtype=tf.float32, initializer=-2.);
    h1 = tf.get_variable("loggauss_m2", dtype=tf.float32, initializer=0.);
    h2 = tf.get_variable("loggauss_logstd2", dtype=tf.float32, initializer=0.);
    sigma_beta = 1 / tf.exp(0.5*(h1+tf.random_normal([1])*tf.exp(h2))) # 1/sqrt(precision)
    if train_lmbd:
        lmbd0 = tf.exp(s1 + tf.random_normal([1])*tf.exp(s2))
        hypers = [s1,s2,h1,h2]
    else:
        lmbd0 = 10/(2*sigma**2)
        print("lambda = ", 2*lmbd0*sigma**2)
        hypers = [h1,h2]
else:
    a1 = b1 = 1.; a2 = 1.; b2 = 1.;
    s1 = 0. ; s2 = -2.; h1 = 0.; h2 = 0.
    sigma_beta = 1.0
    lmbd0 = 0.01/(2*sigma**2)  #0.1 for cs, 10 for exp2
    print("lambda = ", 2*lmbd0*sigma**2)

x = tf.placeholder(tf.float32,[N, P],name='data_x')  
y = tf.placeholder(tf.float32,[N],name='data_y') 


phi = tf.get_variable("phi", dtype=tf.float32, initializer=tf.random.normal([P]))#tf.zeros([P]))
prob = tf.sigmoid(phi)

u_noise = tf.random_uniform(shape=[K,P],maxval=1.0) 
E1 = tf.cast(u_noise>tf.sigmoid(-phi),tf.float32)
E2 = tf.cast(u_noise<tf.sigmoid(phi),tf.float32)

Fun1 = tf.map_fn(fun, E1)
elbo = tf.reduce_mean(Fun1)

Fun2 = tf.map_fn(fun, E2)
F_delta = Fun1 - Fun2

if len(np.shape(F_delta))<2:
    F_delta = F_delta[:,None]
G = F_delta*tf.to_float(E1 - E2) * \
        tf.maximum(current_prob,1-current_prob)/2.0
grad = tf.reduce_mean(G,axis=0)



global_step = tf.Variable(0, trainable=False)

learning_rate1 = 0.0001
train_opt = tf.train.GradientDescentOptimizer(learning_rate1)

gradvars = zip([-grad], [phi])
train_op1 = train_opt.apply_gradients(gradvars)

if train_hyper:
    learning_rate2 = 0.01
    train_op2 = tf.train.GradientDescentOptimizer(learning_rate2).minimize(-elbo,var_list=hypers)
    with tf.control_dependencies([train_op1, train_op2]):
        train_op = tf.no_op()
else:
    train_op = train_op1

init_op=tf.global_variables_initializer()


#%%
  
sess=tf.InteractiveSession()
sess.run(init_op)    
cost_r = []; entropy_r=[]; entropy2_r=[];
import time
start = time.time()
current = start
for i in range(6000):   
    _,cost = sess.run([train_op, elbo],{x:X, y:Y})
    cost_r.append(cost)
    
    probz = np.squeeze(sess.run(prob))       
    entropy = np.sort(- probz * np.log(probz + 1e-8))
    entropy_r.append(np.mean(entropy))
    entropy2_r.append(np.mean(entropy[-10:]))
    if np.mean(entropy[-10:])<0.1:
        break
    if i%500==0:
        print('iter = ', i, 'time = ', time.time()-current) 
        current = time.time()
    

#%%
probability = np.squeeze(sess.run(prob))
z_hat = np.array(probability>0.5).astype(np.int32)  #posterior q(z) MAP
print(np.where(z_hat>0))
print(np.where(z_true>0))


