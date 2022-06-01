library("L0Learn")
library("Matrix")
library("MASS")
library("parallel")

out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "SNR", "RHO", "seed")
norm <- function(x) sqrt(apply(x^2,1,sum)) 

get_beta_V = function(rho, P) {
  beta_true = rep(0, P)
  beta_true[c(1,2,5)] = c(3,1.5,2)
  V = matrix(rep( 0, len=P^2), nrow = P)
  for (i in 1:P){
    for (j in 1:i){
      V[i,j] = rho^(abs(i-j))
      V[j,i] = V[i,j]
    }
  }
  for (i in 1:P){
    V[i,i] = 1
  }
  return(list(beta_true=beta_true, V=V))
}

data_gen <- function(N, P, sigma, seed, rho){
  beta_V = get_beta_V(rho, P)
  beta_true = beta_V$beta_true
  V = beta_V$V
  M = rep(0, P)
  ######
  set.seed(seed)
  ######
  X = mvrnorm(N, mu = M, Sigma = V)
  noise = rnorm(N, mean = 0, sd = sigma)
  Y = as.vector(X %*% beta_true + noise)
  return(list(X=X, Y=Y, V=V, beta_true=beta_true, P=P))
}


data_gen_from_snr = function(N, P, snr, seed, rho) {
  beta_V = get_beta_V(rho, P)
  sigma = get_sigma_from_snr(beta_V$beta_true, beta_V$V, snr)
  return(list(data = data_gen(N, P, sigma, seed, rho), sigma = sigma))
}



sim <- function(seed, N ,P, T, snr, rho){
  dt_out <- data_gen_from_snr(N, P, snr, seed, rho)
  dta <- dt_out[['data']]
  sigma <- dt_out[['sigma']]
  beta_true <- dta$beta_true
  
  dt_out_validation <- data_gen_from_snr(N, P, snr, seed+1000, rho)
  data_validation <- dt_out_validation[['data']]
  
  fit <- L0Learn.fit(dta$X, dta$Y, penalty="L0", maxSuppSize=T, intercept=FALSE, maxIters = 1000, 
                     lambdaGrid =list(seq(from=1, to=0.01, length.out=100))) #default maxIters = 200
  
  lambdas = fit$lambda[[1]]
  
  #my_lmbd = lambdas[which(abs(fit$suppSize[[1]]-T)==min(abs(fit$suppSize[[1]]-T)))[1]]
  
  
  n_lambda = length(lambdas)
  mse_eval = rep(0, n_lambda)
  for (i in 1:n_lambda) {
    Y_pred = predict(fit, newx = data_validation$X, lambda=lambdas[i])
    mse_eval[i] = sqrt(sum((Y_pred - data_validation$Y)**2))
  }
  min_idx = which.min(mse_eval)
  my_lmbd = lambdas[min_idx]
  
  beta_hat = as.matrix(coef(fit, lambda=my_lmbd, gamma=0))
  bs.supp <- which(abs(beta_hat) > 1e-8)
  
  z_true <- c(abs(beta_true)>0)
  z_hat <- c(abs(beta_hat)>0)
  
  TP = sum(z_true * z_hat )    #true is 1, predict 1
  FP = sum((1-z_true) * z_hat) #true is 0, predict 1
  FN = sum(z_true * (1-z_hat)) #true is 1, predict 0
  prec = TP/(TP + FP)
  rec = TP/(TP + FN)
  F1 = 2 * prec * rec / (prec + rec)
  nonzero = length(bs.supp)
  
  RTE = 1 + t(as.matrix(beta_hat-beta_true)) %*% dta$V %*% as.matrix(beta_hat-beta_true) / (sigma^2)
  SNR = t(as.matrix(beta_true)) %*% dta$V %*% as.matrix(beta_true) / (sigma^2)    
  RR = (RTE-1)/SNR
  PVE = 1 - RTE/(SNR+1)
  

  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, snr, rho, seed)

  names(out) = out_cols
  return(out)
}


# run ---------------------------------------------------------------------

main = function(){
  N = 60; P = 200; T = 3; rho0 = 0.5; snr0 = 3
  nreps = 30
  
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  # FastMIO
  method = "FastMIO"
  idx = 1
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (snr in exp(seq(0, log(10), length.out=10))) {
      out_cur = sim(seed, N, P, T, snr, rho0)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file="src.nosync/experiment-1/FastMIO_snr_out.csv")
  write.csv(time, file="src.nosync/experiment-1/FastMIO_snr_time.csv")

  
  out2 = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out2) = out_cols
  idx = 1
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (rho in seq(0, 1, length.out=10)) {
      out_cur = sim(seed, N, P, T, snr0, rho)
      out2[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time2 <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out2, file="src.nosync/experiment-1/FastMIO_rho_out.csv")
  write.csv(time2, file="src.nosync/experiment-1/FastMIO_rho_time.csv")
}

main()















