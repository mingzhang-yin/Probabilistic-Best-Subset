
library("MASS")
library(parallel)

source('src.nosync/utils/fit_utils.R')
source('src.nosync/utils/data_utils.R')
out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "SNR", "RHO", "seed")


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


sim <- function(seed, N ,P, T, snr, rho, method){
  dt_out <- data_gen_from_snr(N, P, snr, seed, rho)
  dta <- dt_out[['data']]
  sigma <- dt_out[['sigma']]
  dt_out_validation <- data_gen_from_snr(N, P, snr, seed+1000, rho)
  data_validation <- dt_out_validation[['data']]
  beta_true <- dta$beta_true
  if (method == "SCAD") {
    beta_intercept = fit_validate(dta, data_validation, method)
    beta_hat = beta_intercept$beta
  }
  else if (method == "lasso") {
    beta_hat = fit_lasso(dta, data_validation)
  }
  
  z_true <- c(abs(beta_true)>1e-3)
  z_hat <- c(abs(beta_hat)>1e-3)
  
  TP = sum(z_true * z_hat)    #true is 1, predict 1
  FP = sum((1-z_true) * z_hat) #true is 0, predict 1
  FN = sum(z_true * (1-z_hat)) #true is 1, predict 0
  prec = TP/(TP + FP)
  rec = TP/(TP + FN)
  F1 = 2 * prec * rec / (prec + rec)
  nonzero = sum(z_hat)
  
  RTE = 1 + t(as.matrix(beta_hat - beta_true)) %*% dta$V %*% as.matrix(beta_hat-beta_true) / (sigma^2)
  RR = (RTE-1)/snr
  PVE = 1 - RTE/(snr+1)
  
  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, snr, rho, seed)
  names(out) = out_cols
  return(out)
}

exp_varying_snr = function(N, P, rho, work_dir, exp_dir){
  nreps = 30
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols

   # lasso
  method = "lasso"
  idx = 1
  
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (snr in exp(seq(0, log(10), length.out=10))) {
      out_cur = sim(seed, N, P, T, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, exp_dir, "lasso_snr_out.csv"))
  write.csv(time, file=file.path(work_dir, exp_dir, "lasso_snr_time.csv"))

  # SCAD
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  idx = 1
  method = "SCAD"
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (snr in exp(seq(0, log(10), length.out=10))) {
      out_cur = sim(seed, N, P, T, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, exp_dir, "scad_snr_out.csv"))
  write.csv(time, file=file.path(work_dir, exp_dir, "scad_snr_time.csv"))
}

exp_varying_rho = function(N, P, snr, work_dir, exp_dir) {
  nreps = 30
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols

   # lasso
  method = "lasso"
  idx = 1
  
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (rho in seq(0, 1, length.out=10)) {
      out_cur = sim(seed, N, P, T, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  #write.csv(out, file=file.path(work_dir, exp_dir, "lasso_rho_out.csv"))
  #write.csv(time, file=file.path(work_dir, exp_dir, "lasso_rho_time.csv"))

  # SCAD
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  idx = 1
  method = "SCAD"
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (rho in seq(0, 1, length.out=10)) {
      out_cur = sim(seed, N, P, T, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  #write.csv(out, file=file.path(work_dir, exp_dir, "scad_rho_out.csv"))
  #write.csv(time, file=file.path(work_dir, exp_dir, "scad_rho_time.csv"))
}

main = function() {
  N = 60
  P = 200
  rho = 0.5
  snr = 3
  work_dir="src.nosync"
  exp_dir = "experiment-1"
  
  #exp_varying_rho(N, P, snr, work_dir, exp_dir)
  exp_varying_snr(N, P, rho, work_dir, exp_dir)
  
}

main()