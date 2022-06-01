library("MASS")
library("logging")
library(parallel)
addHandler(writeToConsole)

source('src.nosync/utils/data_utils.R')
source('src.nosync/utils/fit_utils.R')
directory <- "/mnt/c/projects/bestsubset/src.nosync/experiment-2/lasso_scad_out"
out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "N", "seed" )

get_ground_truth = function(rho, P) {
  beta_true = rep(0, P)
  beta_true[1:10] = 1.0
  V = matrix(rep(0, len=P^2), nrow = P)
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

sim <- function(seed, N ,P, beta_true, V_true, snr, rho, method){
  sigma = get_sigma_from_snr(beta_true, V_true, snr)
  dta <- data_gen(N, P, sigma, seed, rho, beta_true, V_true)
  data_validate = data_gen(N, P, sigma, seed+1000, rho, beta_true, V_true)
  if (method == "SCAD") {
    beta_intercept = fit_validate(dta, data_validate, method)
    beta_hat = beta_intercept$beta
  }
  else if (method == "lasso") {
    beta_hat = fit_lasso(dta, data_validate)
  }

  bs.supp <- which(beta_hat != 0)
  
  z_true <- c(abs(beta_true)>0)
  z_hat <- c(abs(beta_hat)>0)
  
  TP = sum(z_true * z_hat)     #true is 1, predict 1
  FP = sum((1-z_true) * z_hat) #true is 0, predict 1
  FN = sum(z_true * (1-z_hat)) #true is 1, predict 0
  prec = TP/(TP + FP)
  rec = TP/(TP + FN)
  F1 = 2 * prec * rec / (prec + rec)
  nonzero = length(bs.supp)
  
  RTE = 1 + t(as.matrix(beta_hat-beta_true)) %*% dta$V %*% as.matrix(beta_hat-beta_true) / (sigma^2)
  RR = (RTE-1) / snr
  PVE = 1 - RTE / (snr + 1)
  
  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, N, seed)
  names(out) = out_cols
  return(out)
}

main = function(work_dir) {
  N = 100; P = 1000; T = 10; rho = 0.; snr = 5
  beta_V = get_ground_truth(rho, P)
  beta_true = beta_V$beta_true
  V_true = beta_V$V
  nreps = 20
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols

  # lasso
  loginfo("Running lasso experiment.")
  method = "lasso"
  idx = 1
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (N in seq(60, 141,by=8)) {
      out_cur = sim(seed, N, P, beta_true, V_true, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, "experiment-2/lasso_n_out.csv"))
  write.csv(time, file=file.path(work_dir, "experiment-2/lasso_n_time.csv"))

  loginfo("Running scad experiment.")
  # SCAD
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  idx = 1
  method = "SCAD"
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (N in seq(60, 141,by=8)) {
      out_cur = sim(seed, N, P, beta_true, V_true, snr, rho, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, "experiment-2/scad_n_out.csv"))
  write.csv(time, file=file.path(work_dir, "experiment-2/scad_n_time.csv"))
}

# run ---------------------------------------------------------------------
work_dir="src.nosync"
main(work_dir)
