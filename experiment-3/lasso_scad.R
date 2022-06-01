library("MASS")
library("logging")
addHandler(writeToConsole)

source('src/utils/data_utils.R')
source('src/utils/fit_utils.R')

out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "SNR", "seed" )

get_ground_truth = function(P) {
  load("src/experiment-3/prostate.RData")
  beta_true = rep(0, P)
  beta_true[c(2,4,6,8,11)] = 1.0 #1,2,3,4,5 / 2,4,6,8,11
  X = as.matrix(X)
  V = cov(X)
  return(list(beta_true=beta_true, V_true=V, X=X))
}

data_gen_prostate <- function(N ,P, sigma, seed, gt){
  ######
  set.seed(seed)
  ######
  noise = rnorm(N, mean = 0, sd = sigma)
  Y = as.vector(gt$X %*% gt$beta_true + noise)

  return(list(X=gt$X, Y=Y, V=gt$V_true, beta_true=gt$beta_true, P=P))
}

sim <- function(seed, N ,P, gt, snr, method){

  sigma = get_sigma_from_snr(gt$beta_true, gt$V_true, snr)
  dta <- data_gen_prostate(N, P, sigma, seed, gt)
  beta_true <- dta$beta_true
  data_validate = data_gen_prostate(N, P, sigma, seed+1000, gt)
  if (method == "SCAD") {
    beta_intercept = fit_validate(dta, data_validate, 'MCP')
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
  
  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, snr, seed)
  names(out) = out_cols
  return(out)
}

main = function(work_dir, exp_dir) {
  N_test = 100; N = 102; P = 1000;
  gt = get_ground_truth(P)
  nreps = 10
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols

  # lasso
  loginfo("Running lasso experiment.")
  method = "lasso"
  idx = 1
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (snr in exp(seq(0, log(10), length.out=10))) {
      out_cur = sim(seed, N, P, gt, snr, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, exp_dir, "lasso_snr_out.csv"))
  write.csv(time, file=file.path(work_dir, exp_dir, "lasso_snr_time.csv"))

  loginfo("Running scad experiment.")
  # SCAD
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  idx = 1
  method = "SCAD"
  start_time <- Sys.time()
  for (seed in 1 : nreps) {
    print(seed)
    for (snr in exp(seq(0, log(10), length.out=10))) {
      out_cur = sim(seed, N, P, gt, snr, method)
      out[idx,] = out_cur
      idx = idx+1
    }
  }
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (10*nreps)
  write.csv(out, file=file.path(work_dir, exp_dir, "scad_snr_out.csv"))
  write.csv(time, file=file.path(work_dir, exp_dir, "scad_snr_time.csv"))
}

# run ---------------------------------------------------------------------
work_dir="src"
exp_dir = "experiment-3"
main(work_dir, exp_dir)
