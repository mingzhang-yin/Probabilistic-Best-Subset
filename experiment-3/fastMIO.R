library("L0Learn")
library("Matrix")
library("MASS")
library("parallel")

out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "SNR", "seed" )
norm <- function(x) sqrt(apply(x^2,1,sum)) 

source('src/utils/data_utils.R')
source('src/utils/fit_utils.R')

get_ground_truth = function(P) {
  load("/experiment-3/prostate.RData")
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


sim <- function(seed, N ,P, gt, snr){
  set.seed(seed)
  sigma = get_sigma_from_snr(gt$beta_true, gt$V_true, snr)
  dta <- data_gen_prostate(N, P, sigma, seed, gt)
  beta_true <- dta$beta_true
  
  data_validate = data_gen_prostate(N, P, sigma, seed+1000, gt)
  
  fit <- L0Learn.fit(dta$X, dta$Y, penalty="L0", intercept=FALSE, maxIters = 1000) #default maxIters = 200
  
  lambdas = fit$lambda[[1]]
  
  # T = 5
  # my_lmbd = lambdas[which(abs(fit$suppSize[[1]]-T)==min(abs(fit$suppSize[[1]]-T)))[1]]
  
  
  n_lambda = length(lambdas)
  mse_eval = rep(0, n_lambda)
  for (i in 1:n_lambda) {
    Y_pred = predict(fit, newx = data_validate$X, lambda=lambdas[i])
    mse_eval[i] = sqrt(sum((Y_pred - data_validate$Y)**2))
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
  prec = TP/(TP + FP + 1e-8)
  rec = TP/(TP + FN + 1e-8)
  F1 = 2 * prec * rec / (prec + rec + 1e-8)
  nonzero = length(bs.supp)
  
  RTE = 1 + t(as.matrix(beta_hat-beta_true)) %*% dta$V %*% as.matrix(beta_hat-beta_true) / (sigma^2)
  SNR = t(as.matrix(beta_true)) %*% dta$V %*% as.matrix(beta_true) / (sigma^2)    
  RR = (RTE-1)/SNR
  PVE = 1 - RTE/(SNR+1)
  
  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, snr, seed)

  names(out) = out_cols
  return(out)
}


# run ---------------------------------------------------------------------

main = function(){
  N = 102; P = 1000;
  gt = get_ground_truth(P)
  nreps = 50
  
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols
  
  method = "FastMIO"
  start_time <- Sys.time()
  out = c()
  SNRs = exp(seq(0, log(10), length.out=10))
  for (snr in SNRs) {
    print(snr)
    start_time <- Sys.time()
    numWorkers <- detectCores()-1
    # Initiate cluster -- FORK only works on a Mac
    cl <- makeCluster(numWorkers,type="FORK") 
    res <- parLapply(cl, 1:nreps, sim, N=N, P=P, gt=gt, snr=snr)
    stopCluster(cl)
    end_time <- Sys.time()
    
    results =  do.call(rbind, res)
    out = rbind(out,results)
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (length(SNRs)*nreps)
  out <- as.data.frame(out)
  #out <- (out %>% drop_na())
  write.csv(out, file="src/experiment-3/FastMIO_snr_out.csv")
  write.csv(time, file="src/experiment-3/FastMIO_snr_time.csv")
}

main()















