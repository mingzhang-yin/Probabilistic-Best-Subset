library("L0Learn")
library("Matrix")
library("MASS")
library("parallel")
library("tidyr")

source('src/utils/data_utils.R')
source('src/utils/fit_utils.R')

out_cols = c("prec", "rec", "F1", "nonzero", "RR", "RTE", "PVE", "N", "seed")
norm <- function(x) sqrt(apply(x^2,1,sum)) 


get_beta_V = function(rho, P) {
  beta_true = rep(0, P)
  beta_true[1:10] = 1.0
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
  print(sigma)
  dt_out_validation <- data_gen_from_snr(N, P, snr, seed+1000, rho)
  data_validation <- dt_out_validation[['data']]
  
  # fit <- L0Learn.fit(dta$X, dta$Y, penalty="L0", maxSuppSize=T, intercept=FALSE, maxIters = 1000, 
  #                    lambdaGrid =list(seq(from=1, to=0.01, length.out=100))) #default maxIters = 200
  fit <- L0Learn.fit(dta$X, dta$Y, penalty="L0", maxSuppSize=T, intercept=FALSE, maxIters = 1000, gammaMax=0) #default maxIters = 200
  
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
  prec = TP/(TP + FP + 1e-8)
  rec = TP/(TP + FN + 1e-8)
  F1 = 2 * prec * rec / (prec + rec + 1e-8)
  nonzero = length(bs.supp)
  
  RTE = 1 + t(as.matrix(beta_hat-beta_true)) %*% dta$V %*% as.matrix(beta_hat-beta_true) / (sigma^2)
  SNR = t(as.matrix(beta_true)) %*% dta$V %*% as.matrix(beta_true) / (sigma^2)    
  RR = (RTE-1)/SNR
  PVE = 1 - RTE/(SNR+1)
  
  
  out = c(prec, rec, F1, nonzero, RR, RTE, PVE, N, seed)
  
  names(out) = out_cols
  return(out)
}


# run ---------------------------------------------------------------------

main = function(){
  P = 1000; T = 10; rho0 = 0.; snr0 = 5
  nreps = 10
  
  out = data.frame(matrix(ncol=length(out_cols), nrow=nreps*10))
  colnames(out) = out_cols

  method = "FastMIO"

  start_time <- Sys.time()
  out = c()
  Ns <- seq(60, 141, by=8)
  for (N in Ns) {
    print(N)
    start_time <- Sys.time()
    numWorkers <- detectCores()-1
    # Initiate cluster -- FORK only works on a Mac
    cl <- makeCluster(numWorkers,type="FORK")
    res <- parLapply(cl, 1:nreps, sim, N=N, P=P, T=T, snr=snr0, rho=rho0)
    stopCluster(cl)
    end_time <- Sys.time()
    
    results =  do.call(rbind, res)
    out = rbind(out,results)
  }
  
  end_time <- Sys.time()
  time <- round(as.numeric(difftime(end_time, start_time, units="secs")), 1) / (length(Ns)*nreps)
  out <- as.data.frame(out)
  out <- (out %>% drop_na())
  write.csv(out, file="src/experiment-2/FastMIO_n_out.csv")
  write.csv(time, file="src/experiment-2/FastMIO_n_time.csv")

}

main()



