data_gen <- function(N, P, sigma, seed, rho, beta_true, V_true){
  M = rep(0, P)
  ######
  set.seed(seed)
  ######
  X = mvrnorm(N, mu = M, Sigma = V_true)
  noise = rnorm(N, mean = 0, sd = sigma)
  Y = as.vector(X %*% beta_true + noise)
  return(list(X=X, Y=Y, V=V_true, beta_true=beta_true, P=P))
}

norm <- function(x) sqrt(apply(x^2,1,sum)) 

get_sigma_from_snr = function(beta_true, cov, snr) {
  sigma = sqrt(t(beta_true) %*% cov %*% beta_true / snr)
  return(sigma)
}