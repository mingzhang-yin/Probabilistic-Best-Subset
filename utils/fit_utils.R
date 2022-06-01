library("ncvreg")
library("glmnet")

fit_validate = function(data_train, data_validate, method) {
  result = ncvreg(data_train$X, data_train$Y, family="gaussian", penalty=method)
  n_lambda = length(result$lambda)
  mse_eval = rep(0, n_lambda)
  for (i in 1:n_lambda) {
    Y_pred = data_validate$X %*% result$beta[2:(data_validate$P + 1), i] + result$beta[1, i]
    mse_eval[i] = sqrt(sum((Y_pred - data_validate$Y)**2))
  }
  min_idx = which.min(mse_eval)
  return(list(beta=result$beta[2:(data_validate$P+1),min_idx], intercept=result$beta[1,min_idx]))
}


fit_lasso = function(data_train, data_validate) {
    result = glmnet:::glmnet.path(data_train$X, data_train$Y, intercept=FALSE)
    n_lambda = length(result$lambda)
    mse_eval = rep(0, n_lambda)
    for (i in 1:n_lambda) {
        Y_pred = data_validate$X %*% result$beta[, i]
        mse_eval[i] = sqrt(sum((Y_pred - data_validate$Y)**2))
    }
    min_idx = which.min(mse_eval)
    return(result$beta[,min_idx])

}