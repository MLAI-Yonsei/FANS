######### functions for partial permutation test #############
### required packages
# library(mvtnorm)
# library(psych)
# library(quadprog)
# library(Matrix)
# library(lmtest)

######### partial permutation test basic function ############

kernel.matrix <- function(X, kernel.list, cov.mat ){
  n = nrow(X)
  if(kernel.list$name == "Gaussian"){
    if( length( kernel.list$alpha ) == 1 ){
      alpha = rep( kernel.list$alpha, ncol(X) )
    }
    K0 = matrix(0, n, n)
    for(jj in 1:ncol(X)){
      K0 = K0 + alpha[jj] * (outer(X[, jj], X[,jj], "-"))^2
    }
    K0 = exp(-1 * K0)
  }
  if(kernel.list$name == "polynomial"){
    K0 = matrix(1, n, n)
    for(jj in 1:ncol(X)){
      K0 = K0 + outer(X[, jj], X[,jj], "*")
    }
    K0 = K0^kernel.list$p
  }
  # K0 = matrix(NA, nrow = n, ncol = n)
  # for(i in 1:n){
  #   for(j in 1:n){
  #     K0[i,j] = ker.fun(X[i,], X[j,], kernel.list)
  #   }
  # }
  K0 = (K0 + t(K0))/2
  if(!is.null(cov.mat)){
    cov.mat.sqrt.inv = solve( sqrtm(cov.mat) )
    K0 = cov.mat.sqrt.inv %*% K0 %*% t(cov.mat.sqrt.inv)
  }
  eigen.decomp = eigen(K0)
  eigen.vector = eigen.decomp$vectors
  eigen.value = eigen.decomp$values + kernel.list$singular
  K = eigen.vector %*% diag( eigen.value ) %*% t( eigen.vector )
  return(list(K=K, eigen.vector = eigen.vector, eigen.value = eigen.value, K0 = K0, eigen.decomp = eigen.decomp, singular = kernel.list$singular ))
}

cov.structure <- function(Z, kernel.result, var.equal){
  H = length(unique(Z))
  n = length(Z)
  K = kernel.result$K
  Identity = diag(n)
  if(var.equal){
    G.all = array(0, dim = c(n, n, H+2))
    G.all[,,1] = K
    G.all[,,H+2] = Identity
    for(h in 1:H){
      G.all[Z==h, Z==h, h+1] = K[Z==h, Z==h]
    }
  }else{
    G.all = array(0, dim = c(n, n, 2*H+1))
    G.all[,,1] = K
    for(h in 1:H){
      G.all[Z==h, Z==h, h+1] = K[Z==h, Z==h]
      G.all[Z==h, Z==h, H+h+1] = Identity[Z==h, Z==h]
    }
  }
  return(G.all)
}


mle.H0 <- function(Y, kernel.result, threshold = 10^(-5)/2){
  K = kernel.result$K
  eigen.vector = kernel.result$eigen.vector
  eigen.value = kernel.result$eigen.value

  n = length(Y)
  U = t(eigen.vector) %*% Y

  ## initial value
  delta2 = as.vector( t(Y) %*% eigen.vector %*% diag(1/(eigen.value+1)) %*% t(eigen.vector) %*% Y / n )
  sigma02 = delta2
  cov.diag = delta2 * eigen.value + sigma02
  log.lik = sum( dnorm(U, mean = 0, sd = sqrt(cov.diag), log = TRUE) )
  # -2 * log.lik - n * log(2*pi)
  lik.diff = 1 + threshold

  while(lik.diff > threshold){
    log.lik.old = log.lik
    delta2.old = delta2
    sigma02.old = sigma02
    b = (1 - sigma02/cov.diag) * sigma02
    nu1 = sum(b)
    nu2 = sigma02^2 * sum( (U/cov.diag)^2 )
    theta1 = sum(b/eigen.value)
    theta2 = delta2^2* sum(eigen.value * (U/cov.diag)^2)
    delta2 = ( theta1 + theta2 )/n
    sigma02 = (nu1 + nu2)/n
    cov.diag = delta2 * eigen.value + sigma02
    log.lik = sum( dnorm(U, mean = 0, sd = sqrt(cov.diag), log = TRUE) )
    lik.diff = log.lik - log.lik.old
  }

  return(list(delta2=delta2, sigma02 = sigma02, log.lik = log.lik, K = K))
}

# New H0: Common function, but heterogeneous noise across groups
# H0_hetero: Y ~ N(0, delta2*K + Sigma_noise)
# where Sigma_noise = sigma1^2 * I^{(1)} + sigma2^2 * I^{(2)} + ... + sigmaH^2 * I^{(H)}
mle.H0.heterogeneous <- function(Y, kernel.result, Z, threshold = 10^(-5)/2){
  n <- length(Y)
  H <- length(unique(Z))
  K <- kernel.result$K
  Identity <- diag(n)
  
  # Construct G.all: [K, I^{(1)}, I^{(2)}, ..., I^{(H)}]
  G.all <- array(0, dim = c(n, n, H + 1))
  
  # First component: common kernel K
  G.all[,,1] <- K
  
  # H components: group-specific noise
  for(h in 1:H) {
    G.all[Z==h, Z==h, h+1] <- Identity[Z==h, Z==h]
  }
  
  # Initial tau values: use homogeneous H0 as starting point
  est.H0.homogeneous <- mle.H0(Y, kernel.result, threshold)
  tau.initial <- c(est.H0.homogeneous$delta2, 
                   rep(est.H0.homogeneous$sigma02, H))
  
  # Use mle.fish.em to estimate
  est <- mle.fish.em(Y = Y, G.all = G.all, tau = tau.initial, threshold = threshold)
  
  # Extract results
  delta2 <- est$tau[1]
  sigma2.groups <- est$tau[2:(H+1)]
  
  return(list(
    delta2 = delta2,
    sigma2.groups = sigma2.groups,
    log.lik = est$log.lik,
    lik.diff = est$lik.diff,
    K = K
  ))
}

mle.fish.em <- function(Y, G.all, tau, threshold = 10^(-5)/2){
  n = length(Y)
  Sigma = matrix(0, nrow = n, ncol = n)
  for(g in 1:dim(G.all)[3]){
    Sigma = Sigma + tau[g] * G.all[,,g]
  }
  
  # Use Cholesky decomposition for better performance
  Sigma.chol = chol(Sigma)
  Omega = chol2inv(Sigma.chol)
  log.lik = dmvnorm(Y, mean = rep(0, n), sigma = Sigma, log = TRUE)

  fir.der = rep(NA, dim(G.all)[3])
  sec.der = matrix(NA, nrow = dim(G.all)[3], ncol = dim(G.all)[3])
  count.fish = 1

  lik.diff = threshold + 1

  while(lik.diff > threshold){
    log.lik.old = log.lik
    Omega.G = array(dim = dim(G.all))
    for(g in 1:dim(G.all)[3]){
      Omega.G[,,g] = Omega %*% G.all[,,g]
    }
    for(g in 1:dim(G.all)[3]){
      fir.der[g] = -0.5*sum(diag( Omega.G[,,g])) + 0.5*sum( diag( Omega %*% Y %*% t(Y) %*% Omega.G[,,g]) )
      for(j in g:dim(G.all)[3]){
        sec.der[g,j] = -0.5*sum(diag(Omega.G[,,g]%*%Omega.G[,,j]))
        sec.der[j,g] = sec.der[g,j]
      }
    }
    if(count.fish < 100){
      sec.der = (1+1/2*log(log(count.fish)+1))*sec.der
    }else{
      sec.der = sec.der - count.fish^2/2 * diag(dim(G.all)[3])
    }
    tune = norm(sec.der,"2")
    delta.tau = solve.QP(Dmat = -1*sec.der/tune, dvec = fir.der/tune, Amat = diag(dim(G.all)[3]), bvec = -1*tau)
    delta.tau$solution
    tau = tau + delta.tau$solution

    Sigma = matrix(0, nrow = n, ncol = n)
    for(g in 1:dim(G.all)[3]){
      Sigma = Sigma + tau[g] * G.all[,,g]
    }
    Sigma.chol = chol(Sigma)
    Omega = chol2inv(Sigma.chol)
    log.lik = dmvnorm(Y, mean = rep(0, n), sigma = Sigma, log = TRUE)

    lik.diff = abs(log.lik-log.lik.old)
    count.fish = count.fish + 1
  }


  lik.diff = threshold + 1
  mu = array(dim = c(n, 1, dim(G.all)[3]))
  Lambda = array(dim = dim(G.all))
  count.em = 1

  while(lik.diff > threshold & count.em < 3000){
    log.lik.old = log.lik
    for(g in 1:dim(G.all)[3]){
      A = tau[g] * G.all[,,g] %*% Omega
      mu[,,g] = A %*% Y
      Lambda[,,g] = tau[g] * G.all[,,g] - tau[g] * A %*% G.all[,,g]
      a = tau[g] * t(mu[,,g]) %*% Omega %*% Y
      b = sum( diag( tau[g] * diag(n) - tau[g]^2 * Omega %*% G.all[,,g] ) )
      tau[g] = (a+b)/n
    }
    Sigma = matrix(0, nrow = n, ncol = n)
    for(g in 1:dim(G.all)[3]){
      Sigma = Sigma + tau[g] * G.all[,,g]
    }
    Sigma.chol = chol(Sigma)
    Omega = chol2inv(Sigma.chol)
    log.lik = dmvnorm(Y, mean = rep(0, n), sigma = Sigma, log = TRUE)
    lik.diff = log.lik - log.lik.old
    count.em = count.em + 1
  }

  return(list(tau = tau, log.lik = log.lik, lik.diff = lik.diff))

}

mle.H1 <- function(Y, cov.alter, var.equal, est.H0, threshold = 10^(-5)/2){
  G.all = cov.alter
  if(var.equal){
    H = dim(G.all)[3] - 2
    tau = c(rep(est.H0$delta2, H+1), est.H0$sigma02)
  }else{
    H = (dim(G.all)[3] - 1)/2
    tau = c(rep(est.H0$delta2, H+1), rep(est.H0$sigma02, H) )
  }
  est.H1 <- mle.fish.em(Y = Y, G.all = G.all, tau = tau, threshold = threshold)
  # print(paste( "convergence: ", est.H1$lik.diff <= threshold, sep = "") )
  return(est.H1)
}

# -2 * est.H0$log.lik - length(Y) * log(2*pi)

test.stat <- function(Y, kernel.result, threshold,
                      # lik ratio tilde{H}_1 vs tilde{H}_0
                      GP.alter, cov.alter, var.equal,
                      # lik ratio tilde{H}_ps vs tilde{H}_0
                      GP.pseudo.alter, kernel.result.group,
                      # lik ratio tilde{H}_ps vs tilde{H}_0_heterogeneous
                      GP.pseudo.alter.hetero,
                      # lik ratio H_1 vs H_0
                      fix.alter, Phi, Z){
  if(GP.alter){
    ## mle under tilde{H}_0 and tilde{H}_1
    est.H0 = mle.H0(Y, kernel.result, threshold)
    est.H1 = mle.H1(Y, cov.alter, var.equal, est.H0, threshold)
    stat = est.H1$log.lik - est.H0$log.lik
    converge = (est.H1$lik.diff <= threshold)
    return(list(stat = stat, converge = converge, est.H0=est.H0))
  }
  if(GP.pseudo.alter){
    ## mle under tilde{H}_0 and tilde{H}_ps
    est.H0 = mle.H0(Y, kernel.result, threshold)
    H = length(kernel.result.group)
    log.lik.group = rep(NA, H)
    for(h in 1:H){
      log.lik.group[h] = mle.H0(Y[Z==h], kernel.result.group[[h]], threshold)$log.lik
    }
    est.H1.pseudo = list(log.lik = sum(log.lik.group))
    stat = est.H1.pseudo$log.lik - est.H0$log.lik
    return(list(stat = stat))
  }
  if(GP.pseudo.alter.hetero){
    ## mle under tilde{H}_0_heterogeneous and tilde{H}_ps
    ## New H0: same function, but heterogeneous noise across groups
    est.H0.hetero = mle.H0.heterogeneous(Y, kernel.result, Z, threshold)
    H = length(kernel.result.group)
    log.lik.group = rep(NA, H)
    for(h in 1:H){
      log.lik.group[h] = mle.H0(Y[Z==h], kernel.result.group[[h]], threshold)$log.lik
    }
    est.H1.pseudo = list(log.lik = sum(log.lik.group))
    stat = est.H1.pseudo$log.lik - est.H0.hetero$log.lik
    return(list(stat = stat, est.H0 = est.H0.hetero))
  }
  if(fix.alter){
    fit0 = lm(Y~Phi)
    fit1 = lm(Y~Phi*factor(Z))
    lrt = lrtest(fit0, fit1)
    stat.lrt = lrt$Chisq[2]/2
    Ftest = anova(fit1, fit0)
    stat.F = Ftest$F[2]
    pval.lrt = lrt$`Pr(>Chisq)`[2]
    pval.F = Ftest$`Pr(>F)`[2]
    return(list(stat.lrt = stat.lrt, stat.F=stat.F, pval.lrt = pval.lrt, pval.F = pval.F))
  }
}

test.stat.GP.alter <- function(Y, kernel.result, threshold, cov.alter, var.equal){
  est.H0 = mle.H0(Y, kernel.result, threshold)
  est.H1 = mle.H1(Y, cov.alter, var.equal, est.H0, threshold)
  stat = est.H1$log.lik - est.H0$log.lik
  converge = (est.H1$lik.diff <= threshold)
  return(list(stat = stat, converge = converge, est.H0=est.H0))
}

test.stat.GP.pseudo.alter <- function(Y, kernel.result, threshold, kernel.result.group, Z){
  ## mle under tilde{H}_0 and tilde{H}_ps
  est.H0 = mle.H0(Y, kernel.result, threshold)
  H = length(kernel.result.group)
  log.lik.group = rep(NA, H)
  for(h in 1:H){
    log.lik.group[h] = mle.H0(Y[Z==h], kernel.result.group[[h]], threshold)$log.lik
  }
  est.H1.pseudo = list(log.lik = sum(log.lik.group))
  stat = est.H1.pseudo$log.lik - est.H0$log.lik
  return(list(stat = stat, est.H0 = est.H0))
}

test.stat.GP.pseudo.alter.hetero <- function(Y, kernel.result, threshold, kernel.result.group, Z){
  ## mle under tilde{H}_0_heterogeneous and tilde{H}_ps
  ## New H0: same function, but heterogeneous noise across groups
  est.H0.hetero = mle.H0.heterogeneous(Y, kernel.result, Z, threshold)
  H = length(kernel.result.group)
  log.lik.group = rep(NA, H)
  for(h in 1:H){
    log.lik.group[h] = mle.H0(Y[Z==h], kernel.result.group[[h]], threshold)$log.lik
  }
  est.H1.pseudo = list(log.lik = sum(log.lik.group))
  stat = est.H1.pseudo$log.lik - est.H0.hetero$log.lik
  return(list(stat = stat, est.H0 = est.H0.hetero))
}

test.stat.fix.alter <- function(Y, Phi, Z){
  fit0 = lm(Y~Phi)
  fit1 = lm(Y~Phi*factor(Z))
  lrt = lrtest(fit0, fit1)
  stat.lrt = lrt$Chisq[2]/2
  Ftest = anova(fit1, fit0)
  stat.F = Ftest$F[2]
  pval.lrt = lrt$`Pr(>Chisq)`[2]
  pval.F = Ftest$`Pr(>F)`[2]
  return(list(stat.lrt = stat.lrt, stat.F=stat.F, pval.lrt = pval.lrt, pval.F = pval.F))
}

GP.reg <- function(Y, kernel.result, threshold){
  est.H0 = mle.H0(Y, kernel.result, threshold)
  lambda.K = kernel.result$eigen.value - kernel.result$singular
  f.hat = kernel.result$eigen.vector %*% diag(lambda.K/(lambda.K+est.H0$sigma02/est.H0$delta2)) %*% t(kernel.result$eigen.vector) %*% Y
  sigma02.hat = mean((Y-f.hat)^2)
  return(list(f.hat = f.hat, sigma02.hat = sigma02.hat))
}

test.stat.GP.reg <- function(Y, kernel.result, threshold, kernel.result.group){
  n = length(Y)
  rss.full <- GP.reg(Y, kernel.result, threshold)$sigma02.hat
  H = length(kernel.result.group)
  rss.group = rep(NA, H)
  n.group = rep(NA, H)
  for(h in 1:H){
    n.group[h] = length(Y[Z==h])
    rss.group[h] = GP.reg(Y[Z==h], kernel.result.group[[h]], threshold)$sigma02.hat
  }
  log.lrt = ( n*log(rss.full) - sum(n.group*log(rss.group)) )/2
  stat = log.lrt
  return(list(stat = stat))
}


GP.band.MLE <- function( X, Y, singular, cov.mat, kernel.name, alpha.input,
                         initial.grid = TRUE, grid, print.grid = 1){
  target <- function(alpha){
    kernel.result <- kernel.matrix(X, kernel.list = list(name=kernel.name, alpha = alpha, singular = singular ), cov.mat = cov.mat )
    est = mle.H0(Y, kernel.result, threshold = 10^(-5)/2)
    return( est$log.lik )
  }
  if(initial.grid){
    # print("begin finding the initial value for bandwidth...")
    grid.each = grid
    grid.all = matrix(NA, nrow = length(grid.each)^(ncol(X)), ncol = ncol(X) )
    for(k in 1:ncol(X)){
      grid.all[, k] = rep( grid.each, each = length(grid.each)^(ncol(X)-k), times = length(grid.each)^(k-1) )
    }
    log.lik.all = rep(NA, length(grid.all))
    for(i in 1:nrow(grid.all)){
      log.lik.all[i] = target( grid.all[i, ] )
      if(i %% print.grid == 0){
        print( paste("try the bandwidth value at grid", i) )
      }
    }
    alpha.ini = grid.all[which.max(log.lik.all), ]
  }else{
    alpha.ini = alpha.input
  }
  # print("get the initial bandwidth value...")
  alpha.opt.list = optim( par = alpha.ini, fn = target, method= "L-BFGS-B", lower = 0, upper = Inf, control = list( fnscale = -1 ) )
  alpha.opt = alpha.opt.list$par
  # print("get the fitted bandwidth...")
  kernel.list = list( name="Gaussian", alpha = alpha.opt, singular = singular )
  return(kernel.list)
}


######### partial permutation test main function #############
## Y and Z are vectors, and X is a matrix without intercept ##

#' Conducting Partial Permutation Test
#'
#' Get the p-value from the partial permutation test.
#'
#' @import mvtnorm
#' @import psych
#' @import quadprog
#' @import Matrix
#' @import lmtest
#'
#' @param Y An \eqn{n} dimensional vector of outcomes
#' @param X An \eqn{n} by \eqn{d} matrix of covariates.
#' @param Z An \eqn{n} dimensional vector of group indicators, taking values 1, 2, ..., \eqn{H}.
#' @param cov.mat An \eqn{n} by \eqn{n} (estimated) error covariance matrix up to a postive scale. It is set to be NULL if it equals identity matrix.
#' @param kernel.list A list object for the choice of kernel matrix. For example, it equals list(name = "Gaussian", alpha = 0.5, singular = 10^(-5)) for Gaussian kernel, where alpha is the parameter for bandwidth and singular is used to avoid potential singularity of kernel matrix; it equals list(name = "polynomial", p = 2, singular = 0) for polynomial kernel, where p is for the degree of freedom.
#' @param perm.max A numerical object for the number of permutations.
#' @param perm.choice A character object specifying how to choose the permutation size. It can take value "GP" and "fix".
#' @param sig.level A numerical object for the significance level.
#' @param alternative A character object specifying the choice of test statistics. It can take values "GP.equal", "GP.unequal" and "GP.pseudo" for likelihood ratio statistics with different alternative Gaussian process models, and "GP.reg" for comparing mean squared errors from pooled and group-specific kernel regression.
#' @param bandwidth.choice A character object specifying how to choose the bandwidth parameter for the kernel function. It can take value "MLE.all" and "fix".
#' @param initial.grid A logical object indicating whether to perform grid search for finding the bandwidth parameter that maximizes the marginal likelihood.
#' @param grid A vector object indicating the grid to search over when initial.grid is TRUE.
#' @export

partial.perm.test <- function(Y, X, Z, cov.mat = NULL, kernel.list = list(name = "Gaussian", alpha = 0.5, singular = 10^(-5)), perm.max = 500, threshold = 10^(-5), perm.choice = "GP", sig.level = 0.05, alternative = "GP.pseudo", print.opt = perm.max+1, bandwidth.choice = "MLE.all", initial.grid = FALSE, grid = seq(from = 0.1, to = 5, by = 1), print.grid = length(grid)^ncol(X)+1, n.cores = 1 ){
  
  result.list = list()
  
  time.start = Sys.time()
  result.list$time.start = time.start
  
  ## standardize
  X = as.matrix(X)
  X.ori = X
  # if(alternative != "Parallel"){
  X = t((t(X) - colMeans(X))/apply(X, 2, sd))
  # }
  Y = as.vector(Y)
  Y = (Y-mean(Y))/sd(Y)
  n = length(Y)

  ## adjust Y
  if(!is.null(cov.mat)){
    Y = solve( sqrtm(cov.mat) ) %*% Y
    Y.group.cov.sd = rep(NA, n)
    cov.mat.group.sqrt.inv = list()
    for(h in 1:length(unique(Z))){
      cov.mat.group.sqrt.inv[[h]] = solve( sqrtm(cov.mat[Z==h, Z==h]) )
      Y.group.cov.sd[Z==h] = cov.mat.group.sqrt.inv[[h]] %*% Y[Z==h]
    }
  }

  singular = kernel.list$singular

  ## choice of bandwitch
  # if( bandwidth.choice == "MLE" ){
  #   alpha.input = kernel.list$alpha
  #   kernel.list = GP.band.MLE( X, Y, singular, cov.mat, kernel.name = kernel.list$name, alpha.input, initial.grid, grid, print.grid)
  #   alpha.opt = kernel.list$alpha
  #   kernel.list = list( name=kernel.list$name, alpha = alpha.opt, singular = singular )
  # }
  #
  # if( bandwidth.choice == "MLE.each" ){
  #   kernel.list.all = list()
  #   alpha.input = kernel.list$alpha
  #   alpha.opt.all = matrix(nrow = length(unique(Z)), ncol = ncol(X))
  #   for(h in 1:length(unique(Z))){
  #     kernel.list.all[[h]] =   GP.band.MLE( X[Z==h, ,drop=FALSE], Y[Z==h], singular, cov.mat, kernel.name = kernel.list$name, alpha.input, initial.grid, grid, print.grid)
  #     alpha.opt.all[h, ] = kernel.list.all[[h]]$alpha
  #   }
  #   alpha.opt = apply(alpha.opt.all, 2, max)
  #   kernel.list = list( name=kernel.list$name, alpha = alpha.opt, singular = singular )
  # }

  if( bandwidth.choice == "MLE.all" ){
    alpha.input = kernel.list$alpha

    kernel.list = GP.band.MLE( X, Y, singular, cov.mat, kernel.name = kernel.list$name, alpha.input, initial.grid, grid, print.grid)
    alpha.opt.0 = kernel.list$alpha

    kernel.list.all = list()
    alpha.opt.all = matrix(nrow = length(unique(Z)), ncol = ncol(X))

    if(is.null(cov.mat)){
      for(h in 1:length(unique(Z))){
        kernel.list.all[[h]] =   GP.band.MLE( X[Z==h, ,drop=FALSE], Y[Z==h], singular, cov.mat, kernel.name = kernel.list$name, alpha.input, initial.grid, grid, print.grid)
        alpha.opt.all[h, ] = kernel.list.all[[h]]$alpha
      }
    }else{
      for(h in 1:length(unique(Z))){
        kernel.list.all[[h]] =  GP.band.MLE( X[Z==h, ,drop=FALSE], Y.group.cov.sd[Z==h], singular, cov.mat[Z==h, Z==h], kernel.name = kernel.list$name, alpha.input, initial.grid, grid, print.grid)
        alpha.opt.all[h, ] = kernel.list.all[[h]]$alpha
      }
    }

    alpha.opt.1 = apply(alpha.opt.all, 2, max)
    alpha.opt = apply(rbind(alpha.opt.0, alpha.opt.1), 2, min)
    print("get the fitted bandwidth...")
    kernel.list = list( name=kernel.list$name, alpha = alpha.opt, singular = singular )
  }

  if( bandwidth.choice == "fix" ){
    kernel.list = kernel.list
  }

  ## kernal matrix
  print(kernel.list)
  kernel.result <- kernel.matrix(X, kernel.list = kernel.list, cov.mat = cov.mat )
  result.list$kernel.result = kernel.result
  result.list$alpha = kernel.list$alpha
  
  time.kernel = Sys.time()
  result.list$time.kernel = time.kernel
  
  ### different choices of test statistic
  if( alternative == "GP.equal" ){
    ## covariance structure under H1
    var.equal = TRUE
    cov.alter <- cov.structure(Z, kernel.result, var.equal)
    result.obs <- test.stat.GP.alter(Y, kernel.result, threshold, cov.alter, var.equal)
    test.obs = result.obs$stat
    converg.obs = result.obs$converge
  }

  if(alternative == "GP.unequal"){
    ## covariance structure under H1
    var.equal = FALSE
    cov.alter <- cov.structure(Z, kernel.result, var.equal)
    result.obs <- test.stat.GP.alter(Y, kernel.result, threshold, cov.alter, var.equal)
    test.obs = result.obs$stat
    converg.obs = result.obs$converge
  }

  if(alternative == "GP.pseudo"){
    kernel.result.group = list()
    H = length(unique(Z))
    for(h in 1:H){
      # kernel.result.group[[h]] = kernel.matrix(X[Z==h,,drop=FALSE], kernel.list = kernel.list )
      kernel.result.group[[h]] = list()
      kernel.result.group[[h]]$K = ( kernel.result$K[Z==h, Z==h] + t(kernel.result$K[Z==h, Z==h]) )/2
      temp = eigen(kernel.result.group[[h]]$K)
      kernel.result.group[[h]]$eigen.vector = temp$vectors
      kernel.result.group[[h]]$eigen.value = temp$values
    }
    result.obs <- test.stat.GP.pseudo.alter(Y, kernel.result, threshold, kernel.result.group, Z)
    test.obs = result.obs$stat
  }

  if(alternative == "GP.pseudo.hetero"){
    kernel.result.group = list()
    H = length(unique(Z))
    for(h in 1:H){
      # kernel.result.group[[h]] = kernel.matrix(X[Z==h,,drop=FALSE], kernel.list = kernel.list )
      kernel.result.group[[h]] = list()
      kernel.result.group[[h]]$K = ( kernel.result$K[Z==h, Z==h] + t(kernel.result$K[Z==h, Z==h]) )/2
      temp = eigen(kernel.result.group[[h]]$K)
      kernel.result.group[[h]]$eigen.vector = temp$vectors
      kernel.result.group[[h]]$eigen.value = temp$values
    }
    result.obs <- test.stat.GP.pseudo.alter.hetero(Y, kernel.result, threshold, kernel.result.group, Z)
    test.obs = result.obs$stat
  }

  if(alternative == "fix"){
    Phi = kernel.result$eigen.vector[, 1: as.vector(  rankMatrix( kernel.result$K0 )  )]
    result.obs <- test.stat.fix.alter(Y, Phi, Z)
    test.obs = result.obs$stat.lrt
    test2.obs = result.obs$stat.F
    pval.lrt = result.obs$pval.lrt
    pval.F = result.obs$pval.F
  }

  if(alternative == "GP.reg"){
    kernel.result.group = list()
    H = length(unique(Z))
    for(h in 1:H){
      # kernel.result.group[[h]] = kernel.matrix(X[Z==h,,drop=FALSE], kernel.list = kernel.list )
      kernel.result.group[[h]] = list()
      kernel.result.group[[h]]$K = ( kernel.result$K[Z==h, Z==h] + t(kernel.result$K[Z==h, Z==h]) )/2
      temp = eigen(kernel.result.group[[h]]$K)
      kernel.result.group[[h]]$eigen.vector = temp$vectors
      kernel.result.group[[h]]$eigen.value = temp$values
      kernel.result.group[[h]]$singular = singular
    }
    result.obs <- test.stat.GP.reg(Y, kernel.result, threshold, kernel.result.group)
    test.obs = result.obs$stat
  }

  if(alternative == "Parallel"){
    parallel.obs = parallelism(y = Y, z = as.vector(X.ori), g = as.factor(Z))
    test.obs = -1 * parallel.obs$pvalue
    test2.obs = parallel.obs$score
  }

  print("get the observed test statistic...")

  ## choose permutation size bn
  if(perm.choice == "GP"){
    est.H0 = mle.H0(Y, kernel.result, threshold)
    p0 = 10^(-4) * sig.level
    lambda.K = kernel.result$eigen.value - kernel.result$singular
    v0 = rep(NA, n)
    for(bn in 1:n){
      omega = est.H0$delta2 / est.H0$sigma02 * lambda.K[n-bn+1]
      # v0[bn] = 0.5 * exp( (omega+1)*omega*qchisq(1-p0, bn) ) - 0.5
      v0[bn] = 0.5 * exp( 0.5*omega*qchisq(1-p0, bn) ) - 0.5
    }
    bn = sum(v0+p0 <= 10^(-3)*sig.level)
  }

  if(perm.choice == "fix"){
    # if(!GP.alter){
    #   est.H0 = mle.H0(Y, kernel.result, threshold)
    # }
    # p0 = 10^(-4) * sig.level
    # lambda.K = kernel.result$eigen.value - kernel.result$singular
    # f.hat = kernel.result$eigen.vector %*% diag(lambda.K/(lambda.K+est.H0$sigma02/est.H0$delta2)) %*% t(kernel.result$eigen.vector) %*% Y
    # sigma02.hat = mean((Y-f.hat)^2)

    p0 = 10^(-4) * sig.level
    fit.GP.reg = GP.reg(Y, kernel.result, threshold)
    f.hat = fit.GP.reg$f.hat
    sigma02.hat = fit.GP.reg$sigma02.hat

    v0 = rep(NA, n)
    for(bn in 1:n){
      proj = t( kernel.result$eigen.vector[, (n-bn+1):n] ) %*% f.hat
      omega = sum(proj^2)/sigma02.hat
      # v0[bn] = 0.5 * exp( 2*sqrt(2)*( sqrt(omega*qchisq(1-p0, bn)) + omega ) ) - 0.5
      v0[bn] = 0.5 * exp( 2*sqrt(2)* sqrt(omega) * sqrt(qchisq(1-p0, bn)+ omega) ) - 0.5
    }
    bn = sum(v0+p0 <= 10^(-3)*sig.level)
  }

  if(perm.choice == "rank"){
    bn = n - as.vector( rankMatrix(kernel.result$K0) )
  }
  
  result.list$bn = bn

  ## permutation test
  if(bn <= 1){
    # return(list(p.val = 1, bn = bn, alpha.opt = alpha.opt, alpha.ini = alpha.ini))
    # result.list = list(p.val = 1, bn = bn)
    # result.list$alpha = kernel.list$alpha
    result.list$p.val = 1
    return(result.list)
  }

  W = t( kernel.result$eigen.vector ) %*% Y

  ## iterate the permutation with parallel processing
  cat(sprintf("Using %d cores for parallel processing...\n", n.cores))
  cat("Starting permutation test...\n")
  
  perm_start_time <- Sys.time()
  
  perm_worker <- function(iter, W, n, bn, kernel.result, threshold, 
                          alternative, cov.alter, var.equal, 
                          kernel.result.group, Phi, Z, X.ori, print_opt) {
    W.perm = W[ c( c(1:(n-bn)), sample( c((n-bn+1):n) ) ) ]
    Y.perm = as.vector( kernel.result$eigen.vector %*% W.perm )

    result = list(iter = iter, test = NA, test2 = NA, converge = NA)

    if( alternative == "GP.equal" ){
      ## covariance structure under H1
      var.equal = TRUE
      result.perm <- test.stat.GP.alter(Y.perm, kernel.result, threshold, cov.alter, var.equal)
      result$test = result.perm$stat
      result$converge = result.perm$converge
    }

    if( alternative == "GP.unequal" ){
      ## covariance structure under H1
      var.equal = FALSE
      result.perm <- test.stat.GP.alter(Y.perm, kernel.result, threshold, cov.alter, var.equal)
      result$test = result.perm$stat
      result$converge = result.perm$converge
    }

    if(alternative == "GP.pseudo"){
      result.perm <- test.stat.GP.pseudo.alter(Y.perm, kernel.result, threshold, kernel.result.group, Z)
      result$test = result.perm$stat
    }

    if(alternative == "GP.pseudo.hetero"){
      result.perm <- test.stat.GP.pseudo.alter.hetero(Y.perm, kernel.result, threshold, kernel.result.group, Z)
      result$test = result.perm$stat
    }

    if(alternative == "fix"){
      result.perm <- test.stat.fix.alter(Y.perm, Phi, Z)
      result$test = result.perm$stat.lrt
      result$test2 = result.perm$stat.F
    }

    if(alternative == "GP.reg"){
      result.perm <- test.stat.GP.reg(Y.perm, kernel.result, threshold, kernel.result.group)
      result$test = result.perm$stat
    }

    if(alternative == "Parallel"){
      parallel.perm = parallelism(y = Y.perm, z = as.vector(X.ori), g = as.factor(Z))
      result$test = -1 * parallel.perm$pvalue
      result$test2 = parallel.perm$score
    }

    if(iter %% print_opt == 0){
      print( paste0("get the test statistic for the ", iter, "th permutation...") )
    }

    return(result)
  }
  
  perm_results <- mclapply(1:perm.max, 
                            FUN = perm_worker,
                            W = W, n = n, bn = bn, 
                            kernel.result = kernel.result,
                            threshold = threshold,
                            alternative = alternative,
                            cov.alter = if(exists("cov.alter")) cov.alter else NULL,
                            var.equal = if(exists("var.equal")) var.equal else NULL,
                            kernel.result.group = if(exists("kernel.result.group")) kernel.result.group else NULL,
                            Phi = if(exists("Phi")) Phi else NULL,
                            Z = Z,
                            X.ori = X.ori,
                            print_opt = print.opt,
                            mc.cores = n.cores,
                            mc.set.seed = TRUE)
  
  # Extract results
  test.perm = sapply(perm_results, function(x) x$test)
  
  perm_elapsed <- as.numeric(difftime(Sys.time(), perm_start_time, units = "secs"))
  cat(sprintf("Permutation test completed: %d permutations in %.2f seconds (avg: %.2f sec/perm)\n",
              perm.max, perm_elapsed, perm_elapsed / perm.max))
  
  if(alternative == "GP.equal" | alternative == "GP.unequal"){
    converg.perm = sapply(perm_results, function(x) x$converge)
  }
  
  if(alternative == "fix" | alternative == "Parallel"){
    test2.perm = sapply(perm_results, function(x) x$test2)
  }
  
  print("get the permuted test statistics...")

  p.val = mean(test.perm >= test.obs) + 10^(-3)*sig.level
  
  result.list$test.obs = test.obs
  result.list$test.perm = test.perm
  result.list$p.val = p.val

  if(alternative == "GP.equal" | alternative == "GP.unequal"){
    if( prod( converg.perm ) * converg.obs == 1 ){
      converg.all = TRUE
    }else{
      converg.all = FALSE
    }
    result.list$converg.obs = converg.obs
    result.list$converg.perm = converg.perm
    result.list$converg.all = converg.all
    # result.list = list(p.val = p.val, bn = bn, converg.obs = converg.obs, converg.perm = converg.perm, converg.all = converg.all, test.perm = test.perm, test.obs = test.obs)
  }

  # if(alternative == "GP.pseudo" | alternative == "GP.reg"){
  #   result.list = list(p.val = p.val, bn = bn, test.perm = test.perm, test.obs = test.obs)
  # }

  if(alternative == "fix" ){
    p.val2 = mean(test2.perm >= test2.obs) + 10^(-3)*sig.level
    # result.list = list(p.val = p.val, bn = bn, p.val2=p.val2, pval.lrt = pval.lrt, pval.F = pval.F, test.perm = test.perm, test.obs = test.obs)
    result.list$p.val2 = p.val2
    result.list$pval.lrt = pval.lrt
    result.list$pval.F = pval.F
    result.list$test2.perm = test2.perm
    result.list$test2.obs = test2.obs
  }

  if(alternative == "Parallel"){
    p.val2 = mean(test2.perm >= test2.obs) + 10^(-3)*sig.level
    # result.list = list(p.val = p.val, bn = bn, p.val2=p.val2, test.perm = test.perm, test.obs = test.obs, test2.perm = test2.perm, test2.obs = test2.obs)
    result.list$p.val2 = p.val2
    result.list$test2.perm = test2.perm
    result.list$test2.obs = test2.obs
  }

  # result.list$alpha = kernel.list$alpha
  # result.list$X = X
  # result.list$Y = Y
  
  # result.list$kernel.result = kernel.result
  
  if(perm.choice == "fix"){
    result.list$f.hat = f.hat
  }
  
  time.end = Sys.time()
  
  # result.list$time.start = time.start
  # result.list$time.kernel = time.kernel
  result.list$time.end = time.end

  return(result.list)
}



########### functions for simulation under null ##############
balance.group.cov <- function(n, dimension, case){
  if(case == 1){
    a = c(0.5, 0.5)
    p = c(0.5, 0.5)
  }

  if(case == 2){
    a = c(0.5, 0.5)
    p = c(0.2, 0.8)
  }
  if(case == 3){
    a = c(0.8, 0.2)
    p = c(0.5, 0.5)
  }

  if(case == 4){
    a = c(0.8, 0.2)
    p = c(0.2, 0.8)
  }
  if(case == 5){
    a = c(1, 0)
    p = c(0.5, 0.5)
  }

  Z = rep(1, n)
  Z = Z + (runif(n) <= p[2])

  X = rep(NA, n)
  for(i in 1:n){
    if(runif(1) <= a[Z[i]]){
      X[i] = -1 *  runif(1)
    }else{
      X[i] = runif(1)
    }
  }

  if(dimension == 2){
    X2 = rep(NA, n)
    for(i in 1:n){
      if(runif(1) <= a[Z[i]]){
        X2[i] = -1 *  runif(1)
      }else{
        X2[i] = runif(1)
      }
    }
    X = cbind(X,  X2)
  }
  X = as.matrix(X)
  return(list(X = X, Z = Z, a=a, p=p))
}

fix.fun <- function(x,  dimension, f.choice){
  if(dimension == 1){
    if(f.choice == 1){
      f = x
    }
    if(f.choice == 2){
      f = 2 * x^2 - 1
    }
    if(f.choice == 3){
      f = ( x^3 - 1/4*x ) * 4/3
    }
    if(f.choice == 4){
      f = 1/(1+x^2) * 4 - 3
    }
    if(f.choice == 5){
      f = sin(4*x)
    }
    if(f.choice == 6){
      f = sin(6*x)
    }
    if(f.choice == 7){
      x.prime = 3*x - floor(3*x)
      f = 2 * apply( cbind( abs(x.prime), abs(x.prime - 1) ), 1, min ) * (floor(3*x)%%2 + 1) - 1
    }
  }
  if(dimension == 2){
    if(f.choice == 1){
      f = ( x[1] + x[2] )/2
    }
    if(f.choice == 2){
      f = x[1] * x[2]
    }
    if(f.choice == 3){
      f = 2 * (x[1] + x[2])^3/15 - (x[1] + x[2])/30
    }
    if(f.choice == 4){
      f = 3/(1 + x[1]^2 + x[2]^2) - 2
    }
    if(f.choice == 5){
      f = ( sin(6*x[1]) + x[2] )/2
    }
    if(f.choice == 6){
      f = sin( 6 * (x[1]+x[2]) )
    }
    if(f.choice == 7){
      x.prime = rep(NA, 2)
      f.val = rep(NA, 2)
      for(k in 1:2){
        x.prime[k] = 3*x[k] - floor(3*x[k])
        f.val[k] = 2 * apply( cbind( abs(x.prime[k]), abs(x.prime[k] - 1) ), 1, min ) * (floor(3*x[k])%%2 + 1) - 1
      }
      f = f.val[1] * f.val[2]
    }
  }
  return(f)
}

data.generate <- function(n, dimension, case, f.choice, var.residual){
  cov.group <- balance.group.cov(n, dimension, case)
  X = cov.group$X
  Z = cov.group$Z
  Y = rep(NA, n)
  for(i in 1:n){
    Y[i] = fix.fun(X[i,],  dimension, f.choice) + sqrt(var.residual) * rnorm(1)
  }
  return(list(X=X, Y=Y, Z=Z))
}


####### functions for simulation under alternative ###########
fix.fun.alter <- function(x, z, dimension, f.choice, noise.level){
  if(dimension == 1){
    if(f.choice == 1){
      noise.vec = exp(seq(from = 1, to = 6, length.out = 10))
      if(z==1){
        f = 1 + x
      }
      if(z==2){
        f = 2 + 3*x
      }
    }
    if(f.choice == 2){
      noise.vec = exp(seq(from = -5, to = 0, length.out = 10))
      if(z==1){
        f = 1/3 + x/2
      }
      if(z==2){
        f = (x+1)^2/4
      }
    }
    if(f.choice == 3){
      noise.vec = exp(seq(from = -5, to = 0, length.out = 10))
      if(z==1){
        # f = x^3
        f = 1/3 + x/2
      }
      if(z==2){
        # f = x*0.6
        f = 1/5 + x/2 - x^4 + x^2
      }
    }
  }
  if(dimension == 2){
    if(f.choice == 1){
      noise.vec = exp(seq(from = 1, to = 5, length.out = 10))
      if(z==1){
        f = 1 + x[1] + x[2]
      }
      if(z==2){
        f = 2 + 3*x[1] + x[2]
      }
    }
    if(f.choice == 2){
      noise.vec = exp(seq(from = -5, to = -1, length.out = 10))
      if(z==1){
        f = 1/3 + x[1]/2 + x[2]/2
      }
      if(z==2){
        f = (x[1]+1)^2/4 + (x[2]+1)^2/4 - 1/3
      }
    }
    if(f.choice == 3){
      noise.vec = exp(seq(from = -2, to = 1, length.out = 10))
      if(z==1){
        # f = x[1]^3 + x[2]^3
        f = 1/3 + x[1]/2 + x[2]/2
      }
      if(z==2){
        # f = 0.6 * ( x[1] + x[2] )
        f = 1/3 + x[1]/2 + x[2]/2 + sin(pi*x[1]) * sin(pi*x[2])
      }
    }
  }
  f = f + sqrt(noise.vec[noise.level]) * rnorm(length(z))
  return(f)
}

data.generate.alter <- function(n, dimension, f.choice, noise.level){
  cov.group <- balance.group.cov(n, dimension, case=1)
  X = cov.group$X
  Z = cov.group$Z
  Y = rep(NA, n)
  for(i in 1:n){
    Y[i] = fix.fun.alter(x = X[i,], z = Z[i], dimension, f.choice, noise.level)
  }
  return(list(X=X, Y=Y, Z=Z))
}

####### functions for usual permutation with switching ###########
test.switch.balance.cov <- function(Y, X, Z, cov.set, kernel.list = list(name = "Gaussian", alpha = 0.5, singular = 10^(-5)), perm.max = 500, threshold = 10^(-5), alternative = "GP.pseudo", print.opt = perm.max+1, bandwidth.choice = "MLE", initial.grid = TRUE, grid = seq(from = 0.1, to = 5, by = 0.1), print.grid = length(grid)+1){
  ## standardize covariates and outcomes
  X = t((t(X) - colMeans(X))/apply(X, 2, sd))
  Y = (Y-mean(Y))/sd(Y)
  n = length(Y)
  m = length( unique(cov.set) )

  ## choice of bandwitch
  if(bandwidth.choice == "MLE"){
    if(kernel.list$name == "Gaussian"){
      target <- function(alpha){
        kernel.result <- kernel.matrix(X, kernel.list = list(name="Gaussian", alpha = alpha, singular = kernel.list$singular ) )
        est = mle.H0(Y, kernel.result, threshold = 10^(-5)/2)
        return( est$log.lik )
      }
      if(initial.grid){
        # print("begin finding the initial value for bandwidth...")
        grid.each = grid
        grid.all = matrix(NA, nrow = length(grid.each)^(ncol(X)), ncol = ncol(X) )
        for(k in 1:ncol(X)){
          grid.all[, k] = rep( grid.each, each = length(grid.each)^(ncol(X)-k), times = length(grid.each)^(k-1) )
        }
        log.lik.all = rep(NA, length(grid.all))
        for(i in 1:nrow(grid.all)){
          log.lik.all[i] = target( grid.all[i, ] )
          if(i %% print.grid == 0){
            print( paste("try the bandwidth value at grid", i) )
          }
        }
        alpha.ini = grid.all[which.max(log.lik.all), ]
      }else{
        if(length(kernel.list$alpha) == 1){
          alpha.ini = rep(kernel.list$alpha, ncol(X))
        }else{
          alpha.ini = kernel.list$alpha
        }
      }
      print("get the initial bandwidth value...")
      alpha.opt.list = optim( par = alpha.ini, fn = target, method= "L-BFGS-B", lower = 0, upper = Inf, control = list( fnscale = -1 ) )
      alpha.opt = alpha.opt.list$par
      print("get the fitted bandwidth...")
      kernel.list = list( name="Gaussian", alpha = alpha.opt, singular = kernel.list$singular )
    }
    # print(kernel.list)
  }

  ## kernal matrix
  kernel.result <- kernel.matrix(X, kernel.list = kernel.list )

  ### different choices of test statistic
  if(alternative == "GP.pseudo"){
    GP.pseudo.alter = TRUE
    GP.alter = FALSE
    fix.alter = FALSE
  }

  ## observed value of test statistic
  if(GP.pseudo.alter){
    kernel.result.group = list()
    H = length(unique(Z))
    for(h in 1:H){
      kernel.result.group[[h]] = kernel.matrix(X[Z==h,,drop=FALSE], kernel.list = kernel.list )
    }
    result.obs <- test.stat(Y, kernel.result, threshold,
                            # lik ratio tilde{H}_1 vs tilde{H}_0
                            GP.alter = FALSE, cov.alter, var.equal,
                            # lik ratio tilde{H}_ps vs tilde{H}_0
                            GP.pseudo.alter = TRUE, kernel.result.group,
                            # lik ratio H_1 vs H_0
                            fix.alter = FALSE, Phi, Z)
    test.obs = result.obs$stat
  }

  ## permuted value of the test statistic
  test.perm = rep(NA, perm.max)
  for(iter in 1:perm.max){
    Y.perm = rep(NA, n)
    for(k in 1:m){
      Y.perm[cov.set == k] = sample(Y[cov.set == k] )
    }
    if(GP.pseudo.alter){
      result.perm  <- test.stat(Y = Y.perm, kernel.result, threshold,
                                # lik ratio tilde{H}_1 vs tilde{H}_0
                                GP.alter = FALSE, cov.alter, var.equal,
                                # lik ratio tilde{H}_ps vs tilde{H}_0
                                GP.pseudo.alter = TRUE, kernel.result.group,
                                # lik ratio H_1 vs H_0
                                fix.alter = FALSE, Phi, Z)
      test.perm[iter] = result.perm$stat
    }
  }

  ## p value and results
  p.val = mean(test.perm >= test.obs)
  result.list = list(p.val = p.val, test.perm = test.perm, test.obs = test.obs)

  return(result.list)
}




