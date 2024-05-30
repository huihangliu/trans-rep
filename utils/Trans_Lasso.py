import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV

def mse_fun(beta, est, X_test=None):
    est_err = np.sum((beta - est) ** 2)
    pred_err = np.nan
    if X_test is not None:
        pred_err = np.mean((X_test @ (beta - est)) ** 2)
    return {'est_err': est_err, 'pred_err': pred_err}

def ind_set(n_vec, k_vec):
  ind_re = []
  for k in k_vec:
    # if k == 0:
    #   ind_re.extend(range(n_vec[0]))
    # else:
    #   ind_re.extend(range(sum(n_vec[:k]), sum(n_vec[:(k+1)])))
    ind_re.extend(range(sum(n_vec[:k]), sum(n_vec[:(k+1)])))
  return ind_re

def coef_gen(s, h, q=30, size_A0=0, M=None, sig_beta=None, sig_delta1=None, sig_delta2=None, p=None, exact=True):

  beta0 = np.concatenate([np.repeat(sig_beta, s), np.repeat(0, p - s)])
  W = np.tile(beta0, (M, 1)).T  # Replicating beta0 M times
  W[0, :] -= 2 * sig_beta
  for k in range(M):
    if k < size_A0:
      # data source k is similar to target dataset 0
      if exact: # exact means that the coefficients are exactly the same (for some positions)
        samp0 = np.random.choice(p, h, replace=False)
        W[samp0, k] += np.repeat(-sig_delta1, h)
      else: # exact=False means that the coefficients are similar but not exactly the same
        W[:100, k] += np.random.normal(0, h / 100, 100)
    else: # data source k is not similar to target dataset 0
      if exact:
        samp1 = np.random.choice(p, q, replace=False)
        W[samp1, k] += np.repeat(-sig_delta2, q)
      else:
        W[:100, k] += np.random.normal(0, q / 100, 100)
  return {
    'W': W, 
    'beta0': beta0
  }

def agg_fun(B, X_test, y_test, total_step=10):
  if np.all(B == 0):
    return np.repeat(0, B.shape[0])
  p, K = B.shape

  theta_hat = np.exp(-np.sum((np.tile(y_test, (K, 1)).T - X_test @ B) ** 2, axis=0) / 2)
  theta_hat /= np.sum(theta_hat)
  # print(f"shape of theta_hat: {theta_hat.shape}")
  theta_old = theta_hat.copy()
  beta = B @ theta_hat
  for _ in range(total_step):
    theta_hat = np.exp(-np.sum((np.tile(y_test, (K, 1)).T - X_test @ B) ** 2, axis=0) / 2 + np.sum((np.tile(X_test @ beta, (K, 1)).T - X_test @ B) ** 2, axis=0) / 8)
    theta_hat /= np.sum(theta_hat)
    beta = (B @ theta_hat) / 4 + 3 * beta / 4
    if np.sum(np.abs(theta_hat - theta_old)) < 1e-3:
      break
    theta_old = theta_hat
  return {
    'theta': theta_hat, 
    'beta': beta
  }

def las_kA(X, y, A0, n_vec, lam_const=None):

  p = X.shape[1]
  size_A0 = len(A0)
  beta_kA = np.zeros(p)

  if size_A0 > 0:
    ind_kA = ind_set(n_vec, A0)
    ind_1 = range(n_vec[0])
    y_A = y[ind_kA]

    if lam_const is None:
      elastic_net_cv = ElasticNetCV(cv=8, l1_ratio=1.0, alphas=np.linspace(1, 0.1, 10) * np.sqrt(2 * np.log(p) / len(ind_kA)), fit_intercept=False)
      elastic_net_cv.fit(X[ind_kA, :], y_A)
      lam_const = elastic_net_cv.alpha_ / np.sqrt(2 * np.log(p) / len(ind_kA))

    elastic_net = ElasticNet(alpha=lam_const * np.sqrt(2 * np.log(p) / len(ind_kA)), l1_ratio=1.0, fit_intercept=False)
    elastic_net.fit(X[ind_kA, :], y_A)
    w_kA = elastic_net.coef_

    # Thresholding small coefficients to zero
    w_kA[np.abs(w_kA) < lam_const * np.sqrt(2 * np.log(p) / len(ind_kA))] = 0

    # Fit the second Lasso model
    y_residual = y[ind_1] - X[ind_1, :] @ w_kA
    elastic_net.fit(X[ind_1, :], y_residual)
    delta_kA = elastic_net.coef_
    delta_kA[np.abs(delta_kA) < lam_const * np.sqrt(2 * np.log(p) / len(ind_1))] = 0

    beta_kA = w_kA + delta_kA

  else:
    # only use the target dataset
    elastic_net_cv = ElasticNetCV(cv=8, l1_ratio=1.0, alphas=np.linspace(1, 0.1, 20) * np.sqrt(2 * np.log(p) / n_vec[0]), fit_intercept=False)
    elastic_net_cv.fit(X[:n_vec[0], :], y[:n_vec[0]])
    beta_kA = elastic_net_cv.coef_

  return {
    'beta_kA': beta_kA, 
    'w_kA': w_kA if size_A0 > 0 else np.nan, 
    'lam_const': lam_const
  }

def trans_lasso(X, y, n_vec, I_til):

  M = len(n_vec) - 1 # number of sources dataset
  # p = X.shape[1]
  n_vec[0] -= len(I_til) # number of samples in the target dataset (after removing the first 50 samples)

  # Step 1: Data preparation
  X0_til = X[I_til, :] # Used for aggregation
  y0_til = y[I_til]
  X_rest = np.delete(X, I_til, axis=0) # Used for oracle trans-lasso
  y_rest = np.delete(y, I_til, axis=0)

  # print(f"shape of X0_til: {X0_til.shape}")
  # print(f"shape of y0_til: {y0_til.shape}")
  # print(f"shape of X_rest: {X_rest.shape}")
  # print(f"shape of y_rest: {y_rest.shape}")

  # Step 2: Calculate Rhat
  Rhat = np.zeros(M + 1)
  ind_1 = ind_set(n_vec, [0])

  for k in range(1, M + 1): # k = 1, ..., M. source datasets
    ind_k = ind_set(n_vec, [k])
    Xty_k = (X[ind_k, :].T @ y[ind_k]) / n_vec[k] - (X[ind_1, :].T @ y[ind_1]) / n_vec[0]
    margin_T = np.sort(np.abs(Xty_k), axis=None)[::-1][:round(n_vec[0] / 3)]
    Rhat[k] = np.sum(margin_T ** 2)
  
  # Selection rule based on Rhat
  Tset = []
  for kk in range(1, len(np.unique(np.argsort(Rhat[1:])))):
    Tset.append(np.where(np.argsort(Rhat[1:]) <= kk)[0])
  # print(f"Tset: {len(Tset)}")

  # Initial regression: only use the target dataset
  init_re = las_kA(X_rest, y_rest, A0=[], n_vec=n_vec)
  beta_T = [init_re['beta_kA']]

  # Perform Lasso regression for each selection
  for T_k in Tset:
    # print(f"Current T_k: {T_k}")
    re_k = las_kA(X_rest, y_rest, A0=T_k, n_vec=n_vec, lam_const=init_re['lam_const'])
    beta_T.append(re_k['beta_kA'])

  # print(f"Number of selected sources: {len(beta_T)}")
  beta_T = np.unique(beta_T, axis=0)

  beta_T = beta_T.T # column size is p

  # Aggregation
  agg_re1 = agg_fun(B=beta_T, X_test=X0_til, y_test=y0_til)

  return {
    'beta_hat': agg_re1['beta'], 
    'theta_hat': agg_re1['theta'], 
    'rank_pi': np.argsort(Rhat[1:])
  }
