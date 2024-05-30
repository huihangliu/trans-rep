"""
This file contains the implements of Model averaging transfer learning methods. 
author: Huihang Liu
email: huihang@mail.ustc.edu.cn
"""

# import numpy as np
import autograd.numpy as np   # Thinly-wrapped version of Numpy, for linear_rep() use
from cvxopt import matrix, solvers
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from autograd import grad
from numpy.linalg import norm, svd

class ModelAveraging:
  def __init__(self, num_models=None, model_type=None):
    self.num_models = num_models if num_models is not None else 1
    self.model_types = model_type if model_type is not None else ['linear'] * self.num_models

  def estimate(self, Y, X, Z, model_type):
    if Z.shape[0] != 0: 
      combined_XZ = np.column_stack((X, Z))
    else:
      combined_XZ = X

    size_beta = X.shape[1]

    if model_type == 'linear':
      model_fitted = sm.OLS(Y, combined_XZ).fit()
    elif model_type == 'logistic':
      model_fitted = sm.Logit(Y, combined_XZ).fit()

    coef_hat = model_fitted.params
    return {
      'beta_hat': coef_hat[:size_beta],
      'alpha_hat': coef_hat[size_beta:],
      'variance_beta_hat': model_fitted.bse[:size_beta] ** 2,
      'aic': model_fitted.aic,
      'bic': model_fitted.bic
    }

  def predict_y_main(self, X_, beta_, model_type):
    if model_type == 'linear':
      Y_hat = np.dot(X_, beta_)
    elif model_type == 'logistic':
      tmp_ = np.exp(np.dot(X_, beta_))
      Y_hat = tmp_ / (1 + tmp_)
      Y_hat[np.isinf(tmp_)] = 1
    return Y_hat

  def map(self, Y, X, Z):
    """
    Input
      Y: a list of Ys
      X: a list of Xs
      Z: a list of Zs
    """
    num_models = len(Y)

    data_used = []
    sample_size_models = []
    for Y_m, X_m, Z_m in zip(Y, X, Z):
      data_used.append({'Y': np.array(Y_m), 'X': np.array(X_m), 'Z': np.array(Z_m)})
      sample_size_models.append(len(Y_m))

    # Estimate parameters
    estimated_parameters = []
    beta_hat_models = []
    for data, m_type in zip(data_used, self.model_types):
      est_params = self.estimate(data['Y'], data['X'], data['Z'], m_type)
      estimated_parameters.append(est_params)
      beta_hat_models.append(est_params['beta_hat'])

    beta_hat_models = np.column_stack(beta_hat_models)

    # Predict parameters
    Y_hat_models = []
    if data_used[0]['Z'].shape[0] != 0:
      tmp_XZ = np.column_stack((data_used[0]['X'], data_used[0]['Z']))
    else:
      tmp_XZ = data_used[0]['X']
    for beta_hat, m_type in zip(beta_hat_models.T, self.model_types):
      # debug, ??, this loop could be improved
      tmp_coef_hat = np.concatenate((beta_hat, estimated_parameters[0]['alpha_hat']))
      Y_hat_models.append(self.predict_y_main(tmp_XZ, tmp_coef_hat, self.model_types[0]))

    Y_hat_models = np.column_stack(Y_hat_models)

    # Model averaging using quadratic programming with CVXOPT
    # Leave-One-Out (LOO) Error Calculation
    error_loo_models = np.zeros((sample_size_models[0], num_models))
      
    for idx_cur_sample in range(sample_size_models[0]):
      # Extracting the current sample from the target data (the first data). 
      tmp_Y = data_used[0]['Y']
      tmp_Y_train = np.delete(tmp_Y, idx_cur_sample, axis=0)
      tmp_Y_test = tmp_Y[idx_cur_sample]

      tmp_X = data_used[0]['X']
      tmp_X_train = np.delete(tmp_X, idx_cur_sample, axis=0)
      tmp_X_test = tmp_X[idx_cur_sample]

      tmp_Z = data_used[0]['Z']
      if tmp_Z.shape[0] != 0:
        tmp_Z_train = np.delete(tmp_Z, idx_cur_sample, axis=0)
        tmp_Z_test = tmp_Z[idx_cur_sample]
      else: 
        tmp_Z_train = np.zeros((tmp_X_train.shape[0], 0))
        tmp_Z_test = np.zeros(0)

      for idx_cur_model in range(num_models):
        if idx_cur_model == 0:
          # Re-estimate main model excluding the current sample
          tmp_estimated_params = self.estimate(tmp_Y_train, tmp_X_train, tmp_Z_train, self.model_types[0])
          tmp_beta_hat = tmp_estimated_params['beta_hat']
          tmp_alpha_hat = tmp_estimated_params['alpha_hat']

          # Predict for the excluded sample
          y_hat_loo = self.predict_y_main(
            np.concatenate((tmp_X_test, tmp_Z_test)),
            np.concatenate((tmp_beta_hat, tmp_alpha_hat)),
            self.model_types[0]
          )
        else:
          # Use the coefficients from other models
          tmp_beta_hat = beta_hat_models[:, idx_cur_model]
          y_hat_loo = self.predict_y_main(
              np.concatenate((tmp_X_test, tmp_Z_test)),
              np.concatenate((tmp_beta_hat, estimated_parameters[0]['alpha_hat'])),
              self.model_types[0]
          )

        # Compute the error for the current sample and model
        error_loo_models[idx_cur_sample, idx_cur_model] = y_hat_loo - tmp_Y_test


    H = error_loo_models.T @ error_loo_models / sample_size_models[0] + np.eye(num_models) * 1e-8
    P = matrix(H)
    q = matrix(np.zeros(num_models))

    # Equality constraint: sum of weights equals 1
    A = matrix(1.0, (1, num_models))
    b = matrix(1.0)

    # Inequality constraints: weights are non-negative
    G = matrix(np.vstack((-np.eye(num_models), np.eye(num_models))))
    h = matrix(np.hstack((np.zeros(num_models), np.ones(num_models))))

    # Solving the quadratic programming problem
    solvers.options['show_progress'] = False # Set the verbosity option before calling the solver
    sol = solvers.qp(P, q, G, h, A, b)
    weight_hat_ma = np.array(sol['x']).flatten()

    beta_hat_ma = beta_hat_models @ weight_hat_ma
    # y_ma_hat_main = self.predict_y_main(
    #   np.column_stack((data_used[0]['X'], data_used[0]['Z'])),
    #   np.concatenate((beta_hat_ma, estimated_parameters[0]['alpha_hat'])),
    #   self.model_types[0]
    # )

    return {
      # 'y_hat': y_ma_hat_main,
      'beta_hat': beta_hat_ma,
      'alpha_hat': estimated_parameters[0]['alpha_hat'],
      'weight': weight_hat_ma
    }

    # Usage example
    # map(Y_list, X_list, Z_list, model_type_list)

  def pooled_regression(self, Y, X, Z):
    size_beta = X[0].shape[1] # the dimension of X in the first dataset
    data_used = []
    sample_size_models = []
    for Y_m, X_m, Z_m in zip(Y, X, Z):
      data_used.append({'Y': np.array(Y_m), 'X': np.array(X_m), 'Z': np.array(Z_m)})
      sample_size_models.append(len(Y_m))
      
    X_pooled = np.empty((0, X[0].shape[1] + Z[0].shape[1]))
    Y_pooled = np.array([])
    # Pooling data from all models
    for data in data_used:
      tmp_X_pooled = np.column_stack((data['X'], data['Z']))
      X_pooled = np.vstack((X_pooled, tmp_X_pooled))
      Y_pooled = np.concatenate((Y_pooled, data['Y'].flatten()))

    # Fitting the pooled model
    if self.model_types[0] == 'linear':
      model_fitted = sm.OLS(Y_pooled, X_pooled).fit()
    elif self.model_types[0] == 'logistic':
      model_fitted = sm.Logit(Y_pooled, X_pooled).fit()

    coef_hat = model_fitted.params
    beta_alpha_hat_pooled = coef_hat
    beta_hat_pooled = coef_hat[:size_beta]
    alpha_hat_pooled = coef_hat[size_beta:]

    y_hat = self.predict_y_main(np.column_stack((X[0], Z[0])), beta_alpha_hat_pooled, self.model_types[0])

    return {
      'y_hat': y_hat,
      'beta_hat': beta_hat_pooled,
      'alpha_hat': alpha_hat_pooled,
    }

  def meta_analysis(self, Y, X, Z):
    num_models = len(Y)
    size_beta = X[0].shape[1] # the dimension of X in the first dataset

    data_used = []
    sample_size_models = []
    for Y_m, X_m, Z_m in zip(Y, X, Z):
      data_used.append({'Y': np.array(Y_m), 'X': np.array(X_m), 'Z': np.array(Z_m)})
      sample_size_models.append(len(Y_m))

    # Estimate parameters
    estimated_parameters = []
    beta_hat_models = []
    for data, m_type in zip(data_used, self.model_types):
      est_params = self.estimate(data['Y'], data['X'], data['Z'], m_type)
      estimated_parameters.append(est_params)
      beta_hat_models.append(est_params['beta_hat'])

    beta_hat_models = np.column_stack(beta_hat_models)

    num_models = len(estimated_parameters)
    weight_hat_meta = np.zeros((size_beta, num_models))

    # Computing weights based on the inverse of the estimation variance
    for idx_cur_model in range(num_models):
      weight_hat_meta[:, idx_cur_model] = 1 / ((estimated_parameters[idx_cur_model]['variance_beta_hat']) + 1e-8)

    # Normalizing the weights
    tmp_weight_hat_meta_rowSums = np.sum(weight_hat_meta, axis=1) + 1e-8
    for idx_cur_variable in range(size_beta):
      weight_hat_meta[idx_cur_variable, :] /= tmp_weight_hat_meta_rowSums[idx_cur_variable] + 1e-8
    
    # asign the nan values in weight_hat_meta to 0
    weight_hat_meta[np.isnan(weight_hat_meta)] = 1e-8

    # Computing the weighted average of the coefficients
    beta_hat_meta = np.sum(beta_hat_models * weight_hat_meta, axis=1)

    if len(Z[0]) != 0:
      tmp_XZ = np.column_stack((X[0], Z[0]))
    else:
      tmp_XZ = X[0]
    y_hat = self.predict_y_main(tmp_XZ, np.concatenate((beta_hat_meta, estimated_parameters[0]['alpha_hat'])), self.model_types[0])

    return {
      'y_hat': y_hat,
      'beta_hat': beta_hat_meta,
      'alpha_hat': estimated_parameters[0]['alpha_hat'],
      'weight': weight_hat_meta
    }

  def trans_lasso(self, Y, X, Z):
    """
    First, do lasso regression on source data, and then do lasso on the target data to adjust the estimated coefficient from the first step.
    This method requires the same set of covariates in all models. 
    """
    dim_of_Zs = [Z[t].shape[1] for t in range(self.num_models)]
    # check if the dimension of Z is the same in all models
    if len(set(dim_of_Zs)) != 1:
      raise ValueError("The dimension of Z should be the same in all models for trans-lasso method.")

    num_models = len(Y)
    if num_models != self.num_models:
      print(f"num_models: {num_models}, self.num_models: {self.num_models}")
      raise ValueError("The number of models is not correct in trans-lasso.")

    size_beta = X[0].shape[1]

    # combined all data set together (expect the last one)
    X_source = np.vstack(X[1:])
    Z_source = np.vstack(Z[1:])
    Y_source = np.concatenate(Y[1:])
    XZ_source = np.column_stack((X_source, Z_source))
    # lasso regression on source data, tuned by CV
    lasso_source = LassoCV(cv=5, random_state=0, fit_intercept=False).fit(XZ_source, Y_source.flatten())
    beta_hat_source = lasso_source.coef_[:size_beta]
    alpha_hat_source = lasso_source.coef_[size_beta:]
    # fine-tune the lasso regression on target data
    X_target = X[0]
    Z_target = Z[0]
    Y_target = Y[0]
    XZ_target = np.column_stack((X_target, Z_target))
    lasso_target = LassoCV(cv=5, random_state=0, fit_intercept=False).fit(XZ_target, Y_target.flatten()-lasso_source.predict(XZ_target))
    beta_hat_target = lasso_target.coef_[:size_beta] + beta_hat_source
    alpha_hat_target = lasso_target.coef_[size_beta:] + alpha_hat_source

    y_hat = self.predict_y_main(np.column_stack((X[0], Z[0])), np.concatenate((beta_hat_target, alpha_hat_target)), self.model_types[0])

    return {
      'y_hat': y_hat,
      'beta_hat': beta_hat_target,
      'alpha_hat': alpha_hat_target
    }

  def trans_ridge(self, Y, X, Z):
    """
    First, do ridge regression on source data, and than do lasso on the target data to adjust the estimated coefficient from the first step.
    This method requires the same set of covariates in all models. 
    """
    dim_of_Zs = [Z[t].shape[1] for t in range(self.num_models)]
    # check if the dimension of Z is the same in all models
    if len(set(dim_of_Zs)) != 1:
      raise ValueError("The dimension of Z should be the same in all models for trans-lasso method.")

    num_models = len(Y)
    if num_models != self.num_models:
      print(f"num_models: {num_models}, self.num_models: {self.num_models}")
      raise ValueError("The number of models is not correct in trans-lasso.")

    size_beta = X[0].shape[1]

    # combined all data set together (expect the last one)
    X_source = np.vstack(X[1:])
    Z_source = np.vstack(Z[1:])
    Y_source = np.concatenate(Y[1:])
    XZ_source = np.column_stack((X_source, Z_source))
    # lasso regression on source data, tuned by CV
    ridge_source = RidgeCV(cv=5, fit_intercept=False).fit(XZ_source, Y_source.flatten())
    
    beta_hat_source = ridge_source.coef_[:size_beta]
    alpha_hat_source = ridge_source.coef_[size_beta:]

    
    # fine-tune the lasso regression on target data
    X_target = X[0]
    Z_target = Z[0]
    Y_target = Y[0]
    XZ_target = np.column_stack((X_target, Z_target))
    ridge_target = RidgeCV(cv=5, fit_intercept=False).fit(XZ_target, Y_target.flatten()-ridge_source.predict(XZ_target))
    beta_hat_target = ridge_target.coef_[:size_beta] + beta_hat_source
    alpha_hat_target = ridge_target.coef_[size_beta:] + alpha_hat_source

    y_hat = self.predict_y_main(np.column_stack((X[0], Z[0])), np.concatenate((beta_hat_target, alpha_hat_target)), self.model_types[0])

    return {
      'y_hat': y_hat,
      'beta_hat': beta_hat_target,
      'alpha_hat': alpha_hat_target
    }

  def linear_rep(self, Y, X, Z, r=1, eta = 0.05, delta = 0.05, max_iter = 2000):
    """
    The MTL function is a multi-task learning algorithm that aims to learn a shared representation across multiple tasks.

    1 Param Initialization
      The dimensions of the input data `x` are extracted to get the number of tasks `T`, the number of samples `n`, and the number of features `p`.
      `A_hat` is initialized as a matrix of zeros with dimensions (p, r). The first r diagonal elements are set to 1, indicating that it starts with an identity matrix of size r.
      `theta_hat` is initialized as a matrix of zeros with dimensions (r, T).
    2 Initialization for the first task
      A function `ftotal` is defined to compute the loss for the first task. The loss is the squared error between the predicted and actual outputs.
      The `gradient` of this loss with respect to A and theta is computed using autograd's grad function.
      An iterative process updates A_hat and theta_hat for the first task using `gradient descent`.
    3 Learning Shared Representation:
      A function `ftotal` is defined to compute the overall loss across all tasks. The loss consists of:
      The squared error between the predicted and actual outputs for each task.
      A `regularization term` that penalizes the difference between A_hat and theta_hat.
      The gradient of this overall loss with respect to A and theta is computed.
      An iterative process updates A_hat and theta_hat for all tasks using gradient descent.
    4 Compute Final Estimates:
      The final estimates `beta_hat_step1` for each task are computed as the product of A_hat and theta_hat.
    5 Return Value:
      The function returns `beta_hat_step1`, which are the estimated coefficients or parameters for the tasks.
    """
    
    size_beta = X[0].shape[1]
    shapes_Z = [Z[t].shape for t in range(self.num_models)]
    if len(set(shapes_Z)) != 1:
      raise ValueError("All Zs must have the same shape to stack in linear_rep().")

    XZ = [np.column_stack((X[t], Z[t])) for t in range(self.num_models)]
    x = np.stack(XZ, axis=0)
    Y_tmp = [Y[t].flatten() for t in range(self.num_models)]
    y = np.stack(Y_tmp, axis=0)

    # Param initialization
    T = x.shape[0]
    n = x.shape[1]
    p = x.shape[2]
    A_hat = np.zeros((p, r))
    for t in range(T):
      A_hat[0:r, 0:r] = np.identity(r)
    theta_hat = np.zeros((r, T))

    # Initialization for the first task
    t = 0
    def ftotal(A, theta):
      return (1/n*np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))    
    ftotal_grad = grad(ftotal, argnum = [0, 1])
    for _ in range(200):
      # gradient descent
      S = ftotal_grad(A_hat, theta_hat[:, t])
      A_hat = A_hat - eta*S[0]
      theta_hat[:, t] = theta_hat[:, t] - eta*S[1]
      # Problem: How to garantee the orthogonality of A_hat during the iteration??
      # Over iterations, A_hat can drift away from being orthogonal, which might affect the performance or interpretability of the multi-task learning model.

    # Learning Shared Representation
    for _ in range(max_iter):
      def ftotal(A, theta):
        s = 0
        for t in range(T):
          s = s + 1/n*1/T*np.dot(y[t, :] - x[t, :, :] @ A @ theta[:, t], y[t, :] - x[t, :, :] @ A @ theta[:, t])
        s = s + delta * max(abs(np.linalg.eigh(A.T @ A - theta @ theta.T)[0]))
        return(s)
      ftotal_grad = grad(ftotal, argnum = [0,1])
      S = ftotal_grad(A_hat, theta_hat)
      A_hat = A_hat - eta*S[0]
      theta_hat = theta_hat - eta*S[1]

    # Compute final estimates
    beta_hat_step1 = np.zeros((p, T))
    for t in range(T):
      beta_hat_step1[:, t] = A_hat @ theta_hat[:, t]

    return {
      'beta_hat': beta_hat_step1[:size_beta, 0],
      'alpha_hat': beta_hat_step1[size_beta:, 0]
    }

  def demo(self):
    # Define parameters
    np.random.seed(2023)

    def generate_data(model_type, sample_size, beta, alpha, rho=0.5, sigma_XZ=2, sigma_epsilon=0.5, seed=None, Cov_struc="Compound"):
      if seed is not None: 
          np.random.seed(seed)

      dim_X = len(beta)
      dim_Z = len(alpha)
      dim_XZ_nointercept = dim_X + (dim_Z - 1)  # remove the intercept term

      # Generate X and Z
      Mu_XZ = np.zeros(dim_XZ_nointercept)
      if Cov_struc == "Compound":
          Sig_XZ = np.full((dim_XZ_nointercept, dim_XZ_nointercept), rho)
          np.fill_diagonal(Sig_XZ, 1)
      elif Cov_struc == "AR(1)":
          Sig_XZ = rho ** np.abs(np.subtract.outer(np.arange(dim_XZ_nointercept), np.arange(dim_XZ_nointercept)))
      
      Sig_XZ *= sigma_XZ ** 2
      tmp_XZ = np.random.multivariate_normal(Mu_XZ, Sig_XZ, sample_size)
      X = tmp_XZ[:, :dim_X]
      Z = np.hstack((np.ones((sample_size, 1)), tmp_XZ[:, dim_X:]))

      # Generate Y
      if model_type == "linear":
          epsilon = np.random.normal(0, sigma_epsilon, sample_size)
          Y_expect = X.dot(beta) + Z.dot(alpha)
          Y = Y_expect + epsilon
      elif model_type == "logistic":
          Y_expect = np.exp(X.dot(beta) + Z.dot(alpha))
          prob_Y = Y_expect / (1 + Y_expect)
          Y = np.random.binomial(1, prob_Y)

      return {
          'X': X,
          'Z': Z,
          'Y': Y,
          'Y_expect': Y_expect  # expectation value
      }

    sample_size_models = [100, 200, 100]
    beta_true = [0.5, 0.6, -0.61, -0.48]
    alpha_true = [
        [0.40, 0.60, 0.50, -0.30, -0.25],
        [0.49, 0.08, 0.09, -0.04, -0.06, 2.50],
        [0.51, 0.07, 0.10, -0.05, -0.04, -1, 1]
    ]
    num_models = len(sample_size_models)
    model_types = ["linear"] * num_models
    self.num_models = num_models
    self.model_types = model_types

    # Generate data for models
    Y, X, Z = [], [], []
    for i in range(num_models):
      data = generate_data(
          model_types[i], 
          sample_size_models[i], 
          beta_true, 
          alpha_true[i], 
          Cov_struc="Compound"
        )
      Y.append(data['Y'])
      X.append(data['X'])
      Z.append(data['Z'][:, :5])  # Adjust the size of Z if necessary

    # Apply the map function
    re = self.map(Y, X, Z)
    print(f"MAP beta_hat: {re['beta_hat']}")
    re = self.pooled_regression(Y, X, Z)
    print(f"Pooled regression beta_hat: {re['beta_hat']}")
    re = self.meta_analysis(Y, X, Z)
    print(f"Meta analysis beta_hat: {re['beta_hat']}")

# # Usage
# model_averaging = ModelAveraging()
# model_averaging.demo()
