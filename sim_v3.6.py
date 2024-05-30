"""
date:   2023.01.30
author: huihang@mail.ustc.edu.cn
note:
  [v3.6]
  增加表征的正则化, 约束表征为独立的标准正态或独立的0-1均匀分布. 从训练的结果来看, 似乎是无效的. 在训练过程中, 训练误差一直在下降, 但验证误差并不会一直下降. 
  增加了渐近方差的估计, 改变了 DGP 固定了参数的生成模式. 
  [stl] 的学习方式, 模型的所有部分都使用同一个学习率或者使用不同的学习率. 
"""

# %% Import packages
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.NN_Models import PartialRegressionNN, PartialLinearRegression, SharedNN, RegressionNN, Discriminator
from utils.Data_Generator import DGPs
from utils.Trans_MA import ModelAveraging
from utils.Trans_Lasso import trans_lasso
from utils.auto_save import backup_files
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
import numpy as np
import time
from colorama import init, Fore
import argparse
import platform
system_type = platform.system().lower()
if system_type == 'linux':
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  torch.set_num_threads(1)

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

# %% 
# TODO Re-run exp1

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_source", help="number of samples in training source", type=int, default=400)
parser.add_argument("--n_target", help="number of samples in training target", type=int, default=50)
parser.add_argument("--p", help="data dimension", type=int, default=10)
parser.add_argument("--q", help="data dimension", type=int, default=5)
parser.add_argument("--r", help="representation dimension", type=int, default=5)
parser.add_argument("--r_dgp", help="representation dimension in dgp", type=int, default=5)
parser.add_argument("--T", help="number of source dataset", type=int, default=20)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=1)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=6)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--verbose", help="print training information", type=int, default=True)

args = parser.parse_args()

if len(args.record_dir) > 0:
  backup_files(__file__, args.record_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# %% Some functions about data generation and evaluation

def train_loop(model, data_loader, loss_fn, optimizer, fine_tune=False):
  loss_sum = 0
  for x, y in data_loader:
    pred = model(x, is_training=True)
    loss = loss_fn(pred, y)
    loss_sum += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if fine_tune:
    loss_sum = model.fine_tune(data_loader)
  return loss_sum / len(data_loader)

def train_loop_mtl(models_source, dataloaders_aux, loss_fn, optimizer_pre_train):
  # loop for source datas
  T = len(models_source) # redefine T
  for batch_idx in range(len(dataloaders_aux[0])):
    # we assume that the batch size is the same across different source datasets, so we just loop for the first batch_size
    loss_sum_aux = 0
    for t, dataloader in enumerate(dataloaders_aux):
      # Get the batch
      x, y = next(iter(dataloader))
      pred = models_source[t](x, is_training=True)
      loss = loss_fn(pred, y)
      loss_sum_aux += loss
    optimizer_pre_train.zero_grad()
    # for t in range(T): optimizer_source_linear[t].zero_grad()
    loss_sum_aux.backward()
    optimizer_pre_train.step()
    # for t in range(T): optimizer_source_linear[t].step()

    # fine tune the last layer
    if True:
      loss_aux = np.array([0.0]*T)
      for t, dataloader in enumerate(dataloaders_aux):
        loss_aux[t] = models_source[t].fine_tune(dataloader)
    

  # return [loss_aux[t]/len(dataloaders_aux[t]) for t in range(T)]
  return loss_sum_aux.item() / len(dataloaders_aux[0])

def train_loop_orthognal(models_source, D_net, dataloaders_aux, loss_fn, optimizer_pre_train, optimizer_D, zlr):
  """
  Train the model for one epoch
  Inputs:
    net: Generator network
    D_net: Discriminator network
    trainLoader: Training data loader
    optimizer: Optimizer for the generator network
    optimizer_D: Optimizer for the discriminator network
    zlr: Learning rate for the generator network
    dim_rep: Dimension of the representation
    device: Device for training
  """
  # initialize the parameters
  dim_rep = models_source[0].dim_rep # the dimension of the representation
  shared_net = models_source[0].shared_layers # the shared network of the source models
  p = models_source[0].dim_nonlinear
  q = models_source[0].dim_linear
  MSEloss = nn.MSELoss()

  # Iterate over training data
  for batch_idx in range(len(dataloaders_aux[0])):
    # we assume that the batch size is the same across different source datasets, so we just loop for the first batch_size
    loss_sum_aux = 0
    tmp_w = []
    for t, dataloader in enumerate(dataloaders_aux):
      # Get the batch
      x, y = next(iter(dataloader))

      # store the representations from source models
      w = shared_net(x[:, q:]) # input (nonlinear part) data to the generator, w is the latent representation
      tmp_w.append(w)
      # store the prediction error from source models
      pred = models_source[t](x, is_training=True)
      loss = loss_fn(pred, y)
      loss_sum_aux += loss

    # train the discriminator
    w = torch.cat(tmp_w, dim=0)
    new_w = Variable(w.clone())
    D_real = torch.sigmoid(D_net(new_w)) # input latent representation to the discriminator
    D_fake = torch.sigmoid(D_net(torch.randn(new_w.shape[0], dim_rep))) # input Gaussian noise to the discriminator
    D_loss_real = torch.nn.functional.binary_cross_entropy(D_real, torch.ones(new_w.shape[0], 1))  # loss for real data
    D_loss_fake = torch.nn.functional.binary_cross_entropy(D_fake, torch.zeros(new_w.shape[0], 1)) # loss for fake data
    D_loss = (D_loss_real + D_loss_fake)/2. # loss of the discriminator part
    optimizer_D.zero_grad()
    D_loss.backward()
    optimizer_D.step()
    w.detach_() # detach the representation from the graph

    # manually set the update-direction of weights of the generator
    w_t = Variable(w.clone(), requires_grad=True)
    d = - D_net(w_t)
    d.backward(torch.ones(w_t.shape[0], 1), retain_graph=True) # gradient???
    w = w + zlr * w_t.grad  # manually set the update-direction of weights of the generator

    # train the source models
    latent = torch.cat(tmp_w, dim=0) # latent representation of all data (from source models)
    loss_regularization = MSEloss(w, latent) # push the generator to the direction as we expected (manually set one)
    loss = loss_sum_aux + loss_regularization # loss = MSE + regularize
    optimizer_pre_train.zero_grad()
    loss.backward()
    optimizer_pre_train.step()

    print(f"loss regular: {loss_regularization.item()}, loss sum: {loss_sum_aux.item()}")

    # fine tune the last layer
    loss_aux = np.array([0.0]*args.T)
    for t, dataloader in enumerate(dataloaders_aux):
      loss_aux[t] = models_source[t].fine_tune(dataloader)

  return [loss_aux[t]/len(dataloaders_aux[t]) for t in range(args.T)]

# Function for computing pairwise distances between vectors in x and y
def pairwise_distances(x, y=None):
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    # Computing the norm of vectors in y if y is provided
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y = x
    y_norm = x_norm.view(1, -1)

  # Calculating pairwise distances    
  dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
  return dist

# Function to calculate distance correlation between X and Y
def cor(X, Y, n):
  # Computing pairwise distances for X and Y
  DX = pairwise_distances(X)
  DY = pairwise_distances(Y)
  # Centering matrix for distance matrices
  J = (torch.eye(n) - torch.ones(n,n) / n)
  RX = J @ DX @ J
  RY = J @ DY @ J
  # Computing distance covariance and correlation
  covXY = torch.mul(RX, RY).sum()/(n*n)
  covX = torch.mul(RX, RX).sum()/(n*n)
  covY = torch.mul(RY, RY).sum()/(n*n)
  return covXY/torch.sqrt(covX*covY)

def evaluate(model, dataloader, criterion):
  if type(model) is list: # models['aux-nn'] is a list of models
    loss_aux = []
    T = len(model)
    for t in range(T):
      model[t].eval()  # set the model to evaluation mode
      total_loss = 0.0
      with torch.no_grad():
        for batch_x, batch_y in dataloader[t]:
          predictions = model[t](batch_x)
          loss = criterion(predictions, batch_y)
          total_loss += loss.item()
      model[t].train()  # set the model back to training mode
      loss_aux.append(total_loss / len(dataloader[t]))
    return loss_aux
  else:
    model.eval()  # set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
      for batch_x, batch_y in dataloader:
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        total_loss += loss.item()
    model.train()  # set the model back to training mode
    return total_loss / len(dataloader)

def joint_train(model_names):
  colors = [Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTWHITE_EX, Fore.MAGENTA]
  best_valid, model_color = {}, {}
  for i, name in enumerate(model_names):
    best_valid[name] = np.inf
    model_color[name] = colors[i]
  best_valid_target = np.inf
  test_perf = {}
  est_perf = {}
  best_shared_layers = SharedNN(dim_input=args.p, dim_rep=args.r, width=args.width, depth=args.depth)

  if 'lasso' in model_names:
    # run lasso for all data
    lasso = LassoCV(cv=5, random_state=0)
    lasso.fit(dgps.data_target['feature_train'], dgps.data_target['label_train'])
    test_loss = np.mean((lasso.predict(dgps.data_target['feature_test']) - dgps.data_target['label_test'])**2)
    test_perf['lasso'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - lasso.coef_[:dgps.dim_linear])
    # print(f"estimated value: {lasso.coef_[:dgps.dim_linear]}, true value: {dgps.params['beta'][:,0]}")
    est_perf['lasso'] = est_error
    if args.verbose: print(model_color['lasso'] + f"Model [{'lasso'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )
  
  if 'ols-linear' in model_names:
    # run ols for linear part only, ignore the non-linear part
    X_train_linear = dgps.data_target['feature_train'][:, :dgps.dim_linear]
    X_test_linear = dgps.data_target['feature_test'][:, :dgps.dim_linear]
    est_beta_ols = np.linalg.lstsq(X_train_linear, dgps.data_target['label_train'], rcond=None)[0][:,0]
    test_loss = np.mean((X_test_linear @ est_beta_ols - dgps.data_target['label_test'][:,0])**2)
    test_perf['ols-linear'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_beta_ols)
    est_perf['ols-linear'] = est_error
    if args.verbose: print(model_color['ols-linear'] + f"Model [{'ols-linear'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'ols-all' in model_names:
    # run ols for all data
    est_beta_ols = np.linalg.lstsq(dgps.data_target['feature_train'], dgps.data_target['label_train'], rcond=None)[0][:,0]
    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_ols - dgps.data_target['label_test'][:,0])**2)
    test_perf['ols-all'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_beta_ols[:dgps.dim_linear])
    est_perf['ols-all'] = est_error
    if args.verbose: print(model_color['ols-all'] + f"Model [{'ols-all'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )
  
  if 'ols-oracle' in model_names:
    # the non-linear part is known, just do a linear regression for  from the data
    X_train_linear = dgps.data_target['feature_train'][:, :dgps.dim_linear]
    X_train_nonlinear = dgps.data_target['feature_train'][:, dgps.dim_linear:]
    # Y_train_tmp = Factor_train @ partial_linear_model_target.gamma
    # Y_train_tmp = np.zeros((n_train, 1))
    # for i in range(partial_linear_model_target.dim_nonlinear):
    #   Y_train_tmp += partial_linear_model_target.func_zoo[partial_linear_model_target.func_idx[i]](Factor_train[:, i:i + 1]) * partial_linear_model_target.gamma[i]
    # Y_train_tmp = dgps.data_target['label_train'] - Y_train_tmp
    X_train_tmp = np.concatenate((X_train_linear, dgps.h(X_train_nonlinear)), axis=1)
    est_beta_gamma_ols = np.linalg.lstsq(X_train_tmp, dgps.data_target['label_train'], rcond=None)[0][:,0]
    X_test_linear = dgps.data_target['feature_test'][:, :dgps.dim_linear]
    X_test_nonlinear = dgps.data_target['feature_test'][:, dgps.dim_linear:]
    X_test_tmp = np.concatenate((X_test_linear, dgps.h(X_test_nonlinear)), axis=1)
    # Y_test_tmp = np.zeros((n_test, 1))
    # for i in range(partial_linear_model_target.dim_nonlinear):
    #   Y_test_tmp += partial_linear_model_target.func_zoo[partial_linear_model_target.func_idx[i]](Factor_test[:, i:i + 1]) * partial_linear_model_target.gamma[i]
    # Y_test_tmp = dgps.data_target['label_test'] - Y_test_tmp
    test_loss = np.mean((X_test_tmp @ est_beta_gamma_ols - dgps.data_target['label_test'][:,0])**2)
    test_perf['ols-oracle'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_beta_gamma_ols[:dgps.dim_linear])
    est_perf['ols-oracle'] = est_error
    # print(f"estimated value: {est_beta_ols}, true value: {dgps.params['beta'][:,0]}")
    if args.verbose: print(model_color['ols-oracle'] + f"Model [{'ols-oracle'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )
    
    test_loss = np.mean((X_test_linear @ dgps.params['beta'] - dgps.data_target['label_test'])**2)
  
  if 'trans-lasso' in model_names:
    if False: 
      model_averaging = ModelAveraging(num_models=T+1)
      X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(T)]
      Z_list = [dgps.data_target['feature_train'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train'][t][:, dgps.dim_linear:] for t in range(T)]
      Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
      res_trans_lasso = model_averaging.trans_lasso(Y_list, X_list, Z_list)
      est_beta_gamma = np.concatenate((res_trans_lasso['beta_hat'], res_trans_lasso['alpha_hat']))
    else:
      X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(args.T)]
      Z_list = [dgps.data_target['feature_train'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train'][t][:, dgps.dim_linear:] for t in range(args.T)]
      Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
      X_tmp = np.concatenate(X_list, axis=0)
      Z_tmp = np.concatenate(Z_list, axis=0)
      XZ_tmp = np.concatenate((X_tmp, Z_tmp), axis=1)
      Y_tmp = np.concatenate(Y_list, axis=0)
      Y_tmp = Y_tmp.flatten()
      n_vec = [np.shape(X_list[t])[0] for t in range(args.T+1)]
      prop_re = trans_lasso(XZ_tmp, Y_tmp, n_vec.copy(), I_til=range(n_vec[0]//3)) # use 1/3 data for Q-aggregation
      est_beta_gamma = prop_re['beta_hat']
    
    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['trans-lasso'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_beta_gamma[:dgps.dim_linear])
    est_perf['trans-lasso'] = est_error
    if args.verbose: print(model_color['trans-lasso'] + f"Model [{'trans-lasso'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'meta-analysis' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    # X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(T)]
    # Z_list = [dgps.data_target['feature_train'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train'][t][:, dgps.dim_linear:] for t in range(T)]
    X_list = [dgps.data_target['feature_train']] + [dgps.data_source['feature_train'][t] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train'][:, :0]] + [dgps.data_source['feature_train'][t][:, :0] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_meta_analysis = model_averaging.meta_analysis(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_meta_analysis['beta_hat'], res_meta_analysis['alpha_hat']))
    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['meta-analysis'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_meta_analysis['beta_hat'][:dgps.dim_linear])
    est_perf['meta-analysis'] = est_error
    if args.verbose: print(model_color['meta-analysis'] + f"Model [{'meta-analysis'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'pooled-regression' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train'][t][:, dgps.dim_linear:] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_pooled_regression = model_averaging.pooled_regression(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_pooled_regression['beta_hat'], res_pooled_regression['alpha_hat']))
    
    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['pooled-regression'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_pooled_regression['beta_hat'])
    est_perf['pooled-regression'] = est_error
    if args.verbose: print(model_color['pooled-regression'] + f"Model [{'pooled-regression'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'map' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    X_list = [dgps.data_target['feature_train']] + [dgps.data_source['feature_train'][t] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train'][:, :0]] + [dgps.data_source['feature_train'][t][:, :0] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_map = model_averaging.map(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_map['beta_hat'], res_map['alpha_hat']))

    # print(f"estimated weight: {res_map['weight']}")

    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['map'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_map['beta_hat'][:dgps.dim_linear])
    est_perf['map'] = est_error
    if args.verbose: print(model_color['map'] + f"Model [{'map'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'trans-lasso-spline' in model_names:
    if False: # use 'ma' implementation
      model_averaging = ModelAveraging(num_models=T+1)
      X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(T)]
      Z_list = [dgps.data_target['feature_train_spline'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train_spline'][t][:, dgps.dim_linear:] for t in range(T)]
      Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
      res_trans_lasso = model_averaging.trans_lasso(Y_list, X_list, Z_list)
      est_beta_gamma = np.concatenate((res_trans_lasso['beta_hat'], res_trans_lasso['alpha_hat']))
    else:
      X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(args.T)]
      Z_list = [dgps.data_target['feature_train_spline'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train_spline'][t][:, dgps.dim_linear:] for t in range(args.T)]
      Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
      X_tmp = np.concatenate(X_list, axis=0)
      Z_tmp = np.concatenate(Z_list, axis=0)
      XZ_tmp = np.concatenate((X_tmp, Z_tmp), axis=1)
      Y_tmp = np.concatenate(Y_list, axis=0)
      Y_tmp = Y_tmp.flatten()
      n_vec = [np.shape(X_list[t])[0] for t in range(args.T+1)]
      prop_re = trans_lasso(XZ_tmp, Y_tmp, n_vec.copy(), I_til=range(n_vec[0]//3)) # use 1/3 data for Q-aggregation
      est_beta_gamma = prop_re['beta_hat']
    
    test_loss = np.mean((dgps.data_target['feature_test_spline'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['trans-lasso-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_beta_gamma[:dgps.dim_linear])
    est_perf['trans-lasso-spline'] = est_error
    if args.verbose: print(model_color['trans-lasso-spline'] + f"Model [{'trans-lasso-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'trans-ridge-spline' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train_spline'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train_spline'][t][:, dgps.dim_linear:] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_trans_ridge = model_averaging.trans_ridge(Y_list, X_list, Z_list)
    est_beta_gamma_trans_ridge = np.concatenate((res_trans_ridge['beta_hat'], res_trans_ridge['alpha_hat']))

    test_loss = np.mean((dgps.data_target['feature_test_spline'] @ est_beta_gamma_trans_ridge - dgps.data_target['label_test'][:,0])**2)
    test_perf['trans-ridge-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_trans_ridge['beta_hat'])
    est_perf['trans-ridge-spline'] = est_error
    if args.verbose: print(model_color['trans-ridge-spline'] + f"Model [{'trans-ridge-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'meta-analysis-spline' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    # X_list = [dgps.data_target['feature_train_spline'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train_spline'][t][:, :dgps.dim_linear] for t in range(T)]
    # Z_list = [dgps.data_target['feature_train_spline'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train_spline'][t][:, dgps.dim_linear:] for t in range(T)]
    X_list = [dgps.data_target['feature_train_spline']] + [dgps.data_source['feature_train_spline'][t] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train_spline'][:, :0]] + [dgps.data_source['feature_train_spline'][t][:, :0] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_meta_analysis = model_averaging.meta_analysis(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_meta_analysis['beta_hat'], res_meta_analysis['alpha_hat']))
    test_loss = np.mean((dgps.data_target['feature_test_spline'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['meta-analysis-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_meta_analysis['beta_hat'][:dgps.dim_linear])
    est_perf['meta-analysis-spline'] = est_error
    if args.verbose: print(model_color['meta-analysis-spline'] + f"Model [{'meta-analysis-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'pooled-regression-spline' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    # X_list = [dgps.data_target['feature_train_spline'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train_spline'][t][:, :dgps.dim_linear] for t in range(T)]
    # Z_list = [dgps.data_target['feature_train_spline'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train_spline'][t][:, dgps.dim_linear:] for t in range(T)]
    X_list = [dgps.data_target['feature_train_spline']] + [dgps.data_source['feature_train_spline'][t] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train_spline'][:, :0]] + [dgps.data_source['feature_train_spline'][t][:, :0] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_pooled_regression = model_averaging.pooled_regression(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_pooled_regression['beta_hat'], res_pooled_regression['alpha_hat']))

    # print(f"estimated coefficient: {est_beta_gamma}")

    test_loss = np.mean((dgps.data_target['feature_test_spline'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['pooled-regression-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_pooled_regression['beta_hat'][:dgps.dim_linear])
    est_perf['pooled-regression-spline'] = est_error
    if args.verbose: print(model_color['pooled-regression-spline'] + f"Model [{'pooled-regression-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'map-spline' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    X_list = [dgps.data_target['feature_train_spline']] + [dgps.data_source['feature_train_spline'][t] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train_spline'][:, :0]] + [dgps.data_source['feature_train_spline'][t][:, :0] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_map = model_averaging.map(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_map['beta_hat'], res_map['alpha_hat']))
    
    test_loss = np.mean((dgps.data_target['feature_test_spline'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['map-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_map['beta_hat'][:dgps.dim_linear])
    est_perf['map-spline'] = est_error
    if args.verbose: print(model_color['map-spline'] + f"Model [{'map-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'ols-spline' in model_names:
    # use ols to fit the linear+spline part
    X = dgps.data_target['feature_train_spline']
    Y = dgps.data_target['label_train']
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X, Y)
    test_loss = np.mean((lm.predict(dgps.data_target['feature_test_spline']) - dgps.data_target['label_test'])**2)
    test_perf['ols-spline'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - lm.coef_[0,:dgps.dim_linear])
    est_perf['ols-spline'] = est_error
    if args.verbose: print(model_color['ols-spline'] + f"Model [{'ols-spline'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'linear-rep' in model_names:
    model_averaging = ModelAveraging(num_models=args.T+1)
    X_list = [dgps.data_target['feature_train'][:, :dgps.dim_linear]] + [dgps.data_source['feature_train'][t][:, :dgps.dim_linear] for t in range(args.T)]
    Z_list = [dgps.data_target['feature_train'][:, dgps.dim_linear:]] + [dgps.data_source['feature_train'][t][:, dgps.dim_linear:] for t in range(args.T)]
    Y_list = [dgps.data_target['label_train']] + dgps.data_source['label_train']
    res_linear_rep = model_averaging.linear_rep(Y_list, X_list, Z_list)
    est_beta_gamma = np.concatenate((res_linear_rep['beta_hat'], res_linear_rep['alpha_hat']))
    
    test_loss = np.mean((dgps.data_target['feature_test'] @ est_beta_gamma - dgps.data_target['label_test'][:,0])**2)
    test_perf['linear-rep'] = test_loss
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - res_linear_rep['beta_hat'])
    est_perf['linear-rep'] = est_error
    if args.verbose: print(model_color['linear-rep'] + f"Model [{'linear-rep'}] report: "
                          f"current test loss = {test_loss}, "
                          f"est error = {est_error}"
                          )

  if 'mtl-pl-2lr' in model_names:
    flag_aux_train = 'default' # 'default' 'joint' 'orthogonal'
    flag_target_train = 'default' # 'default' 'joint'

    # train the shared layers
    if args.verbose: print(f"Train shared layers--------------------")
    patience = 50
    last_update_count = 0
    best_valid_target = np.inf
    for epoch in range(num_epochs_pretrain):
      # call train_loop_mtl() or train_loop_orthognal()
      if flag_aux_train == 'default': # debug: source + target
        train_loss = train_loop_mtl(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_train, criterion, optimizers['mtl-pl-2lr']) # original method
      elif flag_aux_train == 'joint': # joint training
        train_loss = train_loop_mtl(models['mtl-pl-2lr']['aux-nn']+[models['mtl-pl-2lr']['target-nn']], dgps.dataloaders_source_train+[dgps.dataloader_target_train], criterion, optimizers['mtl-pl-2lr']) # original method
      elif flag_aux_train == 'orthogonal':
        # learning rate schedule for the regularization of the shared layers
        if epoch < 10: zlr = 4.0
        elif epoch == 20: zlr = 2.0
        elif epoch == 40: zlr = 1.0
        train_loss = train_loop_orthognal(models['mtl-pl-2lr']['aux-nn'], D_net, dgps.dataloaders_source_train, criterion, optimizers['mtl-pl-2lr'], optimizers['discriminator'], zlr=zlr) # add regularization

      valid_loss = evaluate(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_valid, criterion)
      if epoch % 40 == 0 and args.verbose: 
        print(f"Epoch {epoch}\n--------------------")
        print(f"\ttrain loss: {train_loss}, valid loss: {valid_loss}")
      last_update_count += 1
      if True: # old update method: update the shared network when the source models have better valid loss
        if np.sum(valid_loss) < best_valid['mtl-pl-2lr']:
          best_valid['mtl-pl-2lr'] = np.sum(valid_loss)
          test_loss = evaluate(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_test, criterion)
          # fine tune the target model
          models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
          valid_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_valid, criterion)
          test_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_test, criterion)
          # end debug
          if valid_loss_target < 1.05 * best_valid_target:
            best_shared_layers.load_state_dict(models['mtl-pl-2lr']['target-nn'].shared_layers.state_dict())
            if valid_loss_target <  best_valid_target:
              best_valid_target = valid_loss_target
            if True and args.verbose: 
                print(model_color['mtl-pl-2lr'] + f"Model [{'mtl-pl-2lr'}] source update test loss, "
                  f"best valid loss = {valid_loss}, current test loss = {test_loss}"
                )
                print(model_color['mtl-pl-2lr'] + f"[target model] valid loss: {valid_loss_target}, test loss: {test_loss_target}")
          else:
            if args.verbose: print(Fore.WHITE + f"Model [{'mtl-pl-2lr'}] source,"
                  f"valid loss = {valid_loss}, current test loss = {test_loss}"
            )
            if args.verbose: print(Fore.WHITE + f"[target model] valid loss: {valid_loss_target}, test loss: {test_loss_target}")
          last_update_count = 0
      else: # new update method: update the shared network when the target model has better valid loss (may suffer from overfitting)
        # fine tune the target model
        models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
        valid_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_valid, criterion)
        # check valid loss
        if valid_loss_target < best_valid_target:
          best_valid_target = valid_loss_target
          # debug: print the test loss in target data          
          test_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_test, criterion)
          if args.verbose: print(model_color['mtl-pl-2lr'] + f"target model valid loss: {valid_loss_target}, test loss: {test_loss_target}")
          # end debug
          test_loss = evaluate(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_test, criterion)
          best_shared_layers.load_state_dict(models['mtl-pl-2lr']['target-nn'].shared_layers.state_dict())
          if args.verbose: print(Fore.MAGENTA + f"Source Model [{'mtl-pl-2lr'}] update target test loss, "
            f"best valid loss = {valid_loss_target}, current test loss = {test_loss_target}, current source test loss = {test_loss}"
          )
          last_update_count = 0

      # patience termination
      if last_update_count > patience and max(valid_loss) < 0.5:
        # early stop
        break

    # train the target model
    if args.verbose: print(f"Train target model---------------------")
    # initialize the target model with the best shared layers
    models['mtl-pl-2lr']['target-nn'].shared_layers.load_state_dict(best_shared_layers.state_dict())
    if flag_target_train == 'default':
      # only fine tune the last layer for target model
      train_loss_target = models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
    elif flag_target_train == 'joint':
      # fine tune the shared layer + last layer (may suffer from overfitting)
      for epoch in range(100):
        train_loss_target = train_loop(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_train, criterion, optimizers['mtl-pl-2lr'], fine_tune=True)
        valid_loss = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_valid, criterion)
        if valid_loss < best_valid_target:
          best_valid_target = valid_loss
          test_loss = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_test, criterion)
          test_perf['mtl-pl-2lr'] = test_loss
          est_error = np.linalg.norm(dgps.params['beta'][:,0] - models['mtl-pl-2lr']['target-nn'].linear_output.weight.detach().numpy()[0, :dgps.dim_linear])
          est_perf['mtl-pl-2lr'] = est_error
          best_shared_layers.load_state_dict(models['mtl-pl-2lr']['target-nn'].shared_layers.state_dict())
          if args.verbose: print(model_color['mtl-pl-2lr'] + f"Target Model [{'mtl-pl-2lr'}] update test loss, "
                                      f"best valid loss = {valid_loss}, current test loss = {test_loss}, "
                                      f"est error = {est_error}"
                                      )
      # final fine tune the target model
      models['mtl-pl-2lr']['target-nn'].shared_layers.load_state_dict(best_shared_layers.state_dict())
      train_loss_target = models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
    # estimate the noise level
    sigma2_hat = train_loss_target.detach().numpy()
    # evaluate the target model on test data and report the performance
    test_loss = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_test, criterion)
    test_perf['mtl-pl-2lr'] = test_loss
    est_mlt = models['mtl-pl-2lr']['target-nn'].linear_output.weight.detach().numpy()[0, :dgps.dim_linear]
    est_error = np.linalg.norm(dgps.params['beta'][:,0] - est_mlt)
    est_perf['mtl-pl-2lr'] = est_error
    if args.verbose: print(model_color['mtl-pl-2lr'] + f"Model [{'mtl-pl-2lr'}] report: "
                      f"final test loss = {test_loss}, "
                      f"est error = {est_error}"
                    )

    # [begin] estimate the variance of the estimator
    X_train = dgps.data_target['feature_train'][:, :dgps.dim_linear]
    Z_train_torch = torch.tensor(dgps.data_target['feature_train'][:, dgps.dim_linear:], requires_grad=False).float()
    H_hat = models['mtl-pl-2lr']['target-nn'].shared_layers(Z_train_torch).detach().numpy()
    J_hat = X_train.T @ (np.eye(dgps.n_train_target) - H_hat @ np.linalg.inv(H_hat.T @ H_hat) @ H_hat.T) @ X_train / dgps.n_train_target
    est_var = sigma2_hat * np.linalg.inv(J_hat) / dgps.n_train_target # the estimated variance of the estimator with known noise level
    if False and args.verbose: print(f"estimated variance: {est_var}")
    # [end] estimate variance

  # train the target model using different methods
  patience = 50
  last_update_count = {name: 0 for name in model_names}
  for epoch in range(num_epochs):
    if epoch % 40 == 0:
      if args.verbose: print(f"Epoch {epoch}\n--------------------")
    for model_name in model_names:
      if last_update_count[model_name] > patience: continue
      if model_name not in ['vanilla-orginal', 'vanilla-pl-1lr', 'dropout-vanilla-pl-1lr', 'vanilla-pl-2lr', 'dropout-vanilla-pl-2lr', 'stl-pl-1lr', 'dropout-stl-pl-1lr', 'stl-pl-2lr', 'dropout-stl-pl-2lr']: continue
      flag_fine_tune = True if model_name in ['vanilla-pl-2lr', 'dropout-vanilla-pl-2lr', 'stl-pl-2lr', 'dropout-stl-pl-2lr'] else False # 
      train_loss = train_loop(models[model_name], dgps.dataloader_target_train, criterion, optimizers[model_name], fine_tune=flag_fine_tune)
      valid_loss = evaluate(models[model_name], dgps.dataloader_target_valid, criterion)
      if valid_loss < best_valid[model_name]:
        best_valid[model_name] = valid_loss
        test_loss = evaluate(models[model_name], dgps.dataloader_target_test, criterion)
        test_perf[model_name] = test_loss
        est_error = np.linalg.norm(dgps.params['beta'][:,0] - models[model_name].linear_output.weight.detach().numpy()[0, :dgps.dim_linear]) if model_name != 'vanilla-orginal' else 0
        est_perf[model_name] = est_error
        if args.verbose: print(model_color[model_name] + f"Model [{model_name}] update test loss, "
                                    f"best valid loss = {valid_loss}, current test loss = {test_loss}, "
                                    f"est error = {est_error}"
                                    )
        last_update_count[model_name] = 0
      last_update_count[model_name] += 1
      if epoch % 40 == 0:
        if args.verbose: print(Fore.BLACK + f"Model [{model_name}]: Epoch {epoch}/{num_epochs}, Train MSE: {train_loss}, Valid MSE: {valid_loss}")
  pred_result = np.zeros((1, len(model_names)))
  est_result = np.zeros((1, len(model_names)))

  for i, name in enumerate(model_names):
    est_result[0, i] = est_perf[name]
    pred_result[0, i] = test_perf[name]

  return pred_result, est_result


# %% generate data
noise_level = 0.3 # 0.3 1.0
"""
Description of the experiments:
  - coef_flag     : 'homogeneous' 'heterogeneous'
      This controls the coefficients in the final linear model.
      If 'homogeneous', then the coefficients are (randomly drawn once) kept the same for all the datasets.
      If 'heterogeneous', then the coefficients are (randomly drawn T+1 times) different for all the datasets.
  - feature_flag  : 'iid-uniform' 'iid-normal' 'compound-normal'
      This controls the features' distribution. 
      If 'iid-uniform', then each dimension of features are drawn from [0-1] uniform distribution, all components are independent. 
      If 'iid-normal', then the features are drawn from N(0, I) normal distribution.
      If 'compound-normal', then the features are drawn from N(0, Sigma) normal distribution where Sigma follows the compound correlated setting. 
  - rep_flag      : 'linear-factor' # 'linear' 'linear-factor' 'additive' 'additive-factor' 'deep'
      This controls the representation function.
      If 'linear', then the representation function is linear in the features, i.e., h(Z) = Z.
      If 'linear-factor', then the representation function is linear in the features, and the model follows a factor model, i.e., h(Z) = g(ZA) = sum_i g_i(ZA_i). 
      If 'additive', then the representation function is additive in the features, i.e., h(Z) = sum_i g_i(Z_i).
      If 'additive-factor', then the representation function is additive in the features, and the model follows a factor structure, i.e., h(Z) = sum_i g_i(ZA_i). 
      If 'deep', then the representation function seems like a deep network.
"""

# homogeneous_design_list = [1, 3, 5, 7, 9, 11]
# heterogeneous_design_list = [2, 4, 6, 8, 10, 12]

# iid_uniform_design_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# conpound_normal_design_list = [11, 12]

# linear_rep_design_list = [1, 2]
# linear_factor_rep_design_list = [3, 4]
# additive_rep_design_list = [5, 6]
# additive_factor_rep_design_list = [7, 8]
# deep_rep_design_list = [9, 10, 11, 12]

seed = None

if args.exp_id == 1:
  coef_flag = 'homogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'linear'
elif args.exp_id == 2:
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'linear'
elif args.exp_id == 3:
  coef_flag = 'homogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'linear-factor'
elif args.exp_id == 4:
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'linear-factor'
elif args.exp_id == 5:
  coef_flag = 'homogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'additive'
elif args.exp_id == 6:
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'additive'
  seed = 2
elif args.exp_id == 7:
  coef_flag = 'homogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'additive-factor'
elif args.exp_id == 8:
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'additive-factor'
  # seed = 10
elif args.exp_id == 9:
  coef_flag = 'homogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'deep'
  seed = 6
elif args.exp_id == 10:
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'deep'
  seed = 2
elif args.exp_id == 11:
  coef_flag = 'homogeneous'
  feature_flag = 'compound-normal'
  rep_flag = 'deep'
elif args.exp_id == 12:
  coef_flag = 'heterogeneous'
  feature_flag = 'compound-normal'
  rep_flag = 'deep'
elif args.exp_id == 13: # this is for debug
  coef_flag = 'heterogeneous'
  feature_flag = 'iid-uniform'
  rep_flag = 'additive'
  seed = 1
else:
  raise ValueError(f"Invalid experiment id: {args.exp_id}")

dgps = DGPs(T=args.T, dim_linear=args.q, dim_nonlinear=args.p, dim_rep=args.r_dgp, rep_flag=rep_flag, coef_flag=coef_flag, feature_flag=feature_flag, n_train_source=args.n_source, n_train_target=args.n_target, noise_level=noise_level, verbose=args.verbose, seed=args.seed)
# dgps.params['betas'] = [np.zeros((dgps.dim_linear, 1)) for t in range(args.T)]
# dgps.params['beta'] = np.zeros((dgps.dim_linear, 1))
if args.verbose: print(dgps)

if False:
  # tune the n_knots for spline methods
  best_n_knots = dgps.tune_knots()
  if args.verbose: print(f"\tthe seleted best n_knots: {best_n_knots}")
  # sample from the dgp
  dgps.sample(best_n_knots, seed=args.seed) # generate samples
else:
  dgps.sample(seed = args.seed) # generate samples



# %% prepare model parameters and define models
num_epochs = 400
num_epochs_pretrain = 400
if args.exp_id in [5, 6]:
  # additive model
  args.depth = 4
  args.width = 300
elif args.exp_id in [7, 8]:
  args.depth = 4
  args.width = 300
elif args.exp_id in [9, 10]:
  # additive-factor and deep model
  args.depth = 6
  args.width = 500
else:
  args.depth = 4
  args.width = 300

torch.manual_seed(42) # fix the random seed for neural networks
vanilla_nn_model_orgigin = RegressionNN(d=args.p+args.q, depth=args.depth, width=args.width)
vanilla_nn_model = PartialRegressionNN(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width, bias_last_layer=True)
dropout_nn_model = PartialRegressionNN(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width, input_dropout=True, dropout_rate=0.6, bias_last_layer=True)
shared_nn = SharedNN(dim_input=args.p, dim_rep=args.r, width=args.width, depth=args.depth)
nn_models_aux = [PartialLinearRegression(dim_linear=args.q, shared_layers=shared_nn) for t in range(args.T)]
nn_model_target = PartialLinearRegression(dim_linear=args.q, shared_layers=shared_nn)
stl_1lr_model = PartialLinearRegression(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width)
dropout_stl_1lr_model = PartialLinearRegression(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width, input_dropout=True, dropout_rate=0.6)
stl_2lr_model = PartialLinearRegression(dim_input=args.p+args.q, dim_linear=args.q, dim_rep=args.r, depth=args.depth, width=args.width)
dropout_stl_2lr_model = PartialLinearRegression(dim_input=args.p+args.q, dim_linear=args.q, dim_rep=args.r, depth=args.depth, width=args.width, input_dropout=True, dropout_rate=0.6)
vanilla_pl_1lr_model = PartialRegressionNN(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width, bias_last_layer=True)
dropout_vanilla_pl_1lr_model = PartialRegressionNN(dim_input=args.p+args.q, dim_linear=args.q, depth=args.depth, width=args.width, input_dropout=True, dropout_rate=0.6, bias_last_layer=True)
D_net = Discriminator(ndim = args.r) # Discriminator for representation regularization of our method
models = {
  'vanilla-orginal': vanilla_nn_model_orgigin, 
  'vanilla-pl-1lr': vanilla_pl_1lr_model,
  'dropout-vanilla-pl-1lr': dropout_vanilla_pl_1lr_model,
  'vanilla-pl-2lr': vanilla_nn_model, 
  'dropout-vanilla-pl-2lr': dropout_nn_model, 
  'stl-pl-1lr': stl_1lr_model,
  'dropout-stl-pl-1lr': dropout_stl_1lr_model,
  'stl-pl-2lr': stl_2lr_model, 
  'dropout-stl-pl-2lr': dropout_stl_2lr_model,
  'mtl-pl-2lr': {
    'aux-nn': nn_models_aux, 
    'target-nn': nn_model_target
  }
}

if args.verbose: print(nn_model_target)

learning_rate_linear = 0.5
learning_rate_nonlinear = args.lr # 1e-3 1e-4
optimizers = {}
for method_name, model_x in models.items():
  # Assign different learning rates
  if method_name in ['vanilla-orginal', 'vanilla-pl-1lr', 'dropout-vanilla-pl-1lr', 'stl-pl-1lr', 'dropout-stl-pl-1lr']:
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate_nonlinear)
  elif method_name in ['vanilla-pl-2lr', 'dropout-vanilla-pl-2lr']:
    optimizer_x = torch.optim.Adam([{
      'params': model_x.nonlinear_layers.parameters(), 
      'lr': learning_rate_nonlinear
    }])
  elif method_name in ['stl-pl-2lr', 'dropout-stl-pl-2lr']:
    optimizer_x = torch.optim.SGD([
      {
        # 'params': model_x.parameters(), 
        # 'lr': learning_rate_nonlinear,
        'params': model_x.shared_layers.parameters(),
        'lr': learning_rate_nonlinear,
        'momentum': 0.9,
      },
      # {
      #   'params': model_x.linear_output.parameters(), 
      #   'lr': 0.5,
      # }
    ])
  elif method_name in ['mtl-pl-2lr']:
    if False:
      optimizer_x = torch.optim.Adam([{
        'params': shared_nn.shared_layers.parameters(), 
        'lr': learning_rate_nonlinear
      }
      ])
    else:
      optimizer_x = torch.optim.SGD([{
        'params': shared_nn.shared_layers.parameters(), 
        'lr': learning_rate_nonlinear,
        'momentum': 0.9,
      }])
    optimizer_source_linear = [torch.optim.Adam(nn_model.linear_output.parameters(), learning_rate_linear) for nn_model in model_x['aux-nn']]
  optimizers[method_name] = optimizer_x

optimizers['discriminator'] = torch.optim.Adam(D_net.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# %% Evaluation
model_names = ['mtl-pl-2lr', 'ols-oracle', 'ols-spline', 'trans-lasso-spline', 'map-spline', 'meta-analysis-spline', 'pooled-regression-spline', 'stl-pl-2lr', 'ols-linear']
# ['mtl-pl-2lr', 'ols-oracle', 'ols-spline', 'trans-lasso-spline', 'map-spline', 'meta-analysis-spline', 'pooled-regression-spline', 'stl-pl-2lr', 'ols-linear']
test_l2_error, est_error = joint_train(model_names)
# add a column for the test error, and a column for the est error, content is the seed
test_l2_error = np.concatenate((test_l2_error, np.array([args.seed]).reshape(1,1)), axis=1)
est_error = np.concatenate((est_error, np.array([args.seed]).reshape(1,1)), axis=1)
if len(args.record_dir) > 0:
  res_pred_file = open(args.record_dir + f"/exp{args.exp_id}_n{args.n_source}_r{args.r}_p{args.p}_pred.csv","a")
  res_est_file = open(args.record_dir + f"/exp{args.exp_id}_n{args.n_source}_r{args.r}_p{args.p}_est.csv","a")
  np.savetxt(res_pred_file, test_l2_error, delimiter=",")
  np.savetxt(res_est_file, est_error, delimiter=",")
  res_pred_file.close()
  res_est_file.close()
end_time = time.time()
print(f"Case with Exp_id: {args.exp_id}, T = {args.T}, n_source = {args.n_source}, n_target = {args.n_target}, p = {args.p}, q = {args.q}, r = {args.r}, seed = {args.seed} done: time = {end_time - start_time} secs")
if args.verbose: print(f"Methods: {model_names}\nTest MSE: {test_l2_error}\nEst error: {est_error}")

# %% Results
