"""
date:   2023.02.18
author: huihang@mail.ustc.edu.cn
note:
  直接输出估计量的加权偏差, 估计方差(两个版本), 然后根据偏差绘制直方图与正态分布曲线进行比较. 
"""

# %% Import packages
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.NN_Models import PartialRegressionNN, PartialLinearRegression, SharedNN, RegressionNN, Discriminator
from utils.Data_Generator import DGPs
from utils.Trans_MA import ModelAveraging
from utils.Trans_Lasso import trans_lasso
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
parser.add_argument("--T", help="number of source dataset", type=int, default=6)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=1)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=13)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--verbose", help="print training information", type=int, default=True)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# %% Some functions about data generation and evaluation
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
    loss_sum_aux.backward()
    optimizer_pre_train.step()

    # fine tune the last layer
    loss_aux = np.array([0.0]*T)
    for t, dataloader in enumerate(dataloaders_aux):
      loss_aux[t] = models_source[t].fine_tune(dataloader)

  return [loss_aux[t]/len(dataloaders_aux[t]) for t in range(T)]

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

  # train the shared layers
  if args.verbose: print(f"Train shared layers--------------------")
  patience = 100
  last_update_count = 0
  for epoch in range(num_epochs_pretrain):
    train_loss = train_loop_mtl(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_train, criterion, optimizers['mtl-pl-2lr']) # original method
    valid_loss = evaluate(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_valid, criterion)
    if epoch % 40 == 0 and args.verbose: 
      print(f"Epoch {epoch}\n--------------------")
      print(f"\ttrain loss: {train_loss}, valid loss: {valid_loss}")
    last_update_count += 1
    if np.sum(valid_loss) < best_valid['mtl-pl-2lr']:
      best_valid['mtl-pl-2lr'] = np.sum(valid_loss)
      test_loss = evaluate(models['mtl-pl-2lr']['aux-nn'], dgps.dataloaders_source_test, criterion)
      # fine tune the target model
      models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
      valid_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_valid, criterion)
      test_loss_target = evaluate(models['mtl-pl-2lr']['target-nn'], dgps.dataloader_target_test, criterion)
      if args.verbose: print(model_color['mtl-pl-2lr'] + f"[target model] valid loss: {valid_loss_target}, test loss: {test_loss_target}")
      # end debug
      best_shared_layers.load_state_dict(models['mtl-pl-2lr']['target-nn'].shared_layers.state_dict())
      if True and args.verbose: print(model_color['mtl-pl-2lr'] + f"Model [{'mtl-pl-2lr'}] update test loss, "
        f"best valid loss = {valid_loss}, current test loss = {test_loss}"
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
  # only fine tune the last layer for target model
  train_loss_target = models['mtl-pl-2lr']['target-nn'].fine_tune(dgps.dataloader_target_train)
  # estimate the noise level
  sigma2_hat = train_loss_target.detach().numpy() * dgps.n_train_target / (dgps.n_train_target - dgps.dim_linear)
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
  est_var = sigma2_hat * np.linalg.inv(J_hat) / (dgps.n_train_target-dgps.dim_linear) # the estimated variance of the estimator with unknown noise level
  alpha = np.ones((dgps.dim_linear, 1))
  alpha = alpha / np.linalg.norm(alpha)
  est_var_vec = alpha.T @ est_var @ alpha
  if args.verbose: print(f"estimated variance: {est_var}")
  # [end] estimate variance

  pred_result = np.zeros((1, len(model_names)))
  est_result = np.zeros((1, len(model_names)))

  for i, name in enumerate(model_names):
    est_result[0, i] = est_perf[name]
    pred_result[0, i] = test_perf[name]

  return pred_result, est_result, np.mean(est_error), est_var_vec


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
  feature_flag = 'compound-normal'
  rep_flag = 'additive'
  seed = 1
else:
  raise ValueError(f"Invalid experiment id: {args.exp_id}")

dgps = DGPs(T=args.T, dim_linear=args.q, dim_nonlinear=args.p, dim_rep=args.r_dgp, rep_flag=rep_flag, coef_flag=coef_flag, feature_flag=feature_flag, n_train_source=args.n_source, n_train_target=args.n_target, noise_level=noise_level, verbose=args.verbose, seed=seed)
if args.verbose: print(dgps)

dgps.sample(seed=args.seed) # generate samples



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
    optimizer_x = torch.optim.Adam([
      {
        'params': model_x.parameters(), 
        'lr': learning_rate_nonlinear
        # 'params': model_x.shared_layers.parameters(),
        # 'lr': learning_rate_nonlinear
      },
      # {
      #   'params': model_x.linear_output.parameters(), 
      #   'lr': 0.5,
      # }
    ])
  elif method_name in ['mtl-pl-2lr']:
    optimizer_x = torch.optim.Adam([{
      'params': shared_nn.shared_layers.parameters(), 
      'lr': learning_rate_nonlinear
    }
    ])
    optimizer_source_linear = [torch.optim.Adam(nn_model.linear_output.parameters(), learning_rate_nonlinear) for nn_model in model_x['aux-nn']]
  optimizers[method_name] = optimizer_x

optimizers['discriminator'] = torch.optim.Adam(D_net.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# %% Evaluation
model_names = ['mtl-pl-2lr']
# ['mtl-pl-2lr', 'ols-oracle', 'ols-spline', 'trans-lasso-spline', 'map-spline', 'meta-analysis-spline', 'pooled-regression-spline', 'stl-pl-2lr', 'ols-linear']
test_l2_error, est_error, bias, var = joint_train(model_names)
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
