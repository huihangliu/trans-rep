"""
update:
2024.01.24: add generalized linear model to the partial linear model. 
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import platform
system_type = platform.system().lower()

class NonlinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim=1, width=300, depth=4, input_dropout=False, dropout_rate=0.0, bias_last_layer=True):
    super(NonlinearRegression, self).__init__()
    self.use_input_dropout = input_dropout
    self.input_dropout = nn.Dropout(p=dropout_rate)
    dim_rep = 5
    widths = [width] * (depth) # previously (depth-1)
    widths.append(dim_rep)
    layers = []
    for i in range(depth+1): # previously depth
      in_features = input_dim if i == 0 else widths[i-1]
      layers.append(nn.Linear(in_features, widths[i]))
      if i != depth: # previously (depth-1)
        layers.append(nn.ReLU())
    self.shared_layers = nn.Sequential(*layers)

    self.output_layer = nn.Linear(widths[-1], output_dim, bias=bias_last_layer)

  def forward(self, x, is_training=False):
    if self.use_input_dropout and is_training:
      x = self.input_dropout(x)
    h = self.shared_layers(x)
    return self.output_layer(h)

class RegressionNN(nn.Module):
  '''
    A class to implement standard relu nn for regression

    ...
    Attributes
    ----------
    use_input_dropout : bool
      whether to use dropout (True) or not (False) in the input layer
    input_dropout : nn.module
      pytorch module of input dropout
    relu_stack : nn.module
      pytorch module to implement relu neural network

    Methods
    ----------
    __init__(x, is_training=False)
      Initialize the module
    forward()
      Implementation of forwards pass
  '''
  def __init__(self, d, depth, width, input_dropout=False, dropout_rate=0.0):
    '''
      Parameters
      ----------
      d : int
        input dimension
      depth : int
        the number of hidden layers of relu network
      width : int
        the number of units in each hidden layer of relu network
      input_dropout : bool, optional
        whether to use input dropout in the input layer (True)
      dropout_rate: float, optional
        the dropout rate for the input dropout
    '''
    super(RegressionNN, self).__init__()
    self.use_input_dropout = input_dropout
    self.input_dropout = nn.Dropout(p=dropout_rate)

    if depth == 0:
      relu_nn = [('linear1', nn.Linear(d, 1))]
    else: 
      relu_nn = [('linear1', nn.Linear(d, width)), ('relu1', nn.ReLU())]
      for i in range(depth - 1):
        relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
        relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
      relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

    self.relu_stack = nn.Sequential(
      OrderedDict(relu_nn)
    )

  def forward(self, x, is_training=False):
    '''
      Parameters
      ----------
      x : torch.tensor
        the (n x p) matrix of the input
      is_training : bool
        whether the forward pass is used in the training (True) or not,
        used for dropout module

      Returns
      ----------
      pred : torch.tensor
        (n, 1) matrix of the prediction
    '''
    if self.use_input_dropout and is_training:
      x = self.input_dropout(x)
    pred = self.relu_stack(x)
    return pred

class SharedNN(nn.Module):
  def __init__(self, dim_input, dim_rep=5, width=300, depth=4, flag_batchnorm=False):
    super(SharedNN, self).__init__()
    widths = [width] * (depth) # previously (depth-1)
    widths.append(dim_rep)
    self.width = width
    self.widths = widths
    self.depth = depth
    self.dim_input = dim_input
    self.dim_rep = dim_rep

    layers = []
    for i in range(depth+1): # previously depth
      in_features = self.dim_input if i == 0 else widths[i-1]
      layers.append(nn.Linear(in_features, widths[i]))
      if i != depth: # previously (depth-1)
        if flag_batchnorm: 
          # print("Using batch normalization in the shared layers\n\n\nUsing batch normalization in the shared layers")
          layers.append(nn.BatchNorm1d(widths[i]))  # batch normalization may improve the performance
        layers.append(nn.ReLU())
    self.shared_layers = nn.Sequential(*layers)

  def forward(self, x):
    h = self.shared_layers(x)
    return h

class PartialLinearRegression(nn.Module):
  def __init__(self, dim_input=None, dim_linear=0, dim_rep=None, dim_output=1, width=300, depth=4, shared_layers=None, input_dropout=False, dropout_rate=0.0, bias_last_layer=False, flag_batchnorm=False, flag_binary_outcome=False):
    """
      Parameters
      ----------
      dim_input : int
        the dimension of the input
      dim_linear : int
        the dimension of the linear part of the input (at the begining 0:dim_linear)
      dim_output : int
        the dimension of the output
      width : int
        the width of the hidden layers
      depth : int
        the depth of the hidden layers
      input_dropout : bool, optional
        whether to use input dropout in the input layer (True)
      dropout_rate: float, optional
        the dropout rate for the input dropout
    """
    super(PartialLinearRegression, self).__init__()
    self.dim_linear = dim_linear
    self.dim_input = dim_input
    if dim_input is not None:
      self.dim_nonlinear = dim_input - dim_linear
    else:
      self.dim_nonlinear = None
    self.use_input_dropout = input_dropout
    self.input_dropout = nn.Dropout(p=dropout_rate)
    self.shared_layers = shared_layers
    self.flag_batchnorm = flag_batchnorm
    self.flag_binary_outcome = flag_binary_outcome

    if shared_layers is None:
      self.dim_rep = dim_rep if dim_rep is not None else 5
      self.depth = depth
      self.width = width
      widths = [width] * (depth-1)
      widths.append(self.dim_rep)
      self.widths = widths
      self.shared_layers = SharedNN(self.dim_nonlinear, self.dim_rep, self.width, self.depth, self.flag_batchnorm)
    else:
      if self.dim_nonlinear is not None and self.shared_layers.dim_input != self.dim_nonlinear:
        raise ValueError("The dimension of the input of the shared layers is not the same as the dimension of the nonlinear part of the input of the partial linear regression model")
      self.dim_nonlinear = self.shared_layers.dim_input
      self.dim_rep = self.shared_layers.dim_rep
      self.width = self.shared_layers.width
      self.widths = self.shared_layers.widths
      self.depth = self.shared_layers.depth

    # linear output layer
    self.linear_output = nn.Linear(self.dim_rep + dim_linear, dim_output, bias=bias_last_layer)
    # sigmoid output layer
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, is_training=False):
    feature_linear = x[:, :self.dim_linear]
    feature_nonlinear = x[:, self.dim_linear:]

    if self.use_input_dropout and is_training: feature_nonlinear = self.input_dropout(feature_nonlinear)

    h = self.shared_layers(feature_nonlinear)

    combined_layer = torch.cat((feature_linear, h), dim=1) # linear part + nonlinear part

    if self.flag_binary_outcome:
       # debug: softmax
      return self.sigmoid(self.linear_output(combined_layer))
    else:
      return self.linear_output(combined_layer)

  def fine_tune(self, dataloader):
    """
    TODO: 尝试同时训练 shared_layers 和 linear_output, 用 early_stopping 可以防止在 target model 上的过拟合问题. 可能会比 OLS 表现更好. 
    可以使用惩罚的方法来避免 fine-tune 时对表示层参数的过度修改. 
    """
    all_X = []
    all_Y = []

    for batch in dataloader:
      X, Y = batch
      all_X.append(X)
      all_Y.append(Y)

    # Concatenate all batches
    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)

    feature_linear = all_X[:, :self.dim_linear]
    feature_nonlinear = all_X[:, self.dim_linear:]
    feature_combined = torch.cat((feature_linear, self.shared_layers(feature_nonlinear)), dim=1) # linear part + nonlinear part

    if self.flag_binary_outcome:
      # used logistic regression to regress the feature_combined to all_Y and obtain the coefficients, then make prediction using the coefficients
      # Loss and Optimizer
      criterion = nn.BCELoss()
      optimizer = optim.Adam(self.linear_output.parameters(), lr=1.0)

      # Train the Network
      # Assuming you have a DataLoader named train_loader
      outputs = self.sigmoid(self.linear_output(feature_combined))
      loss = criterion(outputs, all_Y)
      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Predictions using the coefficients
      predictions = self.sigmoid(self.linear_output(feature_combined))
      # Calculate MSE
      error = torch.mean((all_Y - predictions) ** 2)
    else:
      # Compute OLS coefficients
      if torch.__version__ <= '1.7.1':
        try:
        # A should be positive defined
          A = feature_combined.T @ feature_combined + torch.eye(feature_combined.shape[1]) * 1e-6
          L = torch.cholesky(A, upper=False)
          coefficients = torch.cholesky_solve(feature_combined.T @ all_Y, L, upper=False)
        except:
          coefficients = torch.inverse(feature_combined.T @ feature_combined + torch.eye(feature_combined.shape[1]) * 1e-6) @ feature_combined.T @ all_Y
      else:
        coefficients = torch.linalg.lstsq(feature_combined, all_Y).solution
      # assign the coefficients to the weight of the last layer
      self.linear_output.weight.data.copy_(coefficients.T)
      # if self.dim_rep == 1: self.linear_output.weight.data[0, self.dim_linear:] = 1.0 # this is not good, because, we do not known whether the coefficient of rep is shared or not. We should not assume that it is shared in our framework. 
      # Predictions using the coefficients
      predictions = feature_combined @ coefficients
      # Calculate MSE
      error = torch.mean((all_Y - predictions) ** 2)
    return error

class PartialRegressionNN(nn.Module):
  '''
    A class to implement standard relu nn for partial linear regression without representation learning

    ...
    Attributes
    ----------

    Methods
    ----------
    __init__(x, is_training=False)
      Initialize the module
    forward()
      Implementation of forwards pass
  '''
  def __init__(self, dim_input, dim_linear=0, depth=4, width=300, input_dropout=False, dropout_rate=0.0, bias_last_layer=True):
    '''
      Parameters
      ----------
      dim_input : int
        input dimension
      dim_linear : int
        the dimension of the linear part of the input (at the begining 0:dim_linear)
      depth : int
        the number of hidden layers of relu network
      width : int
        the number of units in each hidden layer of relu network
      input_dropout : bool, optional
        whether to use input dropout in the input layer (True)
      dropout_rate: float, optional
        the dropout rate for the input dropout
      bias_last_layer : bool, optional
        whether to use bias in the last layer (True) or not (False)
    '''
    super(PartialRegressionNN, self).__init__()
    self.dim_input = dim_input
    self.dim_linear = dim_linear
    self.dim_nonlinear = dim_input - dim_linear
    self.use_input_dropout = input_dropout
    self.input_dropout = nn.Dropout(p=dropout_rate)

    relu_nn = [('linear1', nn.Linear(self.dim_nonlinear, width)), ('relu1', nn.ReLU())]
    for i in range(depth - 1):
      relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
      relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

    self.nonlinear_layers = nn.Sequential(
      OrderedDict(relu_nn)
    )
    self.linear_output = nn.Linear(width+dim_linear, 1, bias=bias_last_layer)

  def forward(self, x, is_training=False):
    '''
      Parameters
      ----------
      x : torch.tensor
        the (n x (p+q)) matrix of the input
      is_training : bool
        whether the forward pass is used in the training (True) or not,
        used for dropout module

      Returns
      ----------
      pred : torch.tensor
        (n, 1) matrix of the prediction
    '''
    feature_linear = x[:, :self.dim_linear]
    feature_nonlinear = x[:, self.dim_linear:]

    if self.use_input_dropout and is_training: feature_nonlinear = self.input_dropout(feature_nonlinear)

    h = self.nonlinear_layers(feature_nonlinear)
    combined_layer = torch.cat((feature_linear, h), dim=1)  # linear part + nonlinear part
    
    return self.linear_output(combined_layer)

  def fine_tune(self, dataloader):
    all_X = []
    all_Y = []

    for batch in dataloader:
      X, Y = batch
      all_X.append(X)
      all_Y.append(Y)

    # Concatenate all batches
    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)

    feature_linear = all_X[:, :self.dim_linear]
    feature_nonlinear = all_X[:, self.dim_linear:]
    feature_combined = torch.cat((feature_linear, self.nonlinear_layers(feature_nonlinear)), dim=1) # linear part + nonlinear part

    # Compute OLS coefficients using torch.linalg.lstsq
    coefficients = torch.linalg.lstsq(feature_combined, all_Y).solution
    self.linear_output.weight.data = coefficients.T # assign the coefficients to the weight of the last layer

    # Predictions using the coefficients
    predictions = feature_combined @ coefficients
    # Calculate MSE
    mse = torch.mean((all_Y - predictions) ** 2)

    return mse

# Discriminator class: A neural network for distinguishing real data from generated data
class Discriminator(nn.Module):
  def __init__(self, ndim):
    super(Discriminator, self).__init__()
    
    # Defining the model with linear layers and leaky ReLU activations
    self.model = nn.Sequential(
      nn.Linear(ndim, 16), # orinal 16
      nn.LeakyReLU(0.2),
      nn.Linear(16, 8), # original 8
      nn.LeakyReLU(0.2),
      nn.Linear(8, 1)
    )

  def forward(self, X):
    # Forward pass through the network
    validity = self.model(X)
    return validity
