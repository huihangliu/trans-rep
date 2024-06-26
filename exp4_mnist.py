"""
author: Huihang Liu
date: 2024-02-10
descirption: This is a simple regression model for MNIST dataset. 
[Version 1] Two model is implemented, one is fully connected neural network, the other is convolutional neural network. The second one has much better performance. Accuracy on test data: 0.9304
[Version 2] Arithmetic Operations for MNIST. This is a specially created dataset. Let $y = \beta x + \gamma h(z) + \epsilon$, where $x$ is generated random scalar in $R$, and $z$ is MNIST image, $\beta$ and $\gamma$ are random variables/vectors, and $\epsilon$ is a random noise. The task is to learn the representaion $h()$ and predict $\beta$. We holp that the representation $h()$ learns the label of image $z$. 
[Version 3] This code implement the multi-task learning
"""
# %% 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from colorama import Fore
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

for rep_idx in range(1,6):
  seed = 2024 + rep_idx
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark=False
  output_file = 'output' + str(seed) + '.log'

  verbose = False
  T = 10
  # Define the device to train on
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #%% Network, model, optimizer, loss function
  class RepresentNN_modify(nn.Module):
    def __init__(self, struc='cnn'):
      super(RepresentNN, self).__init__()

      self.model_fcnn = nn.Sequential(
        nn.Linear(28*28, 512),  # MNIST images are 28x28
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 1)       # Output layer for regression
      )

      self.model_cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=256),
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 10),       # representation dimensional is 10
        nn.LogSoftmax(dim=1)      # softmax for classification
      )

      self.struc = struc
      if struc == 'cnn':
        self.model = self.model_cnn
      elif struc == 'fc':
        self.model = self.model_fcnn
      else:
        raise ValueError('struc should be either "cnn" or "fcnn"')

    def forward(self, x):
      if self.struc == 'fcnn':
        x = x.view(-1, 28*28)       # Flatten the images for fully connected neural network

      output = self.model(x)
      return output

  class RepresentNN(nn.Module):
    def __init__(self):
      super(RepresentNN, self).__init__()
      self.model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=256),
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 10),     # representation dimensional is 10
        nn.LogSoftmax(dim=1)    # softmax for classification
      )

    def forward(self, x):
      output = self.model(x)
      return output

  class RegressionNN(nn.Module):
    def __init__(self, dim_linear, dim_rep, rep_model=None, bias_last_layer=False):
      super(RegressionNN, self).__init__()

      self.rep_model = RepresentNN() if rep_model is None else rep_model
      self.dim_linear = dim_linear
      self.dim_rep = dim_rep

      self.linear_layers = nn.Sequential(
        nn.Linear(dim_linear+dim_rep, 1, bias=bias_last_layer) # Output layer for regression
      )
      self.rep_linear = nn.Linear(10, 1, bias=bias_last_layer)

    def forward(self, x, z):
      output_rep = self.rep_model(z)
      output_rep = torch.exp(output_rep) # convert the value into probability
      # output_rep = self.rep_linear(output_rep)
      output = self.linear_layers(torch.cat([x, output_rep], dim=1))
      return output

    def fine_tune(self, dataloader, device):
      with torch.no_grad():
        all_X = []
        all_Z = []
        all_Y = []

        for X, Z, Y in dataloader:
          all_X.append(X)
          all_Z.append(Z)
          all_Y.append(Y)

        # Concatenate all batches
        feature_linear = torch.cat(all_X, dim=0)
        feature_nonlinear = torch.cat(all_Z, dim=0)
        all_Y = torch.cat(all_Y, dim=0)

        rep_model = self.rep_model.to('cpu')
        feature_combined = torch.cat((feature_linear, torch.exp(rep_model(feature_nonlinear))), dim=1) # linear part + nonlinear part

        # Compute OLS coefficients using torch.linalg.lstsq
        coefficients = np.linalg.lstsq(feature_combined.to('cpu').detach().numpy(), all_Y.to('cpu').detach().numpy(), rcond=None)[0]
        coefficients = torch.tensor(coefficients).to(device)

        # assign the coefficients to the weight of the last layer
        self.linear_layers[0].weight[0].data.copy_(coefficients[:,0]) # use copy_(), rather than =, to avoid breaking the computation graph
        predictions = feature_combined @ coefficients[:,0].to('cpu')
        
        # Calculate MSE
        error = torch.mean((all_Y[:, 0] - predictions) ** 2)

        # release the memory
        all_X = all_Z = all_Y = None
        feature_linear = feature_nonlinear = feature_combined = None
        rep_model = None
        coefficients = None
      return error

  class SpecialMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, beta=None, gamma=None, noise_level=1.0):
      """
      Args:
          mnist_dataset (Dataset): The original MNIST dataset.
      """
      self.mnist_dataset = mnist_dataset
      self.x = torch.randn(len(mnist_dataset), 1)  # Random scalar x
      self.beta = torch.randn(1) if beta is None else beta # Random variable/vector beta
      self.gamma = torch.randn(1) if gamma is None else gamma # Random variable/vector gamma
      self.epsilon = torch.randn(len(mnist_dataset), 1) * noise_level  # Random noise epsilon

    def __len__(self):
      return len(self.mnist_dataset)

    def __getitem__(self, idx):
      z, label = self.mnist_dataset[idx]  # Get MNIST image and label
      x = self.x[idx]
      y = self.beta * x + self.gamma * label + self.epsilon[idx]  # Compute label y
      return x, z, y

  def train_loop_mtl(models_source, dataloaders_source, loss_fn, optimizer_pre_train, optimizers_linear):
    """
    Train the models using the source datasets
    Notes:
      - We assume that the batch size is the same across different source datasets, so we just loop for the first dataloader
    Parameters:
      models_source: list of models
      dataloaders_source: list of dataloaders
      loss_fn: loss function
      optimizer_pre_train: optimizer
    """
    # loop for source datas
    T = len(models_source) # redefine T
    loss_aux = np.array([0.0]*T)
    for t in range(T):
      models_source[t].train()
      models_source[t].to(device)
    for batch_idx in range(len(dataloaders_source[0])): # len(dataloaders_source[0]) is the number of batches in the first source dataset
      loss_sum_aux = 0
      for t, dataloader in enumerate(dataloaders_source):
        # Get the batch
        x, z, y = next(iter(dataloader))
        x, z, y = x.to(device), z.to(device), y.to(device).float()
        pred = models_source[t](x, z)
        loss = loss_fn(pred, y)
        loss_aux[t] = loss.item()
        # update the linear layers
        optimizers_linear[t].zero_grad()
        loss.backward(retain_graph=True)
        optimizers_linear[t].step()
        # get the loss for the pre-trained model
        pred = models_source[t](x, z)
        loss = loss_fn(pred, y)
        loss_sum_aux += loss
      # update the representation model
      optimizer_pre_train.zero_grad()
      loss_sum_aux.backward()
      optimizer_pre_train.step()

    return [loss_aux[t]/len(dataloaders_source[t]) for t in range(T)]

  def evaluate(model, loader, device):
    if type(model) is list: # 'model' is a list of models
      loss_aux = []
      T = len(model)
      for t in range(T):
        model[t].eval()  # set the model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():
          for x, images, labels in loader[t]:
            x, images, labels = x.to(device), images.to(device), labels.to(device).float()
            predictions = model[t](x, images)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
        model[t].train()  # set the model back to training mode
        loss_aux.append(total_loss / len(loader[t]))
      return loss_aux
    else: # 'model' is a single model
      model.eval()
      model.to(device)
      total_loss = 0
      with torch.no_grad():
        for x, images, labels in loader:
          x, images, labels = x.to(device), images.to(device), labels.to(device).float()
          outputs = model(x, images)
          loss = criterion(outputs, labels)
          total_loss += loss.item()
      model.train()
      return total_loss / len(loader)

  model_rep = RepresentNN().to(device)
  # rep_model.load_state_dict(torch.load('trained_model/classify_mnist_v1_model.pth'))
  models_source = [RegressionNN(dim_linear=1, dim_rep=10, rep_model=model_rep).to(device) for t in range(T)]
  model_target = RegressionNN(dim_linear=1, dim_rep=10, rep_model=model_rep).to(device)
  model_rep_best_valid = RepresentNN().to(device)
  criterion = nn.MSELoss()
  optimizer_rep = optim.Adam(model_rep.parameters())
  optimizers_linear = [optim.Adam(models_source[t].linear_layers.parameters(), lr=0.5) for t in range(T)]
  # optimizer only includes the model's rep_model
  scheduler = ReduceLROnPlateau(optimizer_rep, mode='min', factor=0.2, patience=3, verbose=True, min_lr=0.00001)


  # %% Load data
  img_shape = (28, 28)
  batch_size = 128
  flag_small_dataset = True

  # Define the dataset and dataloader
  transform = transforms.Compose([
    # transforms.Resize(img_shape),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])
  # Load the MNIST dataset
  mnist_train = datasets.MNIST(root='data', train=True, transform=transform, download=False)
  sample_size = mnist_train.__len__()

  # set source data
  sample_size_source_train = int(sample_size * 0.4)
  sample_size_source_valid = int(500)
  dataset_source = []
  loader_source_train = []
  loader_source_valid = []
  for t in range(T):
    # Create the special dataset
    arith_mnist_dataset_train = SpecialMNISTDataset(mnist_train)
    subset_indices = torch.randperm(sample_size)[:(sample_size_source_train+sample_size_source_valid)]
    subset_indices_train = subset_indices[:sample_size_source_train]
    subset_indices_valid = subset_indices[sample_size_source_train:]
    subset_data_train = torch.utils.data.Subset(arith_mnist_dataset_train, subset_indices_train)
    subset_data_valid = torch.utils.data.Subset(arith_mnist_dataset_train, subset_indices_valid)
    train_loader = DataLoader(subset_data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(subset_data_valid, batch_size=batch_size, shuffle=True)
    dataset_source.append(arith_mnist_dataset_train)
    loader_source_train.append(train_loader)
    loader_source_valid.append(valid_loader)

  # set target data
  sample_size_target_train, sample_size_target_valid = 100, 50
  arith_mnist_dataset_train = SpecialMNISTDataset(mnist_train, beta=1.0, gamma=1.0)
  subset_indices = torch.randperm(sample_size)[:(sample_size_target_train+sample_size_target_valid)]
  subset_indices_train = subset_indices[:sample_size_target_train]
  subset_indices_valid = subset_indices[sample_size_target_train:]
  subset_data_train = torch.utils.data.Subset(arith_mnist_dataset_train, subset_indices_train)
  subset_data_valid = torch.utils.data.Subset(arith_mnist_dataset_train, subset_indices_valid)
  loader_target_train = DataLoader(subset_data_train, batch_size=batch_size, shuffle=True)
  loader_target_valid = DataLoader(subset_data_valid, batch_size=batch_size, shuffle=True)
  mnist_test = datasets.MNIST(root='data', train=False, transform=transform, download=False) # True False
  arith_mnist_dataset_test = SpecialMNISTDataset(mnist_test, arith_mnist_dataset_train.beta, arith_mnist_dataset_train.gamma, noise_level=0.0)
  loader_target_test = DataLoader(arith_mnist_dataset_test, batch_size=batch_size, shuffle=False)

  # %% Train data

  # debug
  print("Debug, before training")
  loss_target_valid = evaluate(model_target, loader_target_valid, device)
  loss_target_test = evaluate(model_target, loader_target_test, device)
  est_err_beta = np.linalg.norm(arith_mnist_dataset_test.beta - model_target.linear_layers[0].weight[0,0].to('cpu').detach().numpy())
  print(f"[Target] valid loss: {loss_target_valid}, test loss: {loss_target_test}")
  print(f"True beta: {arith_mnist_dataset_test.beta}, initial beta: {model_target.linear_layers[0].weight[0,0].to('cpu').detach().numpy()}")
  print(f"Initial estimation error of beta: {est_err_beta:.4f}")
  # end debug

  num_epochs_pretrain = 25
  loss_best_valid = np.inf
  patience = 5
  last_update_count = 0

  for epoch in range(num_epochs_pretrain):
    train_loss = train_loop_mtl(models_source, loader_source_train, criterion, optimizer_rep, optimizers_linear)
    # evaluate model at validation set
    loss_valid = evaluate(models_source, loader_source_valid, device)
    if epoch % 1 == 0 and verbose: 
      print(f"Epoch {epoch}\n--------------------")
      print(f"\t[Source] train loss: {train_loss}, valid loss: {loss_valid}")
    if np.sum(loss_valid) < loss_best_valid:
      loss_best_valid = np.sum(loss_valid)
      model_rep_best_valid.load_state_dict(model_rep.state_dict())
      # fine tune the target model
      model_target.fine_tune(loader_target_train, device)
      loss_target_valid = evaluate(model_target, loader_target_valid, device)
      if verbose: 
        print(Fore.RED + f"Update representation, with target valid loss: {loss_target_valid}")
        print(Fore.RESET)
      last_update_count = 0

    # Early stopping
    if last_update_count == patience:
      if verbose: print(f'Early stopping at epoch {epoch+1}')
      break
    else:
      last_update_count += 1

    # Reduce learning rate
    scheduler.step(np.sum(loss_valid))

  # %% Train the target model
  if False:
    # load from pre-trained model
    model_target.rep_model.load_state_dict(torch.load('trained_model/rep_model_v3.1.pth'))
  else:
    model_target.rep_model.load_state_dict(model_rep_best_valid.state_dict())
  train_loss_target = model_target.fine_tune(loader_target_train, device)
  loss_target_valid = evaluate(model_target, loader_target_valid, device)
  loss_target_test = evaluate(model_target, loader_target_test, device)
  if verbose: print(f"[Target] train loss: {train_loss_target}, valid loss: {loss_target_valid}, test loss: {loss_target_test}")
  est_err_beta = np.linalg.norm(arith_mnist_dataset_test.beta - model_target.linear_layers[0].weight[0,0].to('cpu').detach().numpy())
  print(f"Estimation error of beta: {est_err_beta:.4f}")

  with open(output_file, 'a') as f:
    print(f"Test loss: {loss_target_test}, Estimation error of beta: {est_err_beta:.4f}", file=f)

  # %% Check the classification performance of the best representation model
  if False:
    def get_acc(model, loader, device):
      model.eval()
      model.to(device)
      with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
          images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
          outputs = model(images)
          correct += (torch.round(outputs) == labels).sum().item()
          total += labels.size(0)
      model.train()
      return correct / total

    test_loader = DataLoader(datasets.MNIST(root='data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    acc_test = get_acc(model_rep_best_valid, test_loader, device)

    print(f"Accuracy on test data: {acc_test:.4f}")
    # This result is poor, because we use the regression model to classify the data

  # %% Recover the true h
  if False:
    test_loader = DataLoader(datasets.MNIST(root='data', train=False, transform=transform), batch_size=batch_size, shuffle=False)

    R = np.concatenate([(np.arange(10)*dataset_source[i].gamma.detach().numpy()).reshape((-1, 1)) for i in range(T)], axis=1)
    R_hat = np.concatenate([models_source[i].linear_layers[0].weight[0].to('cpu').detach().numpy()[1:].reshape((-1, 1)) for i in range(T)], axis=1)
    model = model_rep.to('cpu')
    A = R @ R_hat.T @ np.linalg.pinv(R_hat @ R_hat.T)
    A_inv = np.linalg.pinv(A)
    Z_all, Y_all = [], []
    for Z, Y in test_loader:
      Z_all.append(Z)
      Y_all.append(Y)
    Z_all = torch.cat(Z_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    rep = torch.exp(model(Z_all)).detach().numpy()
    rep_trans = rep @ A_inv

    correct = 0
    total = 0
    for i in range(len(Y_all)):
      correct += (np.argmax(rep_trans[i, :]) == Y_all[i]).item()
      total += 1

    print(f"Accuracy on test data: {correct / total:.4f}")
    # This result is not good, because we use the regression model to classify the data

  # %% Classifcation performance of the best representation model using on a new classification model

  class MyModel(nn.Module):
    """
    THis is a new class, like the RegressionNN, but is for classification the MNIST dataset based on the representation model.
    """
    def __init__(self, rep_model=None):
      super(MyModel, self).__init__()

      self.rep_model = RepresentNN() if rep_model is None else rep_model

      self.linear_layers = nn.Sequential(
        nn.Linear(10, 512),
        nn.ReLU(),
        nn.Linear(512, 10),     # representation dimensional is 10
        nn.LogSoftmax(dim=1)    # softmax for classification
      )

    def forward(self, z):
      output_rep = self.rep_model(z)
      output_rep = torch.exp(output_rep) # convert the value into probability
      # output_rep = self.rep_linear(output_rep)
      output = self.linear_layers(output_rep)
      return output

  def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
      total_loss = 0
      for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    model.train()
    return total_loss / len(loader)

  def get_acc(model, loader, device):
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


  # model_rep.load_state_dict(torch.load('trained_model/rep_model_v3.1.pth'))
  model = MyModel(model_rep).to(device)
  model_best_valid = MyModel().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.linear_layers.parameters())
  scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True, min_lr=0.00001)

  transform = transforms.Compose([
    transforms.Resize(img_shape),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])
  dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True) # True False
  sample_size = dataset.__len__()
  if flag_small_dataset == True:
    sample_size_train = 10000
    sample_size_valid = 500
    subset_indices = torch.randperm(sample_size)[:(sample_size_train+sample_size_valid)]
    subset_indices_train = subset_indices[:sample_size_train]
    subset_indices_valid = subset_indices[sample_size_train:]
    subset_data_train = torch.utils.data.Subset(dataset, subset_indices_train)
    subset_data_valid = torch.utils.data.Subset(dataset, subset_indices_valid)
    train_loader = DataLoader(subset_data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(subset_data_valid, batch_size=batch_size, shuffle=True)
  else:
    sample_size_train = int(sample_size * 0.8)
    sample_size_valid = sample_size - sample_size_train
    subset_indices_train = range(0, sample_size_train)
    subset_indices_valid = range(sample_size_train, sample_size)
    subset_data_train = torch.utils.data.Subset(dataset, subset_indices_train)
    subset_data_valid = torch.utils.data.Subset(dataset, subset_indices_valid)
    train_loader = DataLoader(subset_data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(subset_data_valid, batch_size=batch_size, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  test_loader = DataLoader(datasets.MNIST(root='data', train=False, transform=transform), batch_size=batch_size, shuffle=False)

  num_epochs = 20
  loss_best_valid = np.inf
  acc_best_valid = 0
  patience = 10
  counter = 0

  model.train()
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      # Convert labels to float for regression
      images, labels = images.to(device), labels.to(device)
      
      # Forward pass
      outputs = model(images)
      # loss = criterion(outputs, labels)
      loss = criterion(outputs, labels)
      
      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # evaluate model at validation set
      loss_valid = evaluate(model, valid_loader, device)
      val_accuracy = get_acc(model, valid_loader, device)
      if loss_valid < loss_best_valid:
        loss_best_valid = loss_valid
        acc_best_valid = val_accuracy
        model_best_valid.load_state_dict(model.state_dict())
        counter = 0

      if epoch == 0 and i < 10:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Valid Loss: {loss_valid:.4f}, Best Valid Loss: {loss_best_valid:.4f}, Best Valid Accuracy: {acc_best_valid:.4f}')
      if (i+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Valid Loss: {loss_valid:.4f}, Best Valid Loss: {loss_best_valid:.4f}, Best Valid Accuracy: {acc_best_valid:.4f}')

    # Early stopping
    if counter == patience:
      print(f'Early stopping at epoch {epoch+1}, Step {i+1}')
      break
    else:
      counter += 1
    # Learning rate scheduler
    scheduler.step(val_accuracy)

  model = model_best_valid # Use the best model
  # Switch model to evaluation mode
  loss_test = evaluate(model, test_loader, device)
  acc_test = get_acc(model, test_loader, device)
  print(f"Test loss: {loss_test:.4f}, Test Acc: {acc_test:.4f}")

  with open(output_file, 'a') as f:
    print(f"Test loss: {loss_test:.4f}, Test Acc: {acc_test:.4f}", file=f)

# %% Table 2
res = np.array([[0.2196975917189936, 0.0646, 0.9830],
                [0.126836789936959, 0.0547, 0.9886],
                [0.18481564776429646, 0.2061, 0.9900],
                [0.33374453119084807, 0.0039, 0.9875],
                [0.2163396103284027, 0.0210, 0.9856]])
np.mean(res, axis=0)
np.std(res, axis=0)
