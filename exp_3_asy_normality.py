"""
date:   2024.1.22
author: huihang@mail.ustc.edu.cn
"""

# %% Import packages
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.NN_Models import PartialLinearRegression, SharedNN, Discriminator
from utils.Data_Generator import DGPs
import numpy as np
import time
from colorama import init, Fore
import argparse
import platform

system_type = platform.system().lower()
if system_type == "linux":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

start_time = time.time()

# %%

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_source", help="number of samples in training source", type=int, default=400
)
parser.add_argument(
    "--n_target", help="number of samples in training target", type=int, default=50
)
parser.add_argument("--p", help="nonlinear dimension", type=int, default=10)
parser.add_argument("--q", help="linear dimension", type=int, default=1)
parser.add_argument("--r", help="representation dimension", type=int, default=3)
parser.add_argument("--T", help="number of source dataset", type=int, default=5)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=1)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=6)
parser.add_argument(
    "--record_dir", help="directory to save record", type=str, default=""
)
parser.add_argument(
    "--verbose", help="print training information", type=int, default=True
)

args = parser.parse_args()

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
    T = len(models_source)  # redefine T
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
        loss_aux = np.array([0.0] * T)
        for t, dataloader in enumerate(dataloaders_aux):
            loss_aux[t] = models_source[t].fine_tune(dataloader)

    return [loss_aux[t] / len(dataloaders_aux[t]) for t in range(T)]


def train_loop_orthognal(
    models_source,
    D_net,
    dataloaders_aux,
    loss_fn,
    optimizer_pre_train,
    optimizer_D,
    zlr,
):
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
    dim_rep = models_source[0].dim_rep  # the dimension of the representation
    shared_net = models_source[
        0
    ].shared_layers  # the shared network of the source models
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
            w = shared_net(
                x[:, q:]
            )  # input (nonlinear part) data to the generator, w is the latent representation
            tmp_w.append(w)
            # store the prediction error from source models
            pred = models_source[t](x, is_training=True)
            loss = loss_fn(pred, y)
            loss_sum_aux += loss

        # train the discriminator
        w = torch.cat(tmp_w, dim=0)
        new_w = Variable(w.clone())
        D_real = torch.sigmoid(
            D_net(new_w)
        )  # input latent representation to the discriminator
        D_fake = torch.sigmoid(
            D_net(torch.randn(new_w.shape[0], dim_rep))
        )  # input Gaussian noise to the discriminator
        D_loss_real = torch.nn.functional.binary_cross_entropy(
            D_real, torch.ones(new_w.shape[0], 1)
        )  # loss for real data
        D_loss_fake = torch.nn.functional.binary_cross_entropy(
            D_fake, torch.zeros(new_w.shape[0], 1)
        )  # loss for fake data
        D_loss = (D_loss_real + D_loss_fake) / 2.0  # loss of the discriminator part
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        w.detach_()  # detach the representation from the graph

        # manually set the update-direction of weights of the generator
        w_t = Variable(w.clone(), requires_grad=True)
        d = -D_net(w_t)
        d.backward(torch.ones(w_t.shape[0], 1), retain_graph=True)  # gradient???
        w = (
            w + zlr * w_t.grad
        )  # manually set the update-direction of weights of the generator

        # train the source models
        latent = torch.cat(
            tmp_w, dim=0
        )  # latent representation of all data (from source models)
        loss_regularization = MSEloss(
            w, latent
        )  # push the generator to the direction as we expected (manually set one)
        loss = loss_sum_aux + loss_regularization  # loss = MSE + regularize
        optimizer_pre_train.zero_grad()
        loss.backward()
        optimizer_pre_train.step()

        print(
            f"loss regular: {loss_regularization.item()}, loss sum: {loss_sum_aux.item()}"
        )

        # fine tune the last layer
        loss_aux = np.array([0.0] * args.T)
        for t, dataloader in enumerate(dataloaders_aux):
            loss_aux[t] = models_source[t].fine_tune(dataloader)

    return [loss_aux[t] / len(dataloaders_aux[t]) for t in range(args.T)]


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
    J = torch.eye(n) - torch.ones(n, n) / n
    RX = J @ DX @ J
    RY = J @ DY @ J
    # Computing distance covariance and correlation
    covXY = torch.mul(RX, RY).sum() / (n * n)
    covX = torch.mul(RX, RX).sum() / (n * n)
    covY = torch.mul(RY, RY).sum() / (n * n)
    return covXY / torch.sqrt(covX * covY)


def evaluate(model, dataloader, criterion):
    if type(model) is list:  # models['aux-nn'] is a list of models
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
    colors = [
        Fore.RED,
        Fore.YELLOW,
        Fore.BLUE,
        Fore.GREEN,
        Fore.CYAN,
        Fore.LIGHTRED_EX,
        Fore.LIGHTYELLOW_EX,
        Fore.LIGHTBLUE_EX,
        Fore.LIGHTGREEN_EX,
        Fore.LIGHTCYAN_EX,
        Fore.LIGHTMAGENTA_EX,
        Fore.LIGHTWHITE_EX,
        Fore.MAGENTA,
    ]
    best_valid, model_color = {}, {}
    for i, name in enumerate(model_names):
        best_valid[name] = np.inf
        model_color[name] = colors[i]
    best_valid_target = np.inf
    test_perf = {}
    est_perf = {}
    best_shared_layers = SharedNN(
        dim_input=args.p, dim_rep=args.r, width=args.width, depth=args.depth
    )

    if "mtl-pl-2lr" in model_names:
        flag_aux_train = "default"  # 'default' 'joint' 'orthogonal'
        flag_target_train = "default"  # 'default' 'joint'

        # train the shared layers
        if args.verbose:
            print(f"Train shared layers--------------------")
        patience = 50
        last_update_count = 0
        for epoch in range(num_epochs_pretrain):

            # call train_loop_mtl() or train_loop_orthognal()
            if flag_aux_train == "default":  # debug: source + target
                train_loss = train_loop_mtl(
                    models["mtl-pl-2lr"]["aux-nn"],
                    dgps.dataloaders_source_train,
                    criterion,
                    optimizers["mtl-pl-2lr"],
                )  # original method
            elif flag_aux_train == "joint":  # joint training
                train_loss = train_loop_mtl(
                    models["mtl-pl-2lr"]["aux-nn"]
                    + [models["mtl-pl-2lr"]["target-nn"]],
                    dgps.dataloaders_source_train + [dgps.dataloader_target_train],
                    criterion,
                    optimizers["mtl-pl-2lr"],
                )  # original method
            elif flag_aux_train == "orthogonal":
                # learning rate schedule for the regularization of the shared layers
                if epoch < 10:
                    zlr = 4.0
                elif epoch == 20:
                    zlr = 2.0
                elif epoch == 40:
                    zlr = 1.0
                train_loss = train_loop_orthognal(
                    models["mtl-pl-2lr"]["aux-nn"],
                    D_net,
                    dgps.dataloaders_source_train,
                    criterion,
                    optimizers["mtl-pl-2lr"],
                    optimizers["discriminator"],
                    zlr=zlr,
                )  # add regularization

            valid_loss = evaluate(
                models["mtl-pl-2lr"]["aux-nn"], dgps.dataloaders_source_valid, criterion
            )
            if epoch % 40 == 0 and args.verbose:
                print(f"Epoch {epoch}\n--------------------")
                print(f"\ttrain loss: {train_loss}, valid loss: {valid_loss}")
            last_update_count += 1
            if True:  # old update method
                if np.sum(valid_loss) < best_valid["mtl-pl-2lr"]:
                    best_valid["mtl-pl-2lr"] = np.sum(valid_loss)
                    test_loss = evaluate(
                        models["mtl-pl-2lr"]["aux-nn"],
                        dgps.dataloaders_source_test,
                        criterion,
                    )
                    # end debug
                    best_shared_layers.load_state_dict(
                        models["mtl-pl-2lr"]["target-nn"].shared_layers.state_dict()
                    )
                    if True and args.verbose:
                        print(
                            model_color["mtl-pl-2lr"]
                            + f"Model [{'mtl-pl-2lr'}] update test loss, "
                            f"best valid loss = {valid_loss}, current test loss = {test_loss}"
                        )
                    last_update_count = 0
            else:  # new update method
                # fine tune the target model
                models["mtl-pl-2lr"]["target-nn"].fine_tune(
                    dgps.dataloader_target_train
                )
                valid_loss_target = evaluate(
                    models["mtl-pl-2lr"]["target-nn"],
                    dgps.dataloader_target_valid,
                    criterion,
                )
                # check valid loss
                if valid_loss_target < best_valid_target:
                    best_valid_target = valid_loss_target
                    # debug: print the test loss in target data
                    test_loss_target = evaluate(
                        models["mtl-pl-2lr"]["target-nn"],
                        dgps.dataloader_target_test,
                        criterion,
                    )
                    if args.verbose:
                        print(
                            model_color["mtl-pl-2lr"]
                            + f"target model valid loss: {valid_loss_target}, test loss: {test_loss_target}"
                        )
                    # end debug
                    test_loss = evaluate(
                        models["mtl-pl-2lr"]["aux-nn"],
                        dgps.dataloaders_source_test,
                        criterion,
                    )
                    best_shared_layers.load_state_dict(
                        models["mtl-pl-2lr"]["target-nn"].shared_layers.state_dict()
                    )
                    if args.verbose:
                        print(
                            Fore.MAGENTA
                            + f"Source Model [{'mtl-pl-2lr'}] update target test loss, "
                            f"best valid loss = {valid_loss_target}, current test loss = {test_loss_target}, current source test loss = {test_loss}"
                        )
                    last_update_count = 0

            # patience termination
            if last_update_count > patience and max(valid_loss) < 0.5:
                # early stop
                break

        # train the target model
        if args.verbose:
            print(f"Train target model---------------------")
        # initialize the target model with the best shared layers
        models["mtl-pl-2lr"]["target-nn"].shared_layers.load_state_dict(
            best_shared_layers.state_dict()
        )
        if flag_target_train == "default":
            # only fine tune the last layer for target model
            train_loss_target = models["mtl-pl-2lr"]["target-nn"].fine_tune(
                dgps.dataloader_target_train
            )
        elif flag_target_train == "joint":
            # fine tune the shared layer + last layer
            for epoch in range(100):
                train_loss_target = train_loop(
                    models["mtl-pl-2lr"]["target-nn"],
                    dgps.dataloader_target_train,
                    criterion,
                    optimizers["mtl-pl-2lr"],
                    fine_tune=True,
                )
                valid_loss = evaluate(
                    models["mtl-pl-2lr"]["target-nn"],
                    dgps.dataloader_target_valid,
                    criterion,
                )
                if valid_loss < best_valid_target:
                    best_valid_target = valid_loss
                    test_loss = evaluate(
                        models["mtl-pl-2lr"]["target-nn"],
                        dgps.dataloader_target_test,
                        criterion,
                    )
                    test_perf["mtl-pl-2lr"] = test_loss
                    est_error = np.linalg.norm(
                        dgps.params["beta"][:, 0]
                        - models["mtl-pl-2lr"]["target-nn"]
                        .linear_output.weight.detach()
                        .numpy()[0, : dgps.dim_linear]
                    )
                    est_perf["mtl-pl-2lr"] = est_error
                    best_shared_layers.load_state_dict(
                        models["mtl-pl-2lr"]["target-nn"].shared_layers.state_dict()
                    )
                    if args.verbose:
                        print(
                            model_color["mtl-pl-2lr"]
                            + f"Target Model [{'mtl-pl-2lr'}] update test loss, "
                            f"best valid loss = {valid_loss}, current test loss = {test_loss}, "
                            f"est error = {est_error}"
                        )
            # final fine tune the target model
            models["mtl-pl-2lr"]["target-nn"].shared_layers.load_state_dict(
                best_shared_layers.state_dict()
            )
            train_loss_target = models["mtl-pl-2lr"]["target-nn"].fine_tune(
                dgps.dataloader_target_train
            )
        # estimate the noise level
        sigma2_hat = (
            train_loss_target.detach().numpy()
        )  # this is the mean of prediction error on the target training data
        # print(f"[mtl target] sigma2_hat: {sigma2_hat}")
        # since the validation set is too small, so we do not use it to estimate the noise level
        # evaluate the target model on test data and report the performance
        test_loss = evaluate(
            models["mtl-pl-2lr"]["target-nn"], dgps.dataloader_target_test, criterion
        )
        test_perf["mtl-pl-2lr"] = test_loss
        beta_hat = (
            models["mtl-pl-2lr"]["target-nn"]
            .linear_output.weight.detach()
            .numpy()[0, : dgps.dim_linear]
        )
        est_error = np.linalg.norm(dgps.params["beta"][:, 0] - beta_hat)
        est_perf["mtl-pl-2lr"] = est_error
        if args.verbose:
            print(
                model_color["mtl-pl-2lr"] + f"Model [{'mtl-pl-2lr'}] report: "
                f"final test loss = {test_loss}, "
                f"est error = {est_error}"
            )
    # [begin] estimate the variance of the estimator
    X_train = dgps.data_target["feature_train"][:, : dgps.dim_linear]
    Z_train_torch = torch.tensor(
        dgps.data_target["feature_train"][:, dgps.dim_linear :], requires_grad=False
    ).float()
    H_hat = (
        models["mtl-pl-2lr"]["target-nn"].shared_layers(Z_train_torch).detach().numpy()
    )
    J_hat = (
        X_train.T
        @ (
            np.eye(dgps.n_train_target)
            - H_hat @ np.linalg.inv(H_hat.T @ H_hat) @ H_hat.T
        )
        @ X_train
        / dgps.n_train_target
    )
    est_var = (
        sigma2_hat * np.linalg.inv(J_hat) / dgps.n_train_target
    )  # the estimated variance of the estimator with known noise level
    alpha = np.ones(dgps.dim_linear) / np.sqrt(dgps.dim_linear)
    alpha = alpha.reshape((-1, 1))
    bias_vec = np.mean(dgps.params["beta"][:, 0] - beta_hat)
    var_hat_vec = alpha.T @ est_var @ alpha
    var_hat_vec = var_hat_vec[0, 0]
    # [end] estimate variance
    # [begin] debug
    H_true = dgps.h(dgps.data_target["feature_train"][:, dgps.dim_linear :])
    J_true = (
        X_train.T
        @ (
            np.eye(dgps.n_train_target)
            - H_true @ np.linalg.inv(H_true.T @ H_true) @ H_true.T
        )
        @ X_train
        / dgps.n_train_target
    )
    tmp = (
        np.eye(dgps.n_train_target)
        - H_true @ np.linalg.inv(H_true.T @ H_true) @ H_true.T
    )
    beta_hat_oracle = (
        np.linalg.inv(X_train.T @ tmp @ X_train)
        @ X_train.T
        @ tmp
        @ dgps.data_target["label_train"]
    )
    tmp = (
        np.eye(dgps.n_train_target)
        - X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    )
    gamma_hat_oracle = (
        np.linalg.inv(H_true.T @ tmp @ H_true)
        @ H_true.T
        @ tmp
        @ dgps.data_target["label_train"]
    )
    sigma2_hat = (
        np.linalg.norm(
            dgps.data_target["label_train"]
            - X_train @ beta_hat_oracle
            - H_true @ gamma_hat_oracle
        )
        ** 2
        / dgps.n_train_target
    )
    est_var_oracle = sigma2_hat * np.linalg.inv(J_true) / dgps.n_train_target
    # [end] debug

    pred_result = np.zeros((1, len(model_names)))
    est_result = np.zeros((1, len(model_names)))

    for i, name in enumerate(model_names):
        # est_result[0, i] = est_perf[name]
        # pred_result[0, i] = test_perf[name]
        pred_result[0, i] = test_perf[name] / np.var(dgps.data_target["label_test"])
        est_result[0, i] = est_perf[name] / np.linalg.norm(dgps.params["beta"][:, 0])

    return pred_result, est_result, bias_vec, var_hat_vec


# %% generate data
noise_level = 0.3  # 0.3 1.0

feature_flag = "iid-uniform"
if args.exp_id == 6:
    coef_flag = "heterogeneous"
    rep_flag = "additive"
elif args.exp_id == 8:
    coef_flag = "heterogeneous"
    rep_flag = "additive-factor"
elif args.exp_id == 9:
    coef_flag = "homogeneous"
    rep_flag = "deep"
elif args.exp_id == 10:
    coef_flag = "heterogeneous"
    rep_flag = "deep"

dgps = DGPs(
    T=args.T,
    dim_linear=args.q,
    dim_nonlinear=args.p,
    dim_rep=args.r,
    rep_flag=rep_flag,
    coef_flag=coef_flag,
    feature_flag=feature_flag,
    n_train_source=args.n_source,
    n_train_target=args.n_target,
    noise_level=noise_level,
    seed=None,
)


# re-set the coefficients: beta, betas, gamma, gammas
if True:
    np.random.seed(2024)  # set a fixed seed
    dgps.params["beta"] = np.repeat(1, dgps.dim_linear).reshape((-1, 1))
    dgps.params["betas"] = [np.random.randn(dgps.dim_linear, 1) for t in range(args.T)]
    dgps.params["gamma"] = np.repeat(1, args.r).reshape((-1, 1))
    dgps.params["gammas"] = [np.random.randn(args.r, 1) for t in range(args.T)]
    np.random.seed(args.seed)  # reset the seed
if args.exp_id == 9:
    dgps.params["betas"] = [dgps.params["beta"] for t in range(args.T)]
    dgps.params["gammas"] = [dgps.params["gamma"] for t in range(args.T)]
if args.verbose:
    print(dgps)

dgps.sample(seed=args.seed)  # generate samples


# %% prepare model parameters and define models
num_epochs_pretrain = 400  # 400

torch.manual_seed(42)

shared_nn = SharedNN(
    dim_input=args.p, dim_rep=args.r, width=args.width, depth=args.depth
)
nn_models_aux = [
    PartialLinearRegression(dim_linear=args.q, shared_layers=shared_nn)
    for t in range(args.T)
]
nn_model_target = PartialLinearRegression(dim_linear=args.q, shared_layers=shared_nn)
D_net = Discriminator(
    ndim=args.r
)  # Discriminator for representation regularization of our method
models = {"mtl-pl-2lr": {"aux-nn": nn_models_aux, "target-nn": nn_model_target}}

if args.verbose:
    print(nn_model_target)


optimizers = {}
optimizers["mtl-pl-2lr"] = torch.optim.Adam(
    [{"params": shared_nn.shared_layers.parameters(), "lr": args.lr}]
)
optimizers["discriminator"] = torch.optim.Adam(D_net.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# %% Evaluation
model_names = ["mtl-pl-2lr"]
test_l2_error, est_error, bias, var_hat = joint_train(model_names)

end_time = time.time()
print(
    f"Case with Exp_id: {args.exp_id}, T = {args.T}, n_source = {args.n_source}, n_target = {args.n_target}, p = {args.p}, q = {args.q}, r = {args.r}, seed = {args.seed} done: time = {end_time - start_time} secs"
)
if args.verbose:
    print(f"Test MSE: {test_l2_error}\nEst error: {est_error}")
tmp_res = np.array([bias, var_hat]).reshape((1, -1))
if args.verbose:
    print(f"bias: {bias}, var_hat: {var_hat}")

if len(args.record_dir) > 0:
    res_est_file = open(
        args.record_dir
        + f"/exp{args.exp_id}_n{args.n_source}_r{args.r}_p{args.p}_est.csv",
        "a",
    )
    np.savetxt(res_est_file, tmp_res, delimiter=",")
    res_est_file.close()

# %% Plot

# %% Notes
"""
绘制估计量的 se 与 sd. 

kill -9 $(ps aux | grep huihang | grep python | grep sim_asy-var | tr -s ' '| cut -d ' ' -f 2)

"""
