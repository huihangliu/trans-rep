"""
date:   2023.12.21
author: huihang@mail.ustc.edu.cn
note:
  Real data
"""

# %% Import packages
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.NN_Models import (
    PartialRegressionNN,
    PartialLinearRegression,
    SharedNN,
    RegressionNN,
    Discriminator,
)
from utils.Data_Generator import DGPs, RealData
from utils.Trans_MA import ModelAveraging
from utils.Trans_Lasso import trans_lasso
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
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

# seed = 1
# np.random.seed(seed)
# torch.manual_seed(seed)

start_time = time.time()

# %%
# TODO Re-run exp1

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--r", help="representation dimension", type=int, default=5)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=1)
parser.add_argument("--exp_id", help="experiment id", type=int, default=3)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
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
    T = len(models_source)
    # obtain the total sample size in each dataloader
    sample_size = [len(dataloaders_aux[t].dataset) for t in range(T)]
    sample_ratio = np.array(sample_size) / np.sum(sample_size)
    for batch_idx in range(len(dataloaders_aux[0])):
        # len(dataloaders_aux[0]) is the number of batches in the first source dataset, if the number of batches is different across different source datasets, we have to modify the code
        # I set the batch size as the sample size, so that the number of batches is 1
        loss_sum_aux = 0
        for t, dataloader in enumerate(dataloaders_aux):
            # Get the batch
            x, y = next(iter(dataloader))
            pred = models_source[t](x, is_training=True)
            loss = loss_fn(pred, y)
            loss_sum_aux += loss * sample_ratio[t]
        optimizer_pre_train.zero_grad()
        loss_sum_aux.backward()
        optimizer_pre_train.step()

        # fine tune the last layer
        loss_aux = np.array([0.0] * T)
        for t, dataloader in enumerate(dataloaders_aux):
            loss_aux[t] = models_source[t].fine_tune(dataloader) * sample_ratio[t]

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
        loss_aux = np.array([0.0] * T)
        for t, dataloader in enumerate(dataloaders_aux):
            loss_aux[t] = models_source[t].fine_tune(dataloader)

    return [loss_aux[t] / len(dataloaders_aux[t]) for t in range(T)]


def pairwise_distances(
    x, y=None
):  # Function for computing pairwise distances between vectors in x and y
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


def cor(X, Y, n):  # Function to calculate distance correlation between X and Y
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
    est_perf = {}
    pred_perf = {}
    best_shared_layers = SharedNN(
        dim_input=p, dim_rep=r, width=width, depth=depth, flag_batchnorm=False
    )

    if "ols-linear" in model_names:
        print(f"--------------------OLS--------------------\n")
        # run ols for linear part only, ignore the non-linear part
        X_train_linear = dgps.data_target["feature_train"][:, : dgps.dim_linear]
        est_beta = np.linalg.lstsq(
            X_train_linear, dgps.data_target["label_train"], rcond=None
        )[0][:, 0]
        est = est_beta
        est_perf["ols-linear"] = est
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test_spline"] @ est_beta
            )
            ** 2
        )
        pred_perf["ols-linear"] = pred_loss

    if "trans-lasso-spline" in model_names:
        if args.verbose:
            print(f"------------ Trans Lasso Spline -----------\n")
        X_list = [dgps.data_target["feature_train"][:, : dgps.dim_linear]] + [
            dgps.data_source["feature_train"][t][:, : dgps.dim_linear] for t in range(T)
        ]
        Z_list = [dgps.data_target["feature_train_spline"][:, dgps.dim_linear :]] + [
            dgps.data_source["feature_train_spline"][t][:, dgps.dim_linear :]
            for t in range(T)
        ]
        Y_list = [dgps.data_target["label_train"]] + dgps.data_source["label_train"]

        X_tmp = np.concatenate(X_list, axis=0)
        Z_tmp = np.concatenate(Z_list, axis=0)
        XZ_tmp = np.concatenate((X_tmp, Z_tmp), axis=1)
        Y_tmp = np.concatenate(Y_list, axis=0)
        Y_tmp = Y_tmp.flatten()
        n_vec = [np.shape(X_list[t])[0] for t in range(T + 1)]
        prop_re = trans_lasso(
            XZ_tmp, Y_tmp, n_vec.copy(), I_til=range(n_vec[0] // 3)
        )  # use 1/3 data for Q-aggregation
        est_beta_gamma = prop_re["beta_hat"]
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test_spline"] @ est_beta_gamma
            )
            ** 2
        )
        pred_perf["trans-lasso-spline"] = pred_loss

        est = est_beta_gamma[: dgps.dim_linear]
        est_perf["trans-lasso-spline"] = est

        loss_valid = np.mean(
            (
                dgps.data_target["label_valid"]
                - dgps.data_target["feature_valid_spline"] @ est_beta_gamma
            )
            ** 2
        )
        if args.verbose:
            print(f"Valid loss: {loss_valid}, Test loss: {pred_loss}")

    if "meta-analysis-spline" in model_names:
        if args.verbose:
            print(f"----------- Meta Analysis Spline ----------\n")
        model_averaging = ModelAveraging(num_models=T + 1)
        X_list = [dgps.data_target["feature_train_spline"]] + [
            dgps.data_source["feature_train_spline"][t] for t in range(T)
        ]
        Z_list = [dgps.data_target["feature_train_spline"][:, :0]] + [
            dgps.data_source["feature_train_spline"][t][:, :0] for t in range(T)
        ]
        Y_list = [dgps.data_target["label_train"]] + dgps.data_source["label_train"]
        res_meta_analysis = model_averaging.meta_analysis(Y_list, X_list, Z_list)
        est_beta_gamma = np.concatenate(
            (res_meta_analysis["beta_hat"], res_meta_analysis["alpha_hat"])
        )
        est = res_meta_analysis["beta_hat"][: dgps.dim_linear]
        est_perf["meta-analysis-spline"] = est
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test_spline"] @ est_beta_gamma
            )
            ** 2
        )
        pred_perf["meta-analysis-spline"] = pred_loss

        loss_valid = np.mean(
            (
                dgps.data_target["label_valid"]
                - dgps.data_target["feature_valid_spline"] @ est_beta_gamma
            )
            ** 2
        )
        if args.verbose:
            print(f"Valid loss: {loss_valid}, Test loss: {pred_loss}")

    if "pooled-regression-spline" in model_names:
        if args.verbose:
            print(f"--------- Pooled Regression Spline --------\n")
        model_averaging = ModelAveraging(num_models=T + 1)
        X_list = [dgps.data_target["feature_train_spline"]] + [
            dgps.data_source["feature_train_spline"][t] for t in range(T)
        ]
        Z_list = [dgps.data_target["feature_train_spline"][:, :0]] + [
            dgps.data_source["feature_train_spline"][t][:, :0] for t in range(T)
        ]
        Y_list = [dgps.data_target["label_train"]] + dgps.data_source["label_train"]
        res_pooled_regression = model_averaging.pooled_regression(
            Y_list, X_list, Z_list
        )
        est_beta_gamma = np.concatenate(
            (res_pooled_regression["beta_hat"], res_pooled_regression["alpha_hat"])
        )

        est = res_pooled_regression["beta_hat"][: dgps.dim_linear]
        est_perf["pooled-regression-spline"] = est
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test_spline"] @ est_beta_gamma
            )
            ** 2
        )
        pred_perf["pooled-regression-spline"] = pred_loss

        loss_valid = np.mean(
            (
                dgps.data_target["label_valid"]
                - dgps.data_target["feature_valid_spline"] @ est_beta_gamma
            )
            ** 2
        )
        if args.verbose:
            print(f"Valid loss: {loss_valid}, Test loss: {pred_loss}")

    if "map-spline" in model_names:
        if args.verbose:
            print(f"----------------- MAP Spline --------------\n")
        model_averaging = ModelAveraging(num_models=T + 1)
        X_list = [dgps.data_target["feature_train_spline"]] + [
            dgps.data_source["feature_train_spline"][t] for t in range(T)
        ]
        Z_list = [dgps.data_target["feature_train_spline"][:, :0]] + [
            dgps.data_source["feature_train_spline"][t][:, :0] for t in range(T)
        ]
        Y_list = [dgps.data_target["label_train"]] + dgps.data_source["label_train"]
        res_map = model_averaging.map(Y_list, X_list, Z_list)
        est_beta_gamma = np.concatenate((res_map["beta_hat"], res_map["alpha_hat"]))

        est = res_map["beta_hat"][: dgps.dim_linear]
        est_perf["map-spline"] = est
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test_spline"] @ est_beta_gamma
            )
            ** 2
        )
        pred_perf["map-spline"] = pred_loss

        loss_valid = np.mean(
            (
                dgps.data_target["label_valid"]
                - dgps.data_target["feature_valid_spline"] @ est_beta_gamma
            )
            ** 2
        )
        if args.verbose:
            print(f"Valid loss: {loss_valid}, Test loss: {pred_loss}")

    if "map" in model_names:
        if args.verbose:
            print(f"----------------- MAP Spline --------------\n")
        model_averaging = ModelAveraging(num_models=T + 1)
        X_list = [dgps.data_target["feature_train"]] + [
            dgps.data_source["feature_train"][t] for t in range(T)
        ]
        Z_list = [dgps.data_target["feature_train"][:, :0]] + [
            dgps.data_source["feature_train"][t][:, :0] for t in range(T)
        ]
        Y_list = [dgps.data_target["label_train"]] + dgps.data_source["label_train"]
        res_map = model_averaging.map(Y_list, X_list, Z_list)
        est_beta_gamma = np.concatenate((res_map["beta_hat"], res_map["alpha_hat"]))

        est = res_map["beta_hat"][: dgps.dim_linear]
        est_perf["map"] = est
        pred_loss = np.mean(
            (
                dgps.data_target["label_test"]
                - dgps.data_target["feature_test"] @ est_beta_gamma
            )
            ** 2
        )
        pred_perf["map"] = pred_loss

        loss_valid = np.mean(
            (
                dgps.data_target["label_valid"]
                - dgps.data_target["feature_valid"] @ est_beta_gamma
            )
            ** 2
        )
        if args.verbose:
            print(f"Valid loss: {loss_valid}, Test loss: {pred_loss}")

    if "mtl-pl-2lr" in model_names:
        flag_aux_train = "default"  # 'default', 'joint', 'orthogonal'
        if args.verbose:
            print(f"------------------ MTL[Our] ---------------\n")
        # train the shared layers
        if args.verbose:
            print(f"Train shared layers--------------------\n")
        patience = 80  # 80
        last_update_count = 0
        for epoch in range(num_epochs_pretrain):
            # learning rate schedule for the regularization of the shared layers
            if epoch % 40 == 0 and args.verbose:
                print(f"Epoch {epoch}\n--------------------")

            if flag_aux_train == "default":  # source only
                train_loss = train_loop_mtl(
                    models["mtl-pl-2lr"]["aux-nn"],
                    dgps.dataloaders_source_train,
                    criterion,
                    optimizers["mtl-pl-2lr"],
                )  # original method
                valid_loss = evaluate(
                    models["mtl-pl-2lr"]["aux-nn"],
                    dgps.dataloaders_source_valid,
                    criterion,
                )
            elif flag_aux_train == "joint":  # joint training
                train_loss = train_loop_mtl(
                    models["mtl-pl-2lr"]["aux-nn"]
                    + [models["mtl-pl-2lr"]["target-nn"]],
                    dgps.dataloaders_source_train + [dgps.dataloader_target_train],
                    criterion,
                    optimizers["mtl-pl-2lr"],
                )  # original method
                valid_loss = evaluate(
                    [models["mtl-pl-2lr"]["target-nn"]]
                    + models["mtl-pl-2lr"]["aux-nn"],
                    dgps.dataloaders_source_valid + [dgps.dataloader_target_valid],
                    criterion,
                )
            elif flag_aux_train == "orthogonal":
                if epoch < 10:
                    zlr = 6.0
                elif epoch == 20:
                    zlr = 3.0
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
                    models["mtl-pl-2lr"]["aux-nn"],
                    dgps.dataloaders_source_valid,
                    criterion,
                )

            if epoch % 40 == 0 and args.verbose:
                print(f"train loss: {train_loss}, valid loss: {valid_loss}")
            last_update_count += 1
            if np.sum(valid_loss) < best_valid["mtl-pl-2lr"]:
                best_valid["mtl-pl-2lr"] = np.sum(valid_loss)
                best_shared_layers.load_state_dict(
                    models["mtl-pl-2lr"]["target-nn"].shared_layers.state_dict()
                )
                # fine tune the target model
                if True and args.verbose:
                    print(
                        model_color["mtl-pl-2lr"]
                        + f"Model [{'mtl-pl-2lr'}] update best valid loss {valid_loss}"
                    )
                last_update_count = 0
            if last_update_count > patience:  #  and max(valid_loss) < 5
                # early stop
                break
        if args.verbose:
            print(f"--------------------End shared layers--------------------\n")

        # initialize the target model with the best shared layers
        models["mtl-pl-2lr"]["target-nn"].shared_layers.load_state_dict(
            best_shared_layers.state_dict()
        )
        # fine tune the target model for mtl method
        train_loss_target = models["mtl-pl-2lr"]["target-nn"].fine_tune(
            dgps.dataloader_target_train
        )
        valid_loss = evaluate(
            models["mtl-pl-2lr"]["target-nn"], dgps.dataloader_target_valid, criterion
        )
        if flag_test:
            test_loss = evaluate(
                models["mtl-pl-2lr"]["target-nn"],
                dgps.dataloader_target_test,
                criterion,
            )
        est = (
            models["mtl-pl-2lr"]["target-nn"]
            .linear_output.weight.detach()
            .numpy()[0, : dgps.dim_linear]
        )
        est_perf["mtl-pl-2lr"] = est
        print(f"estimated beta: {est}")
        print(
            f"estimates of coeff: {models['mtl-pl-2lr']['target-nn'].linear_output.weight.detach().numpy()[0, :]}"
        )
        if flag_test:
            pred_perf["mtl-pl-2lr"] = test_loss

        res_mtl = []
        for t in range(T):
            # if args.verbose: print(f"Beta Estimation of Model [{t+1}]: {models['mtl-pl-2lr']['aux-nn'][t].linear_output.weight.detach().numpy()[0, :dgps.dim_linear]}") # print the linear coefficient of all models
            res_mtl.append(
                models["mtl-pl-2lr"]["aux-nn"][t]
                .linear_output.weight.detach()
                .numpy()[0, : dgps.dim_linear]
            )

        # estimate the noise level
        sigma2_hat = (
            train_loss_target.detach().numpy()
        )  # this is the mean of prediction error on the target training data
        # [begin] estimate the variance of the estimator
        X_train = dgps.data_target["feature_train"][:, : dgps.dim_linear]
        Z_train_torch = torch.tensor(
            dgps.data_target["feature_train"][:, dgps.dim_linear :], requires_grad=False
        ).float()
        H_hat = (
            models["mtl-pl-2lr"]["target-nn"]
            .shared_layers(Z_train_torch)
            .detach()
            .numpy()
        )
        mu_hat = (
            X_train.T @ H_hat @ np.linalg.inv(H_hat.T @ H_hat + np.eye(args.r) * 1e-8)
        )
        XH_trin = np.concatenate((X_train, H_hat), axis=1)
        est_coef = (
            np.linalg.inv(XH_trin.T @ XH_trin)
            @ XH_trin.T
            @ dgps.data_target["label_train"]
        )
        residual = dgps.data_target["label_train"] - XH_trin @ est_coef
        residual = residual.flatten()
        tmp = np.zeros((dgps.dim_linear, dgps.dim_linear))
        for i in range(dgps.n_train_target):
            tmp += (
                residual[i] ** 2
                * (X_train[i, :].T - mu_hat @ H_hat[i, :].T)
                @ (X_train[i, :].T - mu_hat @ H_hat[i, :].T).T
            )
        tmp = tmp / dgps.n_train_target
        J_hat = (
            X_train.T
            @ (
                np.eye(dgps.n_train_target)
                - H_hat
                @ np.linalg.inv(H_hat.T @ H_hat + np.eye(args.r) * 1e-8)
                @ H_hat.T
            )
            @ X_train
            / dgps.n_train_target
            + np.eye(dgps.dim_linear) * 1e-8
        )
        if False:  # old version
            est_var = (
                sigma2_hat
                * np.linalg.inv(J_hat + np.eye(dgps.dim_linear) * 1e-8)
                / dgps.n_train_target
            )  # the estimated variance of the estimator with known noise level
        else:
            est_var = (
                np.linalg.inv(J_hat) @ tmp @ np.linalg.inv(J_hat) / dgps.n_train_target
            )
        # print(f"Estimated variance of the estimator: {est_var}")
        print(
            f"Estimated variance: {sigma2_hat * np.linalg.inv(J_hat) / dgps.n_train_target}"
        )
        # [end] estimate variance

        if args.verbose:
            print(f"--------------------End target model--------------------\n")

    # train the target model using different methods
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            if args.verbose:
                print(f"Epoch {epoch}\n--------------------")
        for model_name in model_names:
            if model_name not in [
                "vanilla-orginal",
                "vanilla-pl-1lr",
                "dropout-vanilla-pl-1lr",
                "vanilla-pl-2lr",
                "dropout-vanilla-pl-2lr",
                "stl-pl-1lr",
                "dropout-stl-pl-1lr",
                "stl-pl-2lr",
                "dropout-stl-pl-2lr",
            ]:
                continue
            flag_fine_tune = (
                True
                if model_name
                in [
                    "vanilla-pl-2lr",
                    "dropout-vanilla-pl-2lr",
                    "stl-pl-2lr",
                    "dropout-stl-pl-2lr",
                ]
                else False
            )
            train_loss = train_loop(
                models[model_name],
                dgps.dataloader_target_train,
                criterion,
                optimizers[model_name],
                fine_tune=flag_fine_tune,
            )
            valid_loss = evaluate(
                models[model_name], dgps.dataloader_target_valid, criterion
            )
            if valid_loss < best_valid[model_name]:
                best_valid[model_name] = valid_loss
                est = (
                    models[model_name]
                    .linear_output.weight.detach()
                    .numpy()[0, : dgps.dim_linear]
                    if model_name != "vanilla-orginal"
                    else 0
                )
                est_perf[model_name] = est
                if args.verbose:
                    print(
                        model_color[model_name]
                        + f"Model [{model_name}]: update current train loss: {train_loss}, best valid loss: {valid_loss}"
                    )
            if epoch % 40 == 0:
                if args.verbose:
                    print(
                        Fore.BLACK
                        + f"Model [{model_name}]: Epoch {epoch}/{num_epochs}, Train MSE: {train_loss}, Valid MSE: {valid_loss}"
                    )
            pred_loss = evaluate(
                models[model_name], dgps.dataloader_target_test, criterion
            )
            pred_perf[model_name] = pred_loss

    # est_result = np.zeros((1, len(model_names)))
    pred_result = np.zeros((1, len(model_names)))
    for i, name in enumerate(model_names):
        # est_result[0, i] = est_perf[name][0] # print the first estimated coefficient
        if flag_test:
            pred_result[0, i] = pred_perf[name]

    est_result = est_perf["mtl-pl-2lr"].reshape(
        1, -1
    )  # only return the estimates of mtl method
    return pred_result, est_result


# %% generate data

flag_test = False
# 'penn_jae' 'credit_card' 'house_rent'
if args.exp_id == 1:
    dgps = RealData("penn_jae", flag_test=flag_test)
    criterion = nn.MSELoss()
    flag_binary_outcome = False
elif args.exp_id == 2:
    dgps = RealData("credit_card", flag_test=flag_test)
    criterion = nn.MSELoss()
    flag_binary_outcome = False
elif args.exp_id == 3:
    dgps = RealData("house_rent", flag_test=flag_test)
    criterion = nn.MSELoss()
    flag_binary_outcome = False
p = dgps.dim_nonlinear
q = dgps.dim_linear
T = dgps.T


# %% prepare model parameters and define models
r = args.r
num_epochs = 50
num_epochs_pretrain = 400
depth = 4
width = 300

torch.manual_seed(42)
vanilla_nn_model_orgigin = RegressionNN(d=p + q, depth=depth, width=width)
vanilla_nn_model = PartialRegressionNN(
    dim_input=p + q, dim_linear=q, depth=depth, width=width, bias_last_layer=True
)
dropout_nn_model = PartialRegressionNN(
    dim_input=p + q,
    dim_linear=q,
    depth=depth,
    width=width,
    input_dropout=True,
    dropout_rate=0.6,
    bias_last_layer=True,
)
dropout_my_nn_model = PartialLinearRegression(
    dim_input=p + q,
    dim_linear=q,
    dim_rep=r,
    depth=depth,
    width=width,
    input_dropout=True,
    dropout_rate=0.6,
)
shared_nn = SharedNN(
    dim_input=p, dim_rep=r, width=width, depth=depth, flag_batchnorm=False
)
nn_models_aux = [
    PartialLinearRegression(
        dim_linear=q, shared_layers=shared_nn, flag_binary_outcome=flag_binary_outcome
    )
    for t in range(T)
]
nn_model_target = PartialLinearRegression(
    dim_linear=q, shared_layers=shared_nn, flag_binary_outcome=flag_binary_outcome
)
stl_1lr_model = PartialLinearRegression(
    dim_input=p + q, dim_linear=q, depth=depth, width=width
)
dropout_stl_1lr_model = PartialLinearRegression(
    dim_input=p + q,
    dim_linear=q,
    depth=depth,
    width=width,
    input_dropout=True,
    dropout_rate=0.6,
)
stl_2lr_model = PartialLinearRegression(
    dim_input=p + q,
    dim_linear=q,
    dim_rep=r,
    depth=depth,
    width=width,
    flag_batchnorm=False,
)
vanilla_pl_1lr_model = PartialRegressionNN(
    dim_input=p + q, dim_linear=q, depth=depth, width=width, bias_last_layer=True
)
dropout_vanilla_pl_1lr_model = PartialRegressionNN(
    dim_input=p + q,
    dim_linear=q,
    depth=depth,
    width=width,
    input_dropout=True,
    dropout_rate=0.6,
    bias_last_layer=True,
)
D_net = Discriminator(
    ndim=args.r
)  # Discriminator for representation regularization of our method
models = {
    "vanilla-orginal": vanilla_nn_model_orgigin,
    "vanilla-pl-1lr": vanilla_pl_1lr_model,
    "dropout-vanilla-pl-1lr": dropout_vanilla_pl_1lr_model,
    "vanilla-pl-2lr": vanilla_nn_model,
    "dropout-vanilla-pl-2lr": dropout_nn_model,
    "stl-pl-1lr": stl_1lr_model,
    "dropout-stl-pl-1lr": dropout_stl_1lr_model,
    "stl-pl-2lr": stl_2lr_model,
    "dropout-stl-pl-2lr": dropout_my_nn_model,
    "mtl-pl-2lr": {"aux-nn": nn_models_aux, "target-nn": nn_model_target},
}
# if args.verbose: print(models)
if args.verbose:
    print(nn_model_target)

learning_rate_nonlinear = 1e-3  # 1e-4
optimizers = {}
for method_name, model_x in models.items():
    # Assign different learning rates
    if method_name in [
        "vanilla-orginal",
        "vanilla-pl-1lr",
        "dropout-vanilla-pl-1lr",
        "stl-pl-1lr",
        "dropout-stl-pl-1lr",
    ]:
        optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate_nonlinear)
    elif method_name in ["vanilla-pl-2lr", "dropout-vanilla-pl-2lr"]:
        optimizer_x = torch.optim.Adam(
            [
                {
                    "params": model_x.nonlinear_layers.parameters(),
                    "lr": learning_rate_nonlinear,
                }
            ]
        )
    elif method_name in ["stl-pl-2lr", "dropout-stl-pl-2lr"]:
        optimizer_x = torch.optim.Adam(
            [
                {
                    "params": model_x.shared_layers.parameters(),
                    "lr": learning_rate_nonlinear,
                }
            ]
        )
    elif method_name in ["mtl-pl-2lr"]:
        optimizer_x = torch.optim.Adam(
            [
                {
                    "params": shared_nn.shared_layers.parameters(),
                    "lr": learning_rate_nonlinear,
                    "weight_decay": 0.05,
                }
            ]
        )
    optimizers[method_name] = optimizer_x

optimizers["discriminator"] = torch.optim.Adam(D_net.parameters(), lr=1e-4)


# %% Evaluation
model_names = ["mtl-pl-2lr", "stl-pl-2lr"]
# 'mtl-pl-2lr', 'stl-pl-2lr', 'ols-oracle',  'ols-linear', 'trans-lasso-spline', 'map-spline', 'meta-analysis-spline', 'pooled-regression-spline', 'stl-pl-2lr'
if flag_test == False:
    model_names = ["mtl-pl-2lr"]
pred_perf, est = joint_train(model_names)
if len(args.record_dir) > 0:
    res_est_file = open(args.record_dir + f"/exp{args.exp_id}_r{args.r}_est.csv", "a")
    np.savetxt(res_est_file, est, delimiter=",")
    res_est_file.close()
    res_pred_file = open(args.record_dir + f"/exp{args.exp_id}_r{args.r}_pred.csv", "a")
    np.savetxt(res_pred_file, pred_perf, delimiter=",")
    res_pred_file.close()
end_time = time.time()
print(
    f"Case with exp_id {args.exp_id}, T = {dgps.T}, r = {args.r}, seed = {args.seed} done: time = {end_time - start_time} secs"
)
if args.verbose:
    print(f"Est: {est}\nPredition Loss: {pred_perf}")

# %% Results
