"""
update:
2024-01-24: add credit card data
2023-02-02: remove credit card data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import utils.univariate_funcs as univariate_funcs
from utils.spline_func import transform_multidimensional_X

# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
import matplotlib.pyplot as plt


class RegressionDataset(Dataset):
    """
    A wrapper for regression dataset used in pytorch

    ...
    Attributes
    ----------
    n : int
      number of observations
    feature : np.array
      (n, d) matrix of the explanatory variables
    response : np.array
      (n, 1) matrix of the response variable
    """

    def __init__(self, x, y):
        self.n = np.shape(x)[0]
        if self.n != np.shape(y)[0]:
            raise ValueError("RegressionDataset: Sample size doesn't match!")
        self.feature = x
        self.response = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor(self.feature[idx, :], dtype=torch.float32), torch.tensor(
            self.response[idx, :], dtype=torch.float32
        )


class AdditiveModel:
    """
    The data generating process for nonparametric additive model

    Methods
    ----------
    sample(x)
      Return function value given the covariate
    """

    def __init__(self, num_funcs, rd_size=5, normalize=True):
        """
        A function to initialize the additive model used

        Parameters
        ----------
        num_funcs : int
          number of functions used = dimension of the covariate
        rd_size : int
          the univarate functions are uniformly samples from function
          with index [0, rd_size-1] in the function zoo
        """
        self.func_zoo = [
            univariate_funcs.func1,
            univariate_funcs.func2,
            univariate_funcs.func3,
            univariate_funcs.func4,
            univariate_funcs.func5,
            univariate_funcs.func6,
            univariate_funcs.func7,
            univariate_funcs.func8,
            univariate_funcs.func9,
            univariate_funcs.func10,
            univariate_funcs.func11,
            univariate_funcs.func12,
        ]
        self.func_name = [
            "sin",
            "root_abs",
            "sqrt_abs",
            "sigmoid",
            "cos_pi",
            "sin_pi",
            "-sin",
            "cos_2",
            "tan",
            "log",
            "exp",
            "square",
        ]
        self.num_funcs = num_funcs
        self.func_idx = np.random.randint(0, rd_size, num_funcs)
        self.func_idx[0] = 4
        if self.num_funcs >= 5:
            self.func_idx[4] = 4
        self.normalize = normalize

    def sample(self, x):
        """
        A function to return the function value given the value of the covariate

        Parameters
        ----------
        x : numpy.array
          (n, d) matrix of the covaraite, d is the number of explanatory variables,
          n is the number of data points

        Returns
        ----------
        y : numpy.array
          (n, 1) matrix represent the function value at n data points.
        """
        y = np.zeros((np.shape(x)[0], 1))
        if np.shape(x)[1] < self.num_funcs:
            raise ValueError(
                "AdditiveModel: Data dimension {}, ".format(np.shape(0))
                + "number of additive functions = {}".format(self.num_funcs)
            )
        for i in range(self.num_funcs):
            y = y + self.func_zoo[self.func_idx[i]](x[:, i : i + 1])
        if self.normalize:
            y = y / self.num_funcs
        return y

    def __str__(self):
        s = "Additive Models: f(x) = \n"
        for i in range(self.num_funcs):
            s = s + f"      {self.func_name[self.func_idx[i]]} (x_{i + 1})\n"
        return s


class HidenModel:
    """
    The data generating process for nonparametric additive model

    Methods
    ----------
    sample(x)
      Return function value given the covariate
    """

    def __init__(self, num_funcs=3, flag_linear=False):
        """
        A function to initialize the representation model used
        We randomly choose num_funcs functions from the function zoo

        Parameters
        ----------
        num_funcs : int
          number of functions used = dimension of the representation

        """
        self.func_zoo = [
            univariate_funcs.func0,
            univariate_funcs.func1,
            univariate_funcs.func2,
            univariate_funcs.func3,
            univariate_funcs.func4,
            univariate_funcs.func5,
            univariate_funcs.func6,
            univariate_funcs.func7,
            univariate_funcs.func8,
            univariate_funcs.func9,
            univariate_funcs.func10,
            univariate_funcs.func11,
            univariate_funcs.func12,
        ]
        self.func_name = [
            "x",
            "sin",
            "root_abs",
            "sqrt_abs",
            "sigmoid",
            "cos_pi",
            "sin_pi",
            "-sin",
            "cos_2",
            "tan",
            "log",
            "exp",
            "square",
        ]
        self.num_funcs = num_funcs
        self.func_idx = self.func_idx = np.random.randint(
            1, 6, self.num_funcs
        )  # exclude the first function x
        # # # The following is Fan's setting
        # np.random.randint(1, len(self.func_zoo), self.num_funcs)
        if num_funcs > 0:
            self.func_idx[0] = 5
        # self.func_idx[1] = 2
        # self.func_idx[2] = 1
        # self.func_idx[3] = 2
        if num_funcs > 4:
            self.func_idx[4] = 5

        if flag_linear:
            self.func_idx = [0] * self.num_funcs  # only use linear function

    def __str__(self):
        s = "Hiden Features: f(x) = \n"
        for i in range(self.num_funcs):
            s = s + f"      {self.func_name[self.func_idx[i]]} (x_{i + 1})\n"
        return s


class NonlinearRegressionModel:
    """
    The data generating process for nonparametric model
    Cite Zhou et al. (2022, JASA)

    Methods
    ----------
    sample(x)
      Return function value given the covariate
    """

    def __init__(self, num_funcs=3, normalize=False, hiden_feature=None):
        """
        A function to initialize the additive model used

        Parameters
        ----------
        num_funcs : int
          number of functions used = dimension of the covariate
        rd_size : int
          the univarate functions are uniformly samples from function
          with index [0, rd_size-1] in the function zoo
        """
        self.num_funcs = num_funcs
        if hiden_feature is None:
            hiden_feature = HidenModel(num_funcs=self.num_funcs)
        self.hiden_feature = hiden_feature
        if num_funcs != hiden_feature.num_funcs:
            self.num_funcs = hiden_feature.num_funcs
            Warning(
                "RegressionModel_Huang: num_funcs doesn't match! Use the param in hiden_feature"
            )

        # self.func_idx = [11, 10, 0] # x^2, exp, sin
        self.func_idx = hiden_feature.func_idx
        self.func_zoo = hiden_feature.func_zoo
        # self.gamma = np.random.randn(self.num_funcs, 1)
        self.gamma = np.array([1.0] * self.num_funcs)
        self.normalize = normalize

    def sample(self, x):
        """
        A function to return the function value given the value of the covariate

        Parameters
        ----------
        x : numpy.array
          (n, d) matrix of the covaraite, d is the number of explanatory variables,
          n is the number of data points

        Returns
        ----------
        y : numpy.array
          (n, 1) matrix represent the function value at n data points.
        """
        if np.shape(x)[1] < self.num_funcs:
            raise ValueError(
                "AdditiveModel: Data dimension {}, ".format(np.shape(0))
                + "number of additive functions = {}".format(self.num_funcs)
            )

        y = np.zeros((np.shape(x)[0], 1))
        for i in range(self.num_funcs):
            y += self.func_zoo[self.func_idx[i]](x[:, i : i + 1]) * self.gamma[i]

        if self.normalize:
            y = y / self.num_funcs
        return y

    def __str__(self):
        s = "Additive Models: f(x) = \n"
        for i in range(self.num_funcs):
            s = s + f"      {self.func_name[self.func_idx[i]]} (x_{i + 1})\n"
        return s


RegressionModel_Huang = NonlinearRegressionModel


class FactorModel:
    """
    The data generating process of linear factor model

    ...

    Attributes
    ----------
    loadings : numpy.array
      [p, r] factor loading matrix

    """

    def __init__(self, p, r=5, b_f=1, b_u=1, loadings=None):
        """
        Parameters
        ----------
        p : int
          number of covariates
        r : int
          number of factors
        b_f : float
          noise level of factors
        b_u : float
          noise level of idiosyncratic components
        loadings : numpy.array
          pre-specified factor loading matrix

        Returns
        -------
        loadings : numpy.array
          [p, r] matrix, factor loadings
        """

        self.p = p
        self.r = r
        self.b_f = b_f
        self.b_u = b_u
        if loadings is None:
            self.loadings = np.reshape(
                np.random.uniform(-np.sqrt(3), np.sqrt(3), p * r), (p, r)
            )
        else:
            self.loadings = loadings
        # if r==0:  self.loadings=None

    def sample(self, n, latent=False):
        """
        Parameters
        ----------
        n : int
            number of samples
        latent : bool
            whether return the latent factor structure

        Returns
        -------
        obs : np.array
            [n, p] matrix, observations
        factor : np.array
            [n, r] matrix, factor
        idiosyncratic_error : np.array
            [n, p] matrix, idiosyncratic error
        """
        if self.r > 0:
            factor = np.reshape(
                np.random.uniform(-self.b_f, self.b_f, n * self.r), (n, self.r)
            )
        idiosyncratic_error = np.reshape(
            np.random.uniform(-self.b_u, self.b_u, self.p * n), (n, self.p)
        )
        if self.r > 0:
            obs = np.matmul(factor, np.transpose(self.loadings)) + idiosyncratic_error
        else:
            obs = idiosyncratic_error
        if latent and self.r > 0:
            return obs, factor, idiosyncratic_error
        else:
            return obs


class PartialLinearModel:
    """
    The data generating process for partial linear model

    Methods
    ----------
    sample(x, z)
      x is linear part, z is nonparametric part
      Return function value given the covariate
    """

    def __init__(
        self,
        dim_input,
        dim_linear=0,
        dim_nonlinear=None,
        normalize=False,
        hiden_feature=None,
    ):
        """
        A function to initialize the additive model used

        Parameters
        ----------
        num_funcs : int
          dimension of representation layer
        rd_size : int
          the univarate functions are uniformly samples from function
          with index [0, rd_size-1] in the function zoo
        """
        self.dim_input = dim_input
        self.dim_linear = dim_linear
        self.dim_nonlinear = (
            dim_input - dim_linear if dim_nonlinear is None else dim_nonlinear
        )  # if nonlinear dim is not specified, use ALL rest of the input expect linear part

        self.linear_coef = np.random.randn(self.dim_linear, 1)
        self.gamma = np.random.randn(
            self.dim_nonlinear, 1
        )  #  np.ones((self.dim_nonlinear, 1))

        if hiden_feature is None:
            # this will generate a random representation layer
            hiden_feature = HidenModel(num_funcs=self.dim_nonlinear)
        self.hiden_feature = hiden_feature
        self.func_zoo = hiden_feature.func_zoo
        self.func_idx = hiden_feature.func_idx

        if self.dim_nonlinear != hiden_feature.num_funcs:
            self.dim_input = hiden_feature.num_funcs
            Warning(
                "PartialLinearModel: num_funcs doesn't match! Use the param in hiden_feature"
            )

        self.normalize = normalize

    def sample(self, x):
        """
        A function to return the function value given the value of the covariate

        Parameters
        ----------
        x : numpy.array
          (n, d) matrix of the covaraite, d is the number of explanatory (linear) variables,
          n is the number of data points

        Returns
        ----------
        y : numpy.array
          (n, 1) matrix represent the function value at n data points.
        """

        feature_linear = x[:, : self.dim_linear]
        feature_nonlinear = x[:, self.dim_linear :]
        if np.shape(feature_nonlinear)[1] < self.dim_nonlinear:
            raise ValueError(
                "PartialLinearModel: Nonlinear part data dimension {}, ".format(
                    np.shape(feature_nonlinear)[1]
                )
                + "number of additive functions = {}".format(self.dim_nonlinear)
            )

        y = np.zeros((np.shape(x)[0], 1))
        # linear part
        y += np.matmul(feature_linear, self.linear_coef)
        # nonlinear part
        for i in range(self.dim_nonlinear):
            y += (
                self.func_zoo[self.func_idx[i]](feature_nonlinear[:, i : i + 1])
                * self.gamma[i]
            )  # fixed a bug: x -> feature_nonlinear [2023-12-11]

        if self.normalize:
            y = y / self.dim_nonlinear
        return y

    def __str__(self):
        s = "Partial Linear Models: f(x, z) = \n"
        s += "\tLinear Coef: beta = "
        s += f"\t{self.linear_coef.reshape(1, -1)} x\n"
        s += "\tNonlinear Coef: gamma = "
        s += f" {self.gamma.reshape(1, -1)} h\n"
        s += "\tNonlinear Rep: h = \n"
        for i in range(self.dim_nonlinear):
            s = (
                s
                + f"\t   {self.hiden_feature.func_name[self.hiden_feature.func_idx[i]]} (z_{i + 1})\t"
            )
        return s


class DGPs:
    """
    It
    (1) set different DGPs
    (2) produce the corresponding true model parameters
    (3) sample data from the DGPs
    (4) generate dataloaders for training, validation and testing
    """

    def __init__(
        self,
        T=None,
        dim_linear=None,
        dim_nonlinear=None,
        dim_rep=None,
        rep_flag=None,
        coef_flag=None,
        feature_flag=None,
        n_train_target=None,
        n_train_source=None,
        batch_size=None,
        noise_level=None,
        scale=None,
        profile=None,
        n_knots=None,
        degree=None,
        verbose=False,
        seed=None,
    ):
        """
        parameters:
          T: number of source tasks
          dim_linear: dimension of linear part
          dim_nonlinear: dimension of nonlinear part
          dim_rep: dimension of representation
          rep_flag: 'default', 'linear', 'linear-factor'
          coef_flag: 'default', 'homogeneous', 'heterogeneous'
          feature_flag: 'default', 'iid-normal', 'iid-uniform', 'x_uniform-z_factor'
          n_train_target: number of training samples in target task
          n_train_source: number of training samples in source task
          batch_size: batch size
          noise_level: noise level
          sparsity_gamma (int): sparsity level of gamma, i.e., the number of nonzero elements in gamma
          profile:
            'lisai' for the setting in Li sai's paper
            'tianye' for the setting in Tian Ye's paper
        """
        self.T = T
        self.dim_linear = dim_linear
        self.dim_nonlinear = dim_nonlinear
        self.dim_rep = dim_rep
        self.rep_flag = rep_flag if rep_flag is not None else "default"
        self.coef_flag = coef_flag if coef_flag is not None else "default"
        self.feature_flag = feature_flag if feature_flag is not None else "default"
        self.n_train_target = n_train_target if n_train_target is not None else 100
        self.n_valid_target = self.n_train_target * 3 // 10
        self.n_test_target = 10000
        self.n_train_source = n_train_source if n_train_source is not None else 100
        self.n_valid_source = self.n_train_source * 3 // 10
        self.n_test_source = 10000
        self.batch_size = (
            batch_size if batch_size is not None else self.n_train_source
        )  # 128 self.n_train_source
        self.noise_level = noise_level if noise_level is not None else 0.3
        self.scale = scale if scale is not None else False
        self.verbose = verbose

        self.params = {}

        seed = seed if seed is not None else 2024
        self.seed = seed
        np.random.seed(self.seed)
        # make sure the coefficient is random but keep unchanged.

        if self.rep_flag == "default" or self.rep_flag == "linear":
            # this case means no representation layer, just a linear layer
            if self.dim_rep is None:
                self.dim_rep = self.dim_nonlinear
        elif self.rep_flag == "linear-factor":
            if self.dim_rep is None:
                ValueError(
                    "Please specify the dimension of representation for linear-factor design!"
                )
            self.params["A"] = np.random.randn(self.dim_nonlinear, self.dim_rep)
        elif self.rep_flag == "additive":
            if self.dim_rep is None:
                self.dim_rep = self.dim_nonlinear
        elif self.rep_flag == "additive-factor":
            if self.dim_rep is None:
                ValueError(
                    "Please specify the dimension of representation for additive-factor design!"
                )
            # self.params['A'] = np.reshape(np.random.uniform(-np.sqrt(3), np.sqrt(3), self.dim_nonlinear * self.dim_rep), (self.dim_nonlinear, self.dim_rep))
            # self.params['A'] = np.reshape(np.random.uniform(-1, 1, self.dim_nonlinear * self.dim_rep), (self.dim_nonlinear, self.dim_rep))
            self.params["A"] = np.random.randn(
                self.dim_nonlinear, self.dim_rep
            ) / np.sqrt(self.dim_nonlinear)
            # # set 80% elements of A to be zero:
            # idx = np.random.choice(self.dim_nonlinear*self.dim_rep, int(self.dim_nonlinear*self.dim_rep*0.5), replace=False)
            # self.params['A'].reshape(-1)[idx] = 0
            # # set 5 elements in each column A to be nonzero, others are set as zero:
            # self.params['A'] = np.zeros((self.dim_nonlinear, self.dim_rep))
            # for i in range(self.dim_rep):
            #   idx = np.random.choice(self.dim_nonlinear, 5, replace=False)
            #   self.params['A'][idx, i] = np.random.randn(5) / np.sqrt(5)
            # print("A: ", self.params['A'])

            # np.random.randn(self.dim_nonlinear, self.dim_rep)
        elif self.rep_flag == "deep":
            self.mymodel = MyDeepModel()
            self.mymodel.init_weights()
        else:
            raise ValueError("Unknown rep_flag: {}".format(self.rep_flag))

        if self.coef_flag == "default" or self.coef_flag == "homogeneous":
            if False:
                self.params["betas"] = [
                    np.random.uniform(-1, 1, size=(self.dim_linear, 1))
                ] * T
                self.params["gammas"] = [
                    np.random.uniform(-1, 1, size=(self.dim_rep, 1))
                ] * T
            else:
                self.params["betas"] = [np.random.randn(dim_linear, 1)] * T
                self.params["gammas"] = [np.random.randn(self.dim_rep, 1)] * T
            self.params["beta"] = self.params["betas"][0]
            self.params["gamma"] = self.params["gammas"][0]
        elif self.coef_flag == "heterogeneous":
            if False:  # use uniform distribution
                self.params["betas"] = [
                    np.random.uniform(-1, 1, size=(self.dim_linear, 1))
                    for t in range(T)
                ]
                self.params["gammas"] = [
                    np.random.uniform(-1, 1, size=(self.dim_rep, 1)) for t in range(T)
                ]
                self.params["beta"] = np.random.uniform(
                    -1, 1, size=(self.dim_linear, 1)
                )
                self.params["gamma"] = np.random.uniform(-1, 1, size=(self.dim_rep, 1))
            else:  # use normal distribution
                self.params["betas"] = [
                    np.random.randn(self.dim_linear, 1) for t in range(T)
                ]
                self.params["gammas"] = [
                    np.random.randn(self.dim_rep, 1) * 1 for t in range(T)
                ]

                # The following is the DRO settings
                # list_of_betas = [arr.squeeze() for arr in self.params['betas']]
                # tmp_source_betas = np.stack(list_of_betas, axis=1)
                # list_of_gammas = [arr.squeeze() for arr in self.params['gammas']]
                # tmp_source_gammas = np.stack(list_of_gammas, axis=1)
                # tmp_weight = np.ones((self.T, 1)) / self.T # weight for target model
                # self.params['beta'] = tmp_source_betas @ tmp_weight
                # self.params['gamma'] = tmp_source_gammas @ tmp_weight

                # the following commented contents are for the case of using random initialization
                self.params["beta"] = np.random.randn(self.dim_linear, 1) * 1
                self.params["gamma"] = np.random.randn(self.dim_rep, 1) * 1

        elif self.coef_flag == "similar":
            # betas are shared and gammas are similar across tasks (design 1 in Li Sai's paper, with h=6)
            self.params["beta"] = np.ones((self.dim_linear, 1))
            self.params["gamma"] = np.concatenate(
                (
                    [0.3] * self.sparsity_gamma,
                    [0] * (self.dim_nonlinear - self.sparsity_gamma),
                ),
                axis=0,
            ).reshape(-1, 1)
            self.params["betas"] = [self.params["beta"].copy() for t in range(self.T)]
            self.params["gammas"] = [self.params["gamma"].copy() for t in range(self.T)]
            for t in range(self.T):
                self.params["gammas"][t][0, 0] = -0.3
                idx = np.random.choice(self.dim_nonlinear, 6, replace=False)
                self.params["gammas"][t][idx, 0] = -0.3

        # define the representation functions
        self.func_zoo = [
            univariate_funcs.func0,
            univariate_funcs.func1,
            univariate_funcs.func2,
            univariate_funcs.func3,
            univariate_funcs.func4,
            univariate_funcs.func5,
            univariate_funcs.func6,
            univariate_funcs.func7,
            univariate_funcs.func8,
            univariate_funcs.func9,
            univariate_funcs.func10,
            univariate_funcs.func11,
            univariate_funcs.func12,
            univariate_funcs.func13,
        ]
        self.func_name = [
            "x",
            "sin",
            "root_abs",
            "sqrt_abs",
            "sigmoid",
            "cos_pi",
            "sin_pi",
            "cos_2",
            "-cos",
            "tan",
            "log",
            "exp",
            "square",
            "arctan",
        ]
        # scale the function to have unit variance
        if self.scale:
            self.params["func_scale"] = 2.0 / np.sqrt(
                np.array(
                    [
                        0.3333,
                        0.2727,
                        0.3671,
                        0.0889,
                        0.0189,
                        0.5000,
                        0.5000,
                        0.1987,
                        0.0193,
                        0.5876,
                        0.1905,
                        1.2792,
                        0.0889,
                        0.2453,
                    ]
                )
            )  # scale the function to have unit variance
        else:
            self.params["func_scale"] = np.ones(14)
        if True:
            self.params["func_idx"] = np.random.randint(0, 13, self.dim_rep)
            if self.rep_flag == "additive":
                self.params["func_idx"] = np.random.randint(1, 5, self.dim_rep)

                self.params["func_idx"][0] = 4
                if self.dim_rep >= 4:
                    self.params["func_idx"][4] = 4
        else:
            self.params["func_idx"] = np.arange(1, 1 + self.dim_rep)

        if self.rep_flag == "additive":
            # scale the function to have unit variance
            for i in range(self.dim_rep):
                self.params["gammas"][i] = self.params["gammas"][i] * self.params[
                    "func_scale"
                ][self.params["func_idx"]].reshape(-1, 1)
            # self.params['gamma'] = self.params['gammas'][0]
            # self.params['beta'] = self.params['betas'][0]
        else:
            # scale the beta to reduce the R2 of linear part
            pass
        self.params["beta"] = self.params["beta"] / 2.0

    def h(self, z):
        """
        input z and return a r-dimensional matrix.
        z is a n*p matrix
        output is a n*r matrix
        """
        if self.rep_flag == "default" or self.rep_flag == "linear":
            return z
        elif self.rep_flag == "linear-factor":
            return z @ self.params["A"]
        if self.rep_flag == "additive":
            # do nonlinear transformation on each column of z, the transformation is randomly chosen from func_zoo
            # z is a n*p matrix, output is a n*r matrix
            n = np.shape(z)[0]
            output = np.zeros((n, self.dim_rep))
            for i in range(self.dim_rep):
                output[:, i : i + 1] = self.func_zoo[self.params["func_idx"][i]](
                    z[:, i : i + 1]
                )
            return output
        elif self.rep_flag == "additive-factor":
            # do nonlinear transformation on each column of z @ A, the transformation is randomly chosen from func_zoo
            # z is a n*p matrix
            # output is a n*r matrix
            n = np.shape(z)[0]
            output = np.zeros((n, self.dim_rep))
            tmp_z = z @ self.params["A"]
            for i in range(self.dim_rep):
                output[:, i : i + 1] = self.func_zoo[self.params["func_idx"][i]](
                    tmp_z[:, i : i + 1]
                )
            return output
        elif self.rep_flag == "deep":
            tmp_z = torch.tensor(z, dtype=torch.float32)
            # print(f"shape of tmp_z: {tmp_z.shape}")
            return self.mymodel(tmp_z).detach().numpy()

    def tune_knots(self):
        """
        tune the knots for spline method
        """
        # initialize the spline method
        n_knots_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # , 16, 17, 18, 19, 20
        loss_val_list = []
        for n_knots in n_knots_list:
            self.sample(n_knots=n_knots, seed=self.seed)
            # use ridge spline method to fit the generated data and check the validation loss
            X = self.data_target["feature_train_spline"]
            Y = self.data_target["label_train"]

            lm = LinearRegression(
                fit_intercept=False
            )  # RidgeCV(cv=5, fit_intercept=False)
            lm.fit(X, Y)
            tmp_valid_loss = np.mean(
                (
                    lm.predict(self.data_target["feature_valid_spline"])
                    - self.data_target["label_valid"]
                )
                ** 2
            )
            loss_val_list.append(tmp_valid_loss)

        def find_min_index(lst):
            min_val = min(lst)
            min_index = lst.index(min_val)
            return min_index

        # chooose the best n_knots according to the validation loss
        best_n_knots = n_knots_list[find_min_index(loss_val_list)]

        return best_n_knots

    def sample(self, n_knots=11, seed=2024):
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        """
    generate a list of dataloaders
    """
        # initialize the spline method
        min_val, max_val = -1.0, 1.0
        # n_knots = 11
        degree = 3
        inner_knots = np.linspace(min_val, max_val, n_knots - 2)
        common_knots = np.concatenate(
            ([inner_knots[0]] * degree, inner_knots, [inner_knots[-1]] * degree)
        )

        # generate source data
        self.dataloaders_source_train = []
        self.dataloaders_source_valid = []
        self.dataloaders_source_test = []
        self.data_source = {
            "feature_train": [],
            "feature_train_spline": [],
            "label_train": [],
            "feature_valid": [],
            "label_valid": [],
            "feature_test": [],
            "label_test": [],
        }
        for t in range(self.T):
            if self.feature_flag == "default" or self.feature_flag == "iid-normal":
                all_feature = np.random.randn(
                    self.n_train_source + self.n_valid_source + self.n_test_source,
                    self.dim_linear + self.dim_nonlinear,
                )
            elif self.feature_flag == "iid-uniform":
                all_feature = np.random.uniform(
                    -1,
                    1,
                    (
                        self.n_train_source + self.n_valid_source + self.n_test_source,
                        self.dim_linear + self.dim_nonlinear,
                    ),
                )
                # uniform range 1, 1.5
            elif self.feature_flag == "x_uniform-z_factor":
                # unfinished case
                x = np.random.uniform(
                    -1,
                    1,
                    (
                        self.n_train_source + self.n_valid_source + self.n_test_source,
                        self.dim_linear,
                    ),
                )
                z = np.random.randn(
                    self.n_train_source + self.n_valid_source + self.n_test_source,
                    self.dim_nonlinear,
                )
                all_feature = np.concatenate((x, z), axis=1)
            elif self.feature_flag == "compound-normal":
                mean_value = np.zeros(self.dim_linear + self.dim_nonlinear)
                cov = (
                    np.ones(
                        (
                            self.dim_linear + self.dim_nonlinear,
                            self.dim_linear + self.dim_nonlinear,
                        )
                    )
                    * 0.5
                )
                # set the diagonal of cov to be 1
                row, col = np.diag_indices_from(cov)
                cov[row, col] = 1
                all_feature = np.random.multivariate_normal(
                    mean_value,
                    cov,
                    (self.n_train_source + self.n_valid_source + self.n_test_source),
                )
            else:
                raise ValueError("Unknown feature_flag: {}".format(self.feature_flag))

            x = all_feature[:, : self.dim_linear]
            z = all_feature[:, self.dim_linear :]
            y_true = x @ self.params["betas"][t] + self.h(z) @ self.params["gammas"][t]
            y = (
                y_true
                + np.random.randn(
                    self.n_train_source + self.n_valid_source + self.n_test_source, 1
                )
                * self.noise_level
            )

            all_feature_train = all_feature[: self.n_train_source, :]
            y_train = y[: self.n_train_source, :]
            all_feature_valid = all_feature[
                self.n_train_source : self.n_train_source + self.n_valid_source, :
            ]
            y_valid = y[
                self.n_train_source : self.n_train_source + self.n_valid_source, :
            ]
            all_feature_test = all_feature[
                self.n_train_source + self.n_valid_source :, :
            ]
            y_test = y_true[self.n_train_source + self.n_valid_source :, :]

            self.data_source["feature_train"].append(all_feature_train)
            self.data_source["label_train"].append(y_train)
            self.data_source["feature_valid"].append(all_feature_valid)
            self.data_source["label_valid"].append(y_valid)
            self.data_source["feature_test"].append(all_feature_test)
            self.data_source["label_test"].append(y_test)

            dataloader_train = DataLoader(
                RegressionDataset(all_feature_train, y_train),
                batch_size=self.batch_size,
                shuffle=True,
            )
            dataloader_valid = DataLoader(
                RegressionDataset(all_feature_valid, y_valid),
                batch_size=self.batch_size,
                shuffle=True,
            )
            dataloader_test = DataLoader(
                RegressionDataset(all_feature_test, y_test),
                batch_size=self.batch_size,
                shuffle=False,
            )
            self.dataloaders_source_train.append(dataloader_train)
            self.dataloaders_source_valid.append(dataloader_valid)
            self.dataloaders_source_test.append(dataloader_test)

            # transform the z part using spline method, x is kept unchanged
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_spline = np.concatenate((x, z_spline), axis=1)
            self.data_source["feature_train_spline"].append(
                all_feature_spline[: self.n_train_source, :]
            )

        # generate target data
        if self.feature_flag == "default" or self.feature_flag == "iid-normal":
            all_feature = np.random.randn(
                self.n_train_target + self.n_valid_target + self.n_test_target,
                self.dim_linear + self.dim_nonlinear,
            )
        elif self.feature_flag == "iid-uniform":
            all_feature = np.random.uniform(
                min_val,
                max_val,
                (
                    self.n_train_target + self.n_valid_target + self.n_test_target,
                    self.dim_linear + self.dim_nonlinear,
                ),
            )
        elif self.feature_flag == "x_uniform-z_factor":
            x = np.random.uniform(
                -1,
                1,
                (
                    self.n_train_target + self.n_valid_target + self.n_test_target,
                    self.dim_linear,
                ),
            )
            z = np.random.randn(
                self.n_train_target + self.n_valid_target + self.n_test_target,
                self.dim_nonlinear,
            )
            all_feature = np.concatenate((x, z), axis=1)
        elif self.feature_flag == "compound-normal":
            mean_value = np.zeros(self.dim_linear + self.dim_nonlinear)
            cov = (
                np.ones(
                    (
                        self.dim_linear + self.dim_nonlinear,
                        self.dim_linear + self.dim_nonlinear,
                    )
                )
                * 0.8
            )
            # set the diagonal of cov to be 1
            row, col = np.diag_indices_from(cov)
            cov[row, col] = 1
            all_feature = np.random.multivariate_normal(
                mean_value,
                cov,
                (self.n_train_target + self.n_valid_target + self.n_test_target),
            )
        else:
            raise ValueError("Unknown feature_flag: {}".format(self.feature_flag))

        # calculate the R-square of target model
        x = all_feature[:, : self.dim_linear]
        z = all_feature[:, self.dim_linear :]
        y_true = x @ self.params["beta"] + self.h(z) @ self.params["gamma"]
        y = (
            y_true
            + np.random.randn(
                self.n_train_target + self.n_valid_target + self.n_test_target, 1
            )
            * self.noise_level
        )
        R2_X = np.var(x @ self.params["beta"]) / np.var(y)
        R2_eps = np.var(y_true) / np.var(y)
        R2_h = np.var(self.h(z) @ self.params["gamma"]) / np.var(y)
        if self.verbose:
            print(f"[DGP] R2 of target model linear part: {R2_X}")
        if self.verbose:
            print(f"[DGP] R2 of target model non-linear part: {R2_h}")
        if self.verbose:
            print(f"[DGP] R2 of target model: {R2_eps}")
        # end of calculating R-square

        # data perprocessing
        scaler = StandardScaler()
        all_feature_scaled = np.concatenate(
            (
                all_feature[:, : self.dim_linear],
                scaler.fit_transform(all_feature[:, self.dim_linear :]),
            ),
            axis=1,
        )
        # end of data preprocessing

        dataloader_train = DataLoader(
            RegressionDataset(
                all_feature_scaled[: self.n_train_target, :],
                y[: self.n_train_target, :],
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        dataloader_valid = DataLoader(
            RegressionDataset(
                all_feature_scaled[
                    self.n_train_target : self.n_train_target + self.n_valid_target, :
                ],
                y[self.n_train_target : self.n_train_target + self.n_valid_target, :],
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        dataloader_test = DataLoader(
            RegressionDataset(
                all_feature_scaled[self.n_train_target + self.n_valid_target :, :],
                y_true[self.n_train_target + self.n_valid_target :, :],
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.dataloader_target_train = dataloader_train
        self.dataloader_target_valid = dataloader_valid
        self.dataloader_target_test = dataloader_test

        # transform the z part using spline method, x is kept unchanged
        if True:
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_spline = np.concatenate((x, z_spline), axis=1)
        else:
            all_feature_spline = np.zeros(
                (
                    self.n_train_target + self.n_valid_target + self.n_test_target,
                    self.dim_linear + self.dim_nonlinear,
                )
            )

        self.data_target = {
            "feature_train": all_feature[: self.n_train_target, :],
            "feature_train_spline": all_feature_spline[: self.n_train_target, :],
            "label_train": y[: self.n_train_target, :],
            "feature_valid": all_feature[
                self.n_train_target : self.n_train_target + self.n_valid_target, :
            ],
            "feature_valid_spline": all_feature_spline[
                self.n_train_target : (self.n_train_target + self.n_valid_target), :
            ],
            "label_valid": y[
                self.n_train_target : (self.n_train_target + self.n_valid_target), :
            ],
            "feature_test": all_feature[self.n_train_target + self.n_valid_target :, :],
            "feature_test_spline": all_feature_spline[
                (self.n_train_target + self.n_valid_target) :, :
            ],
            "label_test": y_true[(self.n_train_target + self.n_valid_target) :, :],
        }

    def __str__(self):
        s = "DGPs: \n"
        s += f"\tRep design: {self.rep_flag}, Feature design: {self.feature_flag}, Coef design: {self.coef_flag}\n"
        s += f"\tLinear coeff: {self.params['beta'].T}\n"
        s += f"\tNonlinear coeff: {self.params['gamma'].T}\n"
        if self.rep_flag != "deep":
            s += f"\tRep functions:{[self.func_name[idx] for idx in self.params['func_idx']]}"
        elif self.rep_flag == "deep":
            s += f"\tRep functions:\n{self.mymodel}"

        return s


# Define the neural network model
class MyDeepModel(torch.nn.Module):
    """
  We use the following structure to generate data
           h1()       h2()       h3()      h4()    h5() 
          /    \     /    \     /    \     / \       |
         /      \   /      \   /      \   /   |      |
        /        \ /        \ /        \ /    |      |
      g1()       g2()       g3()       g4()  g5()  g6() 
      /  \       /  \       /  \       /  \  /  \  /  \ 
     /    \     /    \     /    \     /    \/    \/    \ 
    z1    z2   z3    z4   z5    z6   z7    z8    z9   z10
  """

    def __init__(self):
        super(MyDeepModel, self).__init__()

        def func0(x):
            return x

        def func1(x):
            return torch.sin(x)

        def func2(x):
            return torch.sqrt(torch.abs(x)) * 2 - 1

        def func3(x):
            x = torch.where(x > 1.5, torch.tensor(1.5), x)
            return (1 - torch.abs(x)) ** 2

        def func4(x):
            return 1 / (1.0 + torch.exp(-x))

        def func5(x):
            return torch.cos(torch.pi * x / 2)

        def func6(x):
            return torch.sin(torch.pi * x / 2)

        def func7(x):
            return -torch.sin(x)

        def func8(x):
            return torch.cos(2 * x)

        def func9(x):
            x = torch.where(x > 1.0, torch.tensor(1.0), x)
            x = torch.where(x < -1.0, torch.tensor(-1.0), x)
            return torch.tan(x + 0.1)

        def func10(x):
            return torch.log(torch.abs(x + 2))

        def func11(x):
            x = torch.where(x > 1.0, torch.tensor(1.0), x)
            return torch.exp(x) - 1

        def func12(x):
            x = torch.where(x > 2.0, torch.tensor(2.0), x)
            return x**2

        self.func_name = [
            "x",
            "sin",
            "root_abs",
            "sqrt_abs",
            "sigmoid",
            "cos_pi",
            "sin_pi",
            "-sin",
            "cos_2",
            "tan",
            "log",
            "exp",
            "square",
        ]

        # Define layers for g1 to g6
        # Assuming each g layer is a linear layer followed by an activation function
        self.g1 = nn.Linear(2, 1)
        self.g2 = nn.Linear(2, 1)
        self.g3 = nn.Linear(2, 1)
        self.g4 = nn.Linear(2, 1)
        self.g5 = nn.Linear(2, 1)
        self.g6 = nn.Linear(2, 1)

        # Define layers for h1 to h5
        # These layers process the outputs of the g layers
        self.h1 = nn.Linear(2, 1)
        self.h2 = nn.Linear(2, 1)
        self.h3 = nn.Linear(2, 1)
        self.h4 = nn.Linear(2, 1)
        self.h5 = nn.Linear(1, 1)

        self.func_zoo = [
            func0,
            func1,
            func2,
            func3,
            func4,
            func5,
            func6,
            func7,
            func8,
            func9,
            func10,
            func11,
            func12,
        ]
        # random choose the functions from the function zoo with replacement
        self.func_idx = np.random.randint(0, 13, 11)
        # np.array([5, 11, 12, 8, 9, 11, 5, 0, 0, 1, 12])
        # np.random.randint(0, 13, 11)
        # print(f"func_idx: {self.func_idx}")

    def init_weights(self):
        # Custom weight initialization
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.75)  # Set all weights
                nn.init.constant_(layer.bias, 0.0)  # Set all biases to 0

    def forward(self, z):
        # Split the input z into its components
        z1, z2, z3, z4, z5, z6, z7, z8, z9, z10 = (
            z[:, 0:1],
            z[:, 1:2],
            z[:, 2:3],
            z[:, 3:4],
            z[:, 4:5],
            z[:, 5:6],
            z[:, 6:7],
            z[:, 7:8],
            z[:, 8:9],
            z[:, 9:10],
        )

        # Process inputs through g layers
        g1_output = self.func_zoo[self.func_idx[0]](self.g1(torch.cat((z1, z2), dim=1)))
        g2_output = self.func_zoo[self.func_idx[1]](self.g2(torch.cat((z3, z4), dim=1)))
        g3_output = self.func_zoo[self.func_idx[2]](self.g3(torch.cat((z5, z6), dim=1)))
        g4_output = self.func_zoo[self.func_idx[3]](self.g4(torch.cat((z7, z8), dim=1)))
        g5_output = self.func_zoo[self.func_idx[4]](self.g5(torch.cat((z8, z9), dim=1)))
        g6_output = self.func_zoo[self.func_idx[5]](
            self.g6(torch.cat((z9, z10), dim=1))
        )

        # Process g outputs through h layers
        h1_output = self.func_zoo[self.func_idx[6]](
            self.h1(torch.cat((g1_output, g2_output), dim=1))
        )
        h2_output = self.func_zoo[self.func_idx[7]](
            self.h2(torch.cat((g2_output, g3_output), dim=1))
        )
        h3_output = self.func_zoo[self.func_idx[8]](
            self.h3(torch.cat((g3_output, g4_output), dim=1))
        )
        h4_output = self.func_zoo[self.func_idx[9]](
            self.h4(torch.cat((g4_output, g5_output), dim=1))
        )
        h5_output = self.func_zoo[self.func_idx[10]](self.h5(g6_output))

        # Return the final output
        return torch.cat(
            (h1_output, h2_output, h3_output, h4_output, h5_output), axis=1
        )

    def __str__(self):
        # Build the tree representation string
        tree_str = ""
        # tree_str += "MyDeepModel Tree Structure:\n"
        tree_str += f"\t\t       f{self.func_idx[6]}()       f{self.func_idx[7]}()       f{self.func_idx[8]}()      f{self.func_idx[9]}()    f{self.func_idx[10]}()\n"
        tree_str += f"\t\t      /    \     /    \     /    \     / \       |\n"
        tree_str += f"\t\t     /      \   /      \   /      \   /   |      |\n"
        tree_str += f"\t\t    /        \ /        \ /        \ /    |      |\n"
        tree_str += f"\t\t  f{self.func_idx[0]}()      f{self.func_idx[1]}()       f{self.func_idx[2]}()       f{self.func_idx[3]}()  f{self.func_idx[4]}()  f{self.func_idx[5]}() \n"
        tree_str += f"\t\t  /  \       /  \       /  \       /  \  /  \  /  \ \n"
        tree_str += f"\t\t /    \     /    \     /    \     /    \/    \/    \ \n"
        tree_str += f"\t\tz1    z2   z3    z4   z5    z6   z7    z8    z9   z10"

        return tree_str


# generate some random data to test MyDeepModel
# z = torch.rand((20, 10))
# print(f"shape of z: {z.shape}")
# model = MyDeepModel()
# y_pred = model(z)
# print(y_pred.shape)


# %%
class RealData:
    """
    This class is used to load real data from UCI datasets
    """

    def __init__(self, data_name=None, flag_test=False, verbose=False):
        """
        parameters:
          data_name: name of the dataset
        """
        self.data_name = data_name
        self.flag_test = flag_test
        self.scaler = MinMaxScaler()
        self.verbose = verbose

        if self.data_name == "penn_jae":
            self.dim_linear = None
            self.dim_nonlinear = None
            self.T = None
            self.fetch_bouns()
            if flag_test:
                self.tune_knots()
        elif self.data_name == "credit_card":
            self.dim_linear = 5
            self.dim_nonlinear = 5
            self.T = 5  # total 5 datasets, 1 target and 4 source
            self.fetch_credit()
            if flag_test:
                self.tune_knots()
            raise ValueError("Not supported yet!")
        elif self.data_name == "house_rent":
            self.dim_linear = None
            self.dim_nonlinear = None
            self.T = None
            self.fetch_house()
            if flag_test:
                self.tune_knots()
        else:
            raise ValueError("Unknown data_name: {}".format(self.data_name))

    def fetch_bouns(self, idx_target=4):
        raw_data = pd.read_csv(
            "./real_data/penn_jae/penn_jae.dat", delim_whitespace=True
        )
        raw_data = raw_data.sample(frac=1.0)
        T = len(raw_data["tg"].unique()) - 1  # total 7 datasets, 1 target and 6 sources
        # print(f"Total {T+1} datasets, 1 target and {T} sources")
        # sample size: [3354, 1385, 2428, 1885, 1745, 1831, 1285]
        sample_size = raw_data["tg"].value_counts().sort_index().values
        # target data
        if self.flag_test:
            n_target_test = int(sample_size[idx_target] * 0.33)
            n_target_train = int(sample_size[idx_target] * 0.33)
        else:
            n_target_test = 0
            n_target_train = int((sample_size[idx_target] - n_target_test) * 0.8)
        n_target_valid = int(sample_size[idx_target] - n_target_train - n_target_test)
        self.n_train_target = n_target_train
        # source data
        n_source_train = [int(sample_size[i] * 0.8) for i in range(T + 1)]
        n_source_valid = [int(sample_size[i] - n_source_train[i]) for i in range(T + 1)]
        n_source_train.pop(idx_target)
        n_source_valid.pop(idx_target)

        # data preprocessing
        data = raw_data.copy()
        data["inuidur1"] = np.log(data["inuidur1"])
        # set 'inuidur1' to be zero mean for each 'tg'
        data["inuidur1"] = data.groupby("tg", group_keys=False)["inuidur1"].apply(
            lambda x: x - np.mean(x)
        )
        dummy_enc = OneHotEncoder(drop="first", categories="auto").fit(
            data.loc[:, ["dep"]]
        )
        xx = dummy_enc.transform(data.loc[:, ["dep"]]).toarray()
        data["dep1"] = xx[:, 0]
        data["dep2"] = xx[:, 1]

        y_col = "inuidur1"
        d_cols = ["tg"]
        x_cols = ["female"]
        z_cols = [
            "black",
            "othrace",
            "dep1",
            "dep2",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "agelt35",
            "agegt54",
            "durable",
            "lusd",
            "husd",
        ]
        self.dim_linear = len(x_cols)
        self.dim_nonlinear = len(z_cols)
        self.T = T

        # # standardize the data's z part for each task 'tg' separately
        # for i in range(T+1):
        #   data.loc[data['tg'] == i, z_cols] = self.scaler.fit_transform(data.loc[data['tg'] == i, z_cols])
        # # data.loc[:, z_cols] = self.scaler.fit_transform(data.loc[:, z_cols])

        # target data
        data_tmp1 = data.copy()[raw_data["tg"] == idx_target]
        data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])
        data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)
        self.dataloader_target_train = DataLoader(
            RegressionDataset(
                data_tmp2[:n_target_train, :], data_tmp3[:n_target_train]
            ),
            batch_size=n_target_train,
            shuffle=True,
        )
        self.dataloader_target_valid = DataLoader(
            RegressionDataset(
                data_tmp2[n_target_train : (n_target_train + n_target_valid), :],
                data_tmp3[n_target_train : (n_target_train + n_target_valid)],
            ),
            batch_size=n_target_valid,
            shuffle=True,
        )
        if n_target_test > 0:
            self.dataloader_target_test = DataLoader(
                RegressionDataset(
                    data_tmp2[(n_target_train + n_target_valid) :, :],
                    data_tmp3[(n_target_train + n_target_valid) :],
                ),
                batch_size=n_target_test,
                shuffle=True,
            )
        else:
            self.dataloader_target_test = None
        self.data_target = {
            # train
            "feature_train": data_tmp2[:n_target_train, :],
            "feature_train_spline": [],
            "label_train": data_tmp3[:n_target_train],
            # valid
            "feature_valid": data_tmp2[
                n_target_train : (n_target_train + n_target_valid), :
            ],
            "feature_valid_spline": [],
            "label_valid": data_tmp3[
                n_target_train : (n_target_train + n_target_valid)
            ],
            # test
            "feature_test": data_tmp2[(n_target_train + n_target_valid) :, :],
            "feature_test_spline": [],
            "label_test": data_tmp3[(n_target_train + n_target_valid) :],
        }

        # source data
        self.dataloaders_source_train = []
        self.dataloaders_source_valid = []
        self.data_source = {
            # train
            "feature_train": [],
            "feature_train_spline": [],
            "label_train": [],
            # valid
            "feature_valid": [],
            "feature_valid_spline": [],
            "label_valid": [],
        }
        idx_source = np.setdiff1d(np.arange(T + 1), [idx_target])
        for i in range(len(idx_source)):
            data_tmp1 = data.copy()[raw_data["tg"] == idx_source[i]]
            data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])
            data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)
            self.data_source["feature_train"].append(data_tmp2[: n_source_train[i], :])
            self.data_source["label_train"].append(data_tmp3[: n_source_train[i], :])
            self.data_source["feature_valid"].append(data_tmp2[n_source_train[i] :, :])
            self.data_source["label_valid"].append(data_tmp3[n_source_train[i] :, :])
            dataloader_train = DataLoader(
                RegressionDataset(
                    data_tmp2[: n_source_train[i], :], data_tmp3[: n_source_train[i]]
                ),
                batch_size=n_source_train[i],
                shuffle=True,
            )
            dataloader_valid = DataLoader(
                RegressionDataset(
                    data_tmp2[n_source_train[i] :, :], data_tmp3[n_source_train[i] :]
                ),
                batch_size=n_source_valid[i],
                shuffle=True,
            )
            self.dataloaders_source_train.append(dataloader_train)
            self.dataloaders_source_valid.append(dataloader_valid)

    def fetch_credit(self, idx_target=0):
        """
        This dataset is removed from the manuscript, because it is 0-1 outcome, not continuous outcome.
        We analyze the data “default of credit card clients” of an important bank in Taiwan, publicly available at the UCI machine learning repository. We consider five populations of clients, with credit scores equal to 10k, 110k, 210k, 310k, and 410k, respectively.
        The sizes of the samples from the five populations are respectively 493, 588, 730, 272, and 78.
        Note that if our interest had been prediction in the group with credit score, say lower than 300k, and if we have evidence that the effect from other covariates may be identical in the groups of credit score less than and more than 300k respectively, then we would have divided the sample into two groups.
        """
        data_raw = pd.read_csv("./real_data/credit_card/cc.csv", sep=",")
        data_raw = data_raw.sample(frac=1.0)
        # print(data_raw.iloc[0:5, :])

        # Filter and clean the data
        tmp_data_clean = {}
        tmp_data_clean["cc10k"] = data_raw[data_raw["X1"] == 10000].drop(
            columns=["X0", "X1", "X11"]
        )
        tmp_data_clean["cc110k"] = data_raw[data_raw["X1"] == 110000].drop(
            columns=["X0", "X1", "X11"]
        )
        tmp_data_clean["cc210k"] = data_raw[data_raw["X1"] == 210000].drop(
            columns=["X0", "X1", "X11"]
        )
        tmp_data_clean["cc310k"] = data_raw[data_raw["X1"] == 310000].drop(
            columns=["X0", "X1", "X11"]
        )
        tmp_data_clean["cc260k"] = data_raw[data_raw["X1"] == 410000].drop(
            columns=["X0", "X1", "X11"]
        )
        tmp_data_clean["cc90k"] = data_raw[data_raw["X1"] >= 510000].drop(
            columns=["X0", "X1", "X11"]
        )

        T = 5  # total 6 datasets, 1 target and 5 source
        self.T = T
        sample_size = [493, 588, 730, 272, 78, 206]
        idx_target = 4
        if self.flag_test:
            n_target_test = int(sample_size[idx_target] * 0)
        else:
            n_target_test = 40
        if self.flag_test:
            n_target_train = int(sample_size[idx_target] * 0.5)
        else:
            n_target_train = int((sample_size[idx_target] - n_target_test) * 0.8)
        n_target_valid = int(sample_size[idx_target] - n_target_train - n_target_test)
        self.n_train_target = n_target_train
        # source data
        n_source_train = [int(sample_size[i] * 0.8) for i in range(T + 1)]
        n_source_valid = [int(sample_size[i] - n_source_train[i]) for i in range(T + 1)]
        n_source_train.pop(idx_target)
        n_source_valid.pop(idx_target)

        def cleanccdata(ccdata):
            # Apply transformations based on the conditions
            ccdata["X2"] = ccdata["X2"].replace(2, 0)
            ccdata["X3"] = ccdata["X3"].replace(2, 1)
            ccdata["X3"] = ccdata["X3"].apply(lambda x: 0 if x >= 3 else x)
            ccdata["X4"] = ccdata["X4"].apply(lambda x: 0 if x >= 2 else x)

            # Apply transformations involving ratios
            for col in ["X6", "X7", "X8", "X9", "X10"]:
                x_col = f"X{int(col[1:]) + 12}"
                ccdata[col] = np.where(
                    ccdata[x_col] == 0,
                    1,
                    ccdata[f"X{int(col[1:]) + 7}"] / ccdata[x_col],
                )
                ccdata[col] = ccdata[col].apply(lambda x: 1 if x > 1 or x < 0 else x)

            return ccdata

        def addinter(ccdata):
            # Add interception term and rearrange columns
            ccdata["intercept"] = np.ones(len(ccdata))
            return ccdata[
                [
                    "intercept",
                    "X2",
                    "X3",
                    "X4",
                    "X5",
                    "X6",
                    "X7",
                    "X8",
                    "X9",
                    "X10",
                    "X12",
                    "Y",
                ]
            ]

        # Apply the functions to the subsets of data
        data_clean = {}
        for key, df in tmp_data_clean.items():
            cleaned_df = cleanccdata(df)
            data_clean[key] = cleaned_df[
                ["X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X12", "Y"]
            ]
            data_clean[key] = addinter(cleaned_df)

        raw_data = data_clean.copy()
        for key, df in raw_data.items():
            raw_data[key] = df.sample(frac=1.0)
        data = raw_data.copy()
        # add the intercept term to each dataset
        for key, df in data.items():
            df["intercept"] = np.ones(len(df))

        name_list = ["cc10k", "cc110k", "cc210k", "cc310k", "cc260k", "cc90k"]

        y_col = "Y"
        x_cols = ["intercept", "X2"]
        z_cols = ["X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X12"]
        self.dim_linear = len(x_cols)
        self.dim_nonlinear = len(z_cols)

        # target data
        data_tmp1 = data[name_list[idx_target]].copy()
        data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])
        data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)
        dataloader_train = DataLoader(
            RegressionDataset(
                data_tmp2[:n_target_train, :], data_tmp3[:n_target_train]
            ),
            batch_size=n_target_train,
            shuffle=True,
        )
        # print(f"shape of train data: {data_tmp2[:n_target_train, :].shape}")
        # print(f"sample size of train data: {n_target_train}")
        # print(f"shape of valid data: {data_tmp2[n_target_train:(n_target_train+n_target_valid), :].shape}")
        dataloader_valid = DataLoader(
            RegressionDataset(
                data_tmp2[n_target_train : (n_target_train + n_target_valid), :],
                data_tmp3[n_target_train : (n_target_train + n_target_valid)],
            ),
            batch_size=n_target_valid,
            shuffle=True,
        )
        if self.flag_test:
            dataloader_test = DataLoader(
                RegressionDataset(
                    data_tmp2[(n_target_train + n_target_valid) :, :],
                    data_tmp3[(n_target_train + n_target_valid) :],
                ),
                batch_size=n_target_test,
                shuffle=True,
            )
        else:
            dataloader_test = None
        self.dataloader_target_train = dataloader_train
        self.dataloader_target_valid = dataloader_valid
        self.dataloader_target_test = dataloader_test
        self.data_target = {
            # train
            "feature_train": data_tmp2[:n_target_train, :],
            "feature_train_spline": [],
            "label_train": data_tmp3[:n_target_train],
            # valid
            "feature_valid": data_tmp2[
                n_target_train : (n_target_train + n_target_test), :
            ],
            "feature_valid_spline": [],
            "label_valid": data_tmp3[n_target_train : (n_target_train + n_target_test)],
            # test
            "feature_test": data_tmp2[(n_target_train + n_target_test) :, :],
            "feature_test_spline": [],
            "label_test": data_tmp3[(n_target_train + n_target_test) :],
        }

        # source data
        self.dataloaders_source_train = []
        self.dataloaders_source_valid = []
        self.data_source = {
            # train
            "feature_train": [],
            "feature_train_spline": [],
            "label_train": [],
            # valid
            "feature_valid": [],
            "feature_valid_spline": [],
            "label_valid": [],
        }
        idx_source = np.setdiff1d(np.arange(T + 1), [idx_target])
        name_list_source = name_list.copy()
        name_list_source.pop(idx_target)
        for i in range(len(idx_source)):
            data_tmp1 = data[name_list_source[i]].copy()
            data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])  # X, Z
            data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)  # Y
            self.data_source["feature_train"].append(data_tmp2[: n_source_train[i], :])
            self.data_source["label_train"].append(data_tmp3[: n_source_train[i], :])
            self.data_source["feature_valid"].append(data_tmp2[n_source_train[i] :, :])
            self.data_source["label_valid"].append(data_tmp3[n_source_train[i] :, :])
            dataloader_train = DataLoader(
                RegressionDataset(
                    data_tmp2[: n_source_train[i], :], data_tmp3[: n_source_train[i]]
                ),
                batch_size=n_source_train[i],
                shuffle=True,
            )
            dataloader_valid = DataLoader(
                RegressionDataset(
                    data_tmp2[n_source_train[i] :, :], data_tmp3[n_source_train[i] :]
                ),
                batch_size=n_source_valid[i],
                shuffle=True,
            )
            self.dataloaders_source_train.append(dataloader_train)
            self.dataloaders_source_valid.append(dataloader_valid)

    def fetch_house(self, idx_target=0):
        """
        中国租房信息数据, 来自 http://www.idatascience.cn/dataset-detail?table_id=100086
        有 4 个变化较大的变量, 其他变量主要是 0-1 的分类变量或变化范围较小的变量.
        """
        data_raw = pd.read_csv("./real_data/house_rent/house_rent.csv", sep=",")
        # rename the '区' in '上海', (+'a'), to avoid the same name with '深圳'
        data_raw.loc[data_raw["城市"] == "上海", "区"] = data_raw.loc[
            data_raw["城市"] == "上海", "区"
        ].apply(lambda x: x + "a")
        data_raw = data_raw[
            [
                "城市",
                "区",
                "价格",
                "室",
                "卫",
                "厅",
                "面积",
                "所属楼层",
                "总楼层",
                "周边学校个数",
                "周边医院个数",
                "是否有床",
                "是否有衣柜",
                "是否有空调",
                "是否有燃气",
            ]
        ]
        # data_raw = data_raw[data_raw['城市'] == '上海'] # select the target city
        # log transform the price
        # data_raw['价格'] = np.log(data_raw['价格'])

        T = (
            len(data_raw["区"].unique()) - 1
        )  # we use different districts as different source domains and set one district as the target domain
        self.T = T
        district_list = data_raw["区"].unique()
        if self.verbose:
            print(f"district_list: {district_list}")
        district_list = [str(district_list[i]) for i in range(len(district_list))]
        sample_size = [
            len(data_raw[data_raw["区"] == district_list[i]]) for i in range(self.T + 1)
        ]

        y_col = "价格"
        x_cols = ["面积", "周边学校个数"]
        z_cols = [
            "室",
            "卫",
            "厅",
            "所属楼层",
            "总楼层",
            "周边医院个数",
            "是否有床",
            "是否有衣柜",
            "是否有空调",
            "是否有燃气",
        ]
        self.dim_linear = len(x_cols)
        self.dim_nonlinear = len(z_cols)

        # # # plot the '所属楼层' vs '价格' for '通州'
        # y_tmp = data_raw[data_raw['区'] == '昌平']['价格']
        # plt.scatter(data_raw[data_raw['区'] == '昌平']['面积'], y_tmp, alpha=0.5, s=40)
        # # y_tmp = data_raw[data_raw['区'] == '普陀a']['价格']
        # # plt.scatter(data_raw[data_raw['区'] == '普陀a']['周边学校个数'], y_tmp, alpha=0.5, marker='>', s=40)
        # y_tmp = data_raw[data_raw['区'] == '奉贤a']['价格']
        # # y_tmp = y_tmp - np.mean(y_tmp)
        # plt.scatter(data_raw[data_raw['区'] == '奉贤a']['面积'], y_tmp, alpha=0.5, marker='s', s=40)
        # y_tmp = data_raw[data_raw['区'] == '盐田']['价格']
        # plt.scatter(data_raw[data_raw['区'] == '盐田']['面积'], y_tmp, alpha=0.5, marker='^', s=40)
        # plt.ylim(bottom=0, top=15000)
        # plt.xlim(left=0, right=200)
        # plt.xlabel('Num of schools', fontsize=14)
        # plt.ylabel('Price', fontsize=14)
        # plt.legend(['Changping', 'Fengxian', 'Yantian'], loc='upper left', fontsize=12)
        # # plt.xticks(fontsize=12)
        # # plt.yticks(fontsize=12)
        # plt.savefig('area_price.pdf')
        # plt.show()

        # raise ValueError("Not supported yet!")

        # re-sample the data, this will change the order of the '区'
        data_raw = data_raw.sample(frac=1.0)

        # standardize the continuous variables on totoal data
        print(
            f"the std of price/area: {np.std(data_raw['价格'])/np.std(data_raw['面积'])}"
        )
        print(
            f"the std of price/numschool: {np.std(data_raw['价格'])/np.std(data_raw['周边学校个数'])}"
        )
        # raise ValueError("Not supported yet!")

        for col in [
            "价格",
            "面积",
            "周边学校个数",
            "所属楼层",
            "总楼层",
            "周边医院个数",
        ]:  # ['价格', '面积', '周边学校个数']
            data_raw[col] = (data_raw[col] - np.mean(data_raw[col])) / np.std(
                data_raw[col]
            )
        # standardize the continuous variables for each task '区' separately
        for col in ["价格"]:
            # standardize the continuous variables for each task '区' separately
            for i in range(self.T + 1):
                # set the mean of col in the each domain as 0
                data_raw.loc[data_raw["区"] == district_list[i], col] = data_raw.loc[
                    data_raw["区"] == district_list[i], col
                ] - np.mean(
                    data_raw.loc[data_raw["区"] == district_list[i], col]
                )  # / np.std(data_raw.loc[data_raw['区'] == district_list[i], col])

        # set idx_target as the domain '通州', '虹口a', '龙岗';
        # stl 的表现比 mtl 更好: '通州', '海淀'
        # 合适的: '昌平', '普陀a', 奉贤a, '盐田'
        idx_target = district_list.index("盐田")
        print(
            f"target domain: {district_list[idx_target]} with sample size: {sample_size[idx_target]}"
        )
        # target data
        if self.flag_test:
            n_target_test = int(sample_size[idx_target] * 0.33)
            n_target_train = int(np.min([sample_size[idx_target] * 0.33, 150]))
        else:
            n_target_test = 0
            n_target_train = int(sample_size[idx_target] * 0.8)
        n_target_valid = int(sample_size[idx_target] - n_target_train - n_target_test)
        self.n_train_target = n_target_train
        # source data
        n_source_train = [int(sample_size[i] * 0.8) for i in range(T + 1)]
        n_source_valid = [int(sample_size[i] - n_source_train[i]) for i in range(T + 1)]
        n_source_train.pop(idx_target)
        n_source_valid.pop(idx_target)
        if self.verbose:
            print(f"[RealData]training sample size: {n_source_train}")

        # target data
        data_tmp1 = data_raw.copy()[data_raw["区"] == district_list[idx_target]]
        data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])  # X + Z
        data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)  # Y
        dataloader_train = DataLoader(
            RegressionDataset(
                data_tmp2[:n_target_train, :], data_tmp3[:n_target_train]
            ),
            batch_size=n_target_train,
            shuffle=True,
        )
        dataloader_valid = DataLoader(
            RegressionDataset(
                data_tmp2[n_target_train : (n_target_train + n_target_valid), :],
                data_tmp3[n_target_train : (n_target_train + n_target_valid)],
            ),
            batch_size=n_target_valid,
            shuffle=True,
        )
        if self.flag_test:
            dataloader_test = DataLoader(
                RegressionDataset(
                    data_tmp2[(n_target_train + n_target_valid) :, :],
                    data_tmp3[(n_target_train + n_target_valid) :],
                ),
                batch_size=n_target_test,
                shuffle=True,
            )
        else:
            dataloader_test = None
        self.dataloader_target_train = dataloader_train
        self.dataloader_target_valid = dataloader_valid
        self.dataloader_target_test = dataloader_test
        self.data_target = {
            # target
            "feature_train": data_tmp2[:n_target_train, :],
            "feature_train_spline": [],
            "label_train": data_tmp3[:n_target_train],
            # valid
            "feature_valid": data_tmp2[
                n_target_train : (n_target_train + n_target_test), :
            ],
            "feature_valid_spline": [],
            "label_valid": data_tmp3[n_target_train : (n_target_train + n_target_test)],
            # test
            "feature_test": data_tmp2[(n_target_train + n_target_test) :, :],
            "feature_test_spline": [],
            "label_test": data_tmp3[(n_target_train + n_target_test) :],
        }

        self.dataloaders_source_train = []
        self.dataloaders_source_valid = []
        self.data_source = {
            # train
            "feature_train": [],
            "feature_train_spline": [],
            "label_train": [],
            # valid
            "feature_valid": [],
            "feature_valid_spline": [],
            "label_valid": [],
        }
        idx_source = np.setdiff1d(np.arange(self.T + 1), [idx_target])
        district_list_source = district_list.copy()
        district_list_source.pop(idx_target)
        for i in range(len(idx_source)):
            data_tmp1 = data_raw.copy()[data_raw["区"] == district_list_source[i]]
            data_tmp2 = np.array(data_tmp1.loc[:, x_cols + z_cols])  # X + Z
            data_tmp3 = np.array(data_tmp1.loc[:, y_col]).reshape(-1, 1)  # Y
            self.data_source["feature_train"].append(data_tmp2[: n_source_train[i], :])
            self.data_source["label_train"].append(data_tmp3[: n_source_train[i], :])
            self.data_source["feature_valid"].append(data_tmp2[n_source_train[i] :, :])
            self.data_source["label_valid"].append(data_tmp3[n_source_train[i] :, :])
            dataloader_train = DataLoader(
                RegressionDataset(
                    data_tmp2[: n_source_train[i], :], data_tmp3[: n_source_train[i]]
                ),
                batch_size=n_source_train[i],
                shuffle=True,
            )
            dataloader_valid = DataLoader(
                RegressionDataset(
                    data_tmp2[n_source_train[i] :, :], data_tmp3[n_source_train[i] :]
                ),
                batch_size=n_source_valid[i],
                shuffle=True,
            )
            self.dataloaders_source_train.append(dataloader_train)
            self.dataloaders_source_valid.append(dataloader_valid)

        self.clean_data()

    def generate_spline(self, n_knots=11, data=None):
        # initialize the spline method
        min_val, max_val = -1.0, 1.0
        # n_knots = 11
        degree = 3
        inner_knots = np.linspace(min_val, max_val, n_knots - 2)
        common_knots = np.concatenate(
            ([inner_knots[0]] * degree, inner_knots, [inner_knots[-1]] * degree)
        )

        if data is None:
            z = self.data_target["feature_train"][:, self.dim_linear :]
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_train_spline = np.concatenate(
                (self.data_target["feature_train"][:, : self.dim_linear], z_spline),
                axis=1,
            )

            z = self.data_target["feature_valid"][:, self.dim_linear :]
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_valid_spline = np.concatenate(
                (self.data_target["feature_valid"][:, : self.dim_linear], z_spline),
                axis=1,
            )

            z = self.data_target["feature_test"][:, self.dim_linear :]
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_test_spline = np.concatenate(
                (self.data_target["feature_test"][:, : self.dim_linear], z_spline),
                axis=1,
            )
            return (
                all_feature_train_spline,
                all_feature_valid_spline,
                all_feature_test_spline,
            )
        else:
            z = data[:, self.dim_linear :]
            z_spline = transform_multidimensional_X(z, common_knots, degree)
            all_feature_spline = np.concatenate(
                (data[:, : self.dim_linear], z_spline), axis=1
            )
            return all_feature_spline

    def tune_knots(self):
        # initialize the spline method
        n_knots_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        loss_val_list = []
        for n_knots in n_knots_list:
            feature_train_spline, feature_valid_spline, freature_test_spline = (
                self.generate_spline(n_knots=n_knots)
            )
            # use ridge spline method to fit the generated data and check the validation loss
            Y = self.data_target["label_train"]

            lm = LinearRegression(
                fit_intercept=False
            )  # RidgeCV(cv=5, fit_intercept=False)
            lm.fit(feature_train_spline, Y)
            tmp_valid_loss = np.mean(
                (lm.predict(feature_valid_spline) - self.data_target["label_valid"])
                ** 2
            )
            loss_val_list.append(tmp_valid_loss)

        def find_min_index(lst):
            min_val = min(lst)
            min_index = lst.index(min_val)
            return min_index

        # chooose the best n_knots according to the validation loss
        best_n_knots = n_knots_list[find_min_index(loss_val_list)]

        # generate the spline feature for target data
        (
            self.data_target["feature_train_spline"],
            self.data_target["feature_valid_spline"],
            self.data_target["feature_test_spline"],
        ) = self.generate_spline(n_knots=best_n_knots)

        # generate the spline feature for source data
        for i in range(self.T):
            tmp = self.generate_spline(
                n_knots=best_n_knots, data=self.data_source["feature_train"][i]
            )
            self.data_source["feature_train_spline"].append(tmp)

            tmp = self.generate_spline(
                n_knots=best_n_knots, data=self.data_source["feature_valid"][i]
            )
            self.data_source["feature_valid_spline"].append(tmp)

    def clean_data(self):
        # count the sample size in each domain, if the sample size is less than 200, we remove it
        tmp_data_source = {
            "feature_train": [],
            "label_train": [],
            "feature_valid": [],
            "label_valid": [],
            "feature_train_spline": [],
            "feature_valid_spline": [],
        }
        tmp_dataloader_train = []
        tmp_dataloader_valid = []
        for i, data_train in enumerate(self.data_source["feature_train"]):
            if data_train.shape[0] > 200:
                tmp_data_source["feature_train"].append(
                    self.data_source["feature_train"][i]
                )
                tmp_data_source["label_train"].append(
                    self.data_source["label_train"][i]
                )
                tmp_data_source["feature_valid"].append(
                    self.data_source["feature_valid"][i]
                )
                tmp_data_source["label_valid"].append(
                    self.data_source["label_valid"][i]
                )
                tmp_dataloader_train.append(
                    DataLoader(
                        RegressionDataset(
                            self.data_source["feature_train"][i],
                            self.data_source["label_train"][i],
                        ),
                        batch_size=self.data_source["feature_train"][i].shape[0],
                        shuffle=True,
                    )
                )
                tmp_dataloader_valid.append(
                    DataLoader(
                        RegressionDataset(
                            self.data_source["feature_valid"][i],
                            self.data_source["label_valid"][i],
                        ),
                        batch_size=self.data_source["feature_valid"][i].shape[0],
                        shuffle=True,
                    )
                )

        self.dataloaders_source_train = tmp_dataloader_train
        self.dataloaders_source_valid = tmp_dataloader_valid
        self.data_source = tmp_data_source
        self.T = len(self.data_source["feature_train"])
        # print the sample size of each domain
        print(
            f"sample size of each domain: {[self.data_source['feature_train'][i].shape[0] for i in range(self.T)]}"
        )


# %%
