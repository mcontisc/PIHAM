import os
import math
import numpy as np
import torch
from torch import Tensor
import warnings
from abc import ABCMeta
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any
from src import tools

# Default parameters
DEFAULT_K = 2  # number of communities
DEFAULT_N = 100  # number of nodes
DEFAULT_L = 1  # number of layers
DEFAULT_Z_categorical = 4  # number of categories for the Categorical attribute
DEFAULT_P_poisson = 1  # number of Poisson attributes
DEFAULT_P_gaussian = 1  # number of Gaussian attributes

DEFAULT_UV_MEAN = 3.0  # mean of the normal distributions for the out-going and in-coming membership U and V
DEFAULT_UV_BIAS = 1.0  # bias for the mean of the normal distributions for the out-going and in-coming membership U
# and V
DEFAULT_U_STD = (
    0.2  # standard deviation of the normal distributions for the out-going membership U
)
DEFAULT_V_STD = (
    0.3  # standard deviation of the normal distributions for the in-coming membership V
)
DEFAULT_W_OFF = (
    -4.0
)  # mean of the normal distributions for the off-diagonal entries of the affinity tensor W
DEFAULT_W_BIAS = 4.0  # bias for the mean of the normal distributions for the diagonal entries of the affinity tensor W
DEFAULT_W_STD = (
    0.45  # standard deviation of the normal distributions for the affinity tensor W
)
DEFAULT_Hcategorical_STD = 0.2  # standard deviation of the normal distributions for the community-covariate matrix
# Hcategorical, related to the categorical attribute.
DEFAULT_Hpoisson_STD = 0.1  # standard deviation of the normal distributions for the community-covariate matrix
# Hpoisson, related to the Poisson attributes.
DEFAULT_Hgaussian_STD = 0.2  # standard deviation of the normal distributions for the community-covariate matrix
# Hgaussian, related to the Gaussian attributes.

DEFAULT_SEED = 10  # seed for the random number generator
DEFAULT_OUT_FOLDER = "data/input/"  # output folder
DEFAULT_SHOW_PLOTS = True  # flag to plot generated data
DEFAULT_LABEL = ""  # label for the output files


class BaseSyntheticNetwork(metaclass=ABCMeta):
    """A base abstract class for generation and management of synthetic networks.
    Suitable for representing any type of synthetic network.
    """

    def __init__(
        self,
        K: int = DEFAULT_K,
        N: int = DEFAULT_N,
        L: int = DEFAULT_L,
        Z_categorical: int = DEFAULT_Z_categorical,
        P_gaussian: int = DEFAULT_P_gaussian,
        P_poisson: int = DEFAULT_P_poisson,
        seed: int = DEFAULT_SEED,
        out_folder: str = DEFAULT_OUT_FOLDER,
        shoW_plots: bool = DEFAULT_SHOW_PLOTS,
        **kwargs,
    ) -> None:
        self.K = K  # number of communities
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.Z_categorical = (
            Z_categorical  # number of category for the Categorical attribute
        )
        self.P_poisson = P_poisson  # number of Poisson attributes
        self.P_gaussian = P_gaussian  # number of Gaussian attributes

        # Set seed random number generator
        self.seed = seed
        torch.manual_seed(self.seed)

        self.out_folder = out_folder
        self.shoW_plots = shoW_plots


class StandardPIHAM(BaseSyntheticNetwork):
    """Generation of heterogeneous and attributed multilayer networks, following
    the PIHAM probabilistic generative model from

    [1] Flexible inference in heterogeneous and attributed multilayer networks,
        Contisciani M., Hobbhahn M., Power E.A., Hennig P., and De Bacco C. (2024)

    This model assumes the existence of a mixed-membership community structure that drives the generation
    of both interactions and node attributes. All latent variables are drawn from normal distributions.

    In this implementation, we generate networks with L = 3 heterogeneous layers (one with binary interactions,
    the second with nonnegative discrete weights, and the third with real values) and three covariates (one
    categorical with Z = 4 categories, one with nonnegative discrete values, and the last with real values).
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        if "parameters_network" in kwargs:
            parameters_network = kwargs["parameters_network"]
        else:
            parameters_network = None
        if "parameters_covariates" in kwargs:
            parameters_covariates = kwargs["parameters_covariates"]
        else:
            parameters_covariates = None

        # Initialize parameters
        self.initialize(**kwargs)
        # Generate adjacency tensor
        self.build_A(parameters_network=parameters_network)
        # Generate design matrix
        self.build_X(parameters_covariates=parameters_covariates)
        # Save generated data
        self.save_data()
        # Save ground-truth parameters
        self.save_parameters()
        # Plot generated data
        if self.shoW_plots:
            self.plot_A()
            self.plot_X()

    def initialize(self, **kwargs) -> None:
        """Set parameters mu and sigma of the normal distributions for the parameters."""

        super().__init__(**kwargs)

        # Label output files
        if "label" in kwargs:
            label = kwargs["label"]
        else:
            msg = f"label parameter was not set. Defaulting to label={DEFAULT_LABEL}"
            warnings.warn(msg)
            label = DEFAULT_LABEL
        self.label = label

        # Parameters for the latent variables
        if "UV_mean" in kwargs:
            UV_mean = kwargs["UV_mean"]
        else:
            msg = f"UV_mean parameter was not set. Defaulting to UV_mean={DEFAULT_UV_MEAN}"
            warnings.warn(msg)
            UV_mean = DEFAULT_UV_MEAN
        self.UV_mean = UV_mean
        if "UV_bias" in kwargs:
            UV_bias = kwargs["UV_bias"]
        else:
            msg = f"UV_bias parameter was not set. Defaulting to UV_bias={DEFAULT_UV_BIAS}"
            warnings.warn(msg)
            UV_bias = DEFAULT_UV_BIAS
        self.UV_bias = UV_bias
        if "U_std" in kwargs:
            U_std = kwargs["U_std"]
        else:
            msg = f"U_std parameter was not set. Defaulting to U_std={DEFAULT_U_STD}"
            warnings.warn(msg)
            U_std = DEFAULT_U_STD
        self.U_std = U_std
        if "V_std" in kwargs:
            V_std = kwargs["V_std"]
        else:
            msg = f"V_std parameter was not set. Defaulting to V_std={DEFAULT_V_STD}"
            warnings.warn(msg)
            V_std = DEFAULT_V_STD
        self.V_std = V_std
        if "W_off" in kwargs:
            W_off = kwargs["W_off"]
        else:
            msg = f"W_off parameter was not set. Defaulting to W_off={DEFAULT_W_OFF}"
            warnings.warn(msg)
            W_off = DEFAULT_W_OFF
        self.W_off = W_off
        if "W_bias" in kwargs:
            W_bias = kwargs["W_bias"]
        else:
            msg = f"W_bias parameter was not set. Defaulting to W_bias={DEFAULT_W_BIAS}"
            warnings.warn(msg)
            W_bias = DEFAULT_W_BIAS
        self.W_bias = W_bias
        if "W_std" in kwargs:
            W_std = kwargs["W_std"]
        else:
            msg = f"W_std parameter was not set. Defaulting to W_std={DEFAULT_W_STD}"
            warnings.warn(msg)
            W_std = DEFAULT_W_STD
        self.W_std = W_std
        if "Hcategorical_std" in kwargs:
            Hcategorical_std = kwargs["Hcategorical_std"]
        else:
            msg = f"Hcategorical_std parameter was not set. Defaulting to Hcategorical_std={DEFAULT_Hcategorical_STD}"
            warnings.warn(msg)
            Hcategorical_std = DEFAULT_Hcategorical_STD
        self.Hcategorical_std = Hcategorical_std
        if "Hpoisson_std" in kwargs:
            Hpoisson_std = kwargs["Hpoisson_std"]
        else:
            msg = f"Hpoisson_std parameter was not set. Defaulting to Hpoisson_std={DEFAULT_Hpoisson_STD}"
            warnings.warn(msg)
            Hpoisson_std = DEFAULT_Hpoisson_STD
        self.Hpoisson_std = Hpoisson_std
        if "Hgaussian_std" in kwargs:
            Hgaussian_std = kwargs["Hgaussian_std"]
        else:
            msg = f"Hgaussian_std parameter was not set. Defaulting to Hgaussian_std={DEFAULT_Hgaussian_STD}"
            warnings.warn(msg)
            Hgaussian_std = DEFAULT_Hgaussian_STD
        self.Hgaussian_std = Hgaussian_std

    def build_A(self, parameters_network: Optional[List[Tensor]] = None) -> None:
        """Generate the adjacency tensor using the latent variables."""

        """
        Set latent variables
        """
        if parameters_network is None:
            # Generate latent variables associated to the network
            self.U, self.V, self.W = self.generate_lv_mmsbm()
        else:
            # Set latent variables associated to the network from input
            self.U, self.V, self.W = parameters_network
            if self.U.size != (self.N, self.K):
                raise ValueError("The shape of the parameter U has to be (N, K).")
            if self.V.size != (self.N, self.K):
                raise ValueError("The shape of the parameter V has to be (N, K).")
            if self.W.size != (self.L, self.K, self.K):
                raise ValueError("The shape of the parameter W has to be (L, K, K).")

        """
        Generate A
        """
        # Bernoulli layer
        Lambda_bernoulli = tools.forward_bernoulli_layer(
            self.U, self.V, self.W[0].reshape(1, self.K, self.K)
        )
        A_bernoulli = torch.distributions.bernoulli.Bernoulli(Lambda_bernoulli).sample()
        # Poisson layer
        Lambda_poisson = tools.forward_poisson_layer(
            self.U, self.V, self.W[1].reshape(1, self.K, self.K)
        )
        A_poisson = torch.distributions.poisson.Poisson(Lambda_poisson).sample()
        # Gaussian layer
        Lambda_gaussian = tools.forward_gaussian_layer(
            self.U, self.V, self.W[2].reshape(1, self.K, self.K)
        )
        A_gaussian = torch.distributions.normal.Normal(Lambda_gaussian, 1).sample()
        # Stack the layer together
        self.A = torch.stack([A_bernoulli, A_poisson, A_gaussian]).reshape(
            self.L, self.N, self.N
        )

    def build_X(self, parameters_covariates: Optional[List[Tensor]] = None) -> None:
        """Generate the design matrix using the latent variables."""

        """
        Set latent variables
        """
        if parameters_covariates is None:
            # Generate latent variables associated to the covariates
            (
                self.Hcategorical,
                self.Hpoisson,
                self.Hgaussian,
            ) = self.generate_lv_communitycovariate()
        else:
            # Set latent variables associated to the covariates from input
            self.Hcategorical, self.Hgaussian, self.Hpoisson = parameters_covariates
            if self.Hcategorical.size != (self.K, self.Z_categorical):
                raise ValueError(
                    "The shape of the parameter Hcategorical has to be (K, Z_categorical)."
                )
            if self.Hpoisson.size != (self.K, self.P_poisson):
                raise ValueError(
                    "The shape of the parameter Hpoisson has to be (K, P_poisson)."
                )
            if self.Hgaussian.size != (self.K, self.P_gaussian):
                raise ValueError(
                    "The shape of the parameter Hgaussian has to be (K, P_gaussian)."
                )

        """
        Generate X
        """
        # Categorical attribute
        pi_categorical = tools.forward_categorical_covariate(
            self.U, self.V, self.Hcategorical
        )
        self.X_categorical = torch.distributions.multinomial.Multinomial(
            total_count=1, probs=pi_categorical
        ).sample()
        # Poisson attributes
        pi_poisson = tools.forward_poisson_covariate(self.U, self.V, self.Hpoisson)
        self.X_poisson = torch.distributions.poisson.Poisson(pi_poisson).sample()
        # Gaussian attributes
        pi_gaussian = tools.forward_gaussian_covariate(self.U, self.V, self.Hgaussian)
        self.X_gaussian = torch.distributions.normal.Normal(pi_gaussian, 1).sample()

    def sample_membership_matrices(self) -> Tuple[Tensor, Tensor]:
        """Generate the NxK out-going (U) and in-coming (V) membership matrices.

        To generate the membership matrices U and V , we assign equal-size group memberships
        and draw the entries of the matrices from distributions with different means,
        according to the group the nodes belong to.
        """

        size = int(self.N / self.K)
        # Define mean for the normal distributions
        mean_matrix = torch.zeros((self.N, self.K)).float()
        for i in range(self.N):
            q = int(math.floor(float(i) / float(size)))
            if q == self.K:
                mean_matrix[i:, self.K - 1] = self.UV_mean
            else:
                mean_matrix[i, q] = self.UV_mean
        mean_matrix = mean_matrix.view(-1)
        self.U_mean = mean_matrix - self.UV_bias
        self.V_mean = mean_matrix - self.UV_bias
        # Draw variables from normal distributions
        U = (
            torch.distributions.normal.Normal(self.U_mean, self.U_std)
            .sample()
            .reshape(self.N, self.K)
        )
        V = (
            torch.distributions.normal.Normal(self.V_mean, self.V_std)
            .sample()
            .reshape(self.N, self.K)
        )

        return U, V

    def sample_affinity_tensor(self) -> Tensor:
        """Generate the LxKxK affinity tensor (W)."""

        # Define mean for the normal distributions
        self.W_mean = torch.zeros((self.L, self.K, self.K)).float() + self.W_off
        self.W_mean = self.W_mean + self.W_bias * torch.eye(self.K).float()
        self.W_mean = self.W_mean.reshape(-1)
        # Draw variables from normal distributions
        W = (
            torch.distributions.normal.Normal(self.W_mean, self.W_std)
            .sample()
            .reshape(self.L, self.K, self.K)
        )

        return W

    def generate_lv_mmsbm(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate latent variables representing community memberships and affinity tensor,
        assuming network layers are independent and communities are shared across layers.
        """

        # Generate U, V
        U, V = self.sample_membership_matrices()
        # Generate W
        W = self.sample_affinity_tensor()

        return U, V, W

    def generate_lv_communitycovariate(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate the community-covariate matrix H, explaining how an attribute
        x is distributed among the K communities.
        """

        # Generate Hcategorical - related to the Categorical attribute
        self.Hcategorical_mean = torch.zeros((self.K, self.Z_categorical))
        for k in range(min(self.K, self.Z_categorical)):
            self.Hcategorical_mean[k, k] = 0.5 + (k + 1)
        if self.Z_categorical > self.K:
            for z in range(self.K, self.Z_categorical):
                self.Hcategorical_mean[:, z] = 0.2
        if self.Z_categorical < self.K:
            for k in range(self.Z_categorical, self.K):
                self.Hcategorical_mean[k, :] = 0.2
        Hcategorical = (
            torch.distributions.normal.Normal(
                self.Hcategorical_mean, self.Hcategorical_std
            )
            .sample()
            .reshape(self.K, self.Z_categorical)
        )
        # Generate Hpoisson - related to the Poisson attributes
        self.Hpoisson_mean = torch.zeros((self.K, self.P_poisson))
        for k in range(self.K):
            self.Hpoisson_mean[k, 0] = 1.5 * (k / 3 + 1)
        Hpoisson = (
            torch.distributions.normal.Normal(self.Hpoisson_mean, self.Hpoisson_std)
            .sample()
            .reshape(self.K, self.P_poisson)
        )
        # Generate Hgaussian - related to the Gaussian attribute
        self.Hgaussian_mean = torch.zeros((self.K, self.P_gaussian))
        for k in range(self.K):
            self.Hgaussian_mean[k, 0] = 1 + (1 - k) * 3
        Hgaussian = (
            torch.distributions.normal.Normal(self.Hgaussian_mean, self.Hgaussian_std)
            .sample()
            .reshape(self.K, self.P_gaussian)
        )

        return Hcategorical, Hpoisson, Hgaussian

    def save_data(self) -> None:
        """Save generated data in a compressed file."""

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        output_data = self.out_folder + "synthetic_data" + self.label + ".pt"
        data_dict = {
            "A": self.A,
            "X_categorical": self.X_categorical,
            "X_poisson": self.X_poisson,
            "X_gaussian": self.X_gaussian,
        }

        torch.save(data_dict, output_data)
        print(f"Data saved in: {output_data}")
        print('To load: data=torch.load(filename), then e.g. data["A"]')

    def save_parameters(self) -> None:
        """Save ground-truth parameters in a compressed file."""

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        output_parameters = (
            self.out_folder + "theta_gt_synthetic_data" + self.label + ".pt"
        )
        parameter_dict = {
            "U": self.U,
            "V": self.V,
            "W": self.W,
            "Hcategorical": self.Hcategorical,
            "Hpoisson": self.Hpoisson,
            "Hgaussian": self.Hgaussian,
        }

        torch.save(parameter_dict, output_parameters)
        print(f"True parameters saved in: {output_parameters}")
        print('To load: theta_gt=torch.load(filename), then e.g. theta_gt["U"]')

    def plot_A(self, cmap: Any = "PuBuGn") -> None:
        """Plot the generated adjacency tensor."""

        labels = {
            0: "$A_{ij}^{1} \in \{0, 1\}$",
            1: "$A_{ij}^{2} \in \mathbb{N}_0$",
            2: "$A_{ij}^{3} \in \mathbb{R}$",
        }
        for layer in range(self.L):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(self.A[layer], cmap=plt.get_cmap(cmap))
            ax.set_title(rf"{labels[layer]}", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    def plot_X(self, cmap: Any = "PuBuGn") -> None:
        """Plot the generated design matrix."""

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.X_categorical, cmap=plt.get_cmap(cmap), aspect="auto")
        ax.set_title(r"$X_{ix} \in \{0, 1\}$", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.X_poisson, cmap=plt.get_cmap(cmap), aspect="auto")
        ax.set_title(r"$X_{ix} \in \mathbb{N}_0$", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.X_gaussian, cmap=plt.get_cmap(cmap), aspect="auto")
        ax.set_title(r"$X_{ix} \in \mathbb{R}$", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()
