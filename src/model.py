import numpy as np
import torch
from torch import Tensor
from torch import optim
from torch import nn
from torch.autograd.functional import hessian
from typing import List, Tuple
from src import tools


class PIHAM(nn.Module):
    """Implementation of the PIHAM probabilistic generative model from

    [1] Flexible inference in heterogeneous and attributed multilayer networks,
        Contisciani M., Hobbhahn M., Power E.A., Hennig P., and De Bacco C. (2024)

    This model is designed to perform probabilistic inference in directed and undirected
    heterogeneous and attributed multilayer networks. At its core, PIHAM assumes the existence
    of a mixed-membership community structure that drives the generation of both interactions
    and node attributes. In addition, the inference of the parameters is performed within
    a Bayesian framework, where both prior and posterior distributions are
    modeled with Gaussian distributions.

    In this implementation, we analyze networks with L = 3 heterogeneous layers (one with binary interactions,
    the second with nonnegative discrete weights, and the third with real values) and three covariates (one
    categorical with Z = 4 categories, one with nonnegative discrete values, and the last with real values).
    """

    def __init__(
        self,
        U_mu_prior: Tensor,
        U_std_prior: Tensor,
        V_mu_prior: Tensor,
        V_std_prior: Tensor,
        W_mu_prior: Tensor,
        W_std_prior: Tensor,
        Hcategorical_mu_prior: Tensor,
        Hcategorical_std_prior: Tensor,
        Hpoisson_mu_prior: Tensor,
        Hpoisson_std_prior: Tensor,
        Hgaussian_mu_prior: Tensor,
        Hgaussian_std_prior: Tensor,
        K: int,
        N: int,
        L: int,
        Z_categorical: int,
        P_poisson: int,
        P_gaussian: int,
        configuration,
    ) -> None:
        """Initialize the probabilistic generative model.

        Parameters:
        -----------
        U_mu_prior: Mean of the prior distribution for the out-going memberships U_ik.
        U_std_prior: Standard deviation of the prior distribution for the out-going memberships U_ik.
        V_mu_prior: Mean of the prior distribution for the in-coming memberships V_ik.
        V_std_prior: Standard deviation of the prior distribution for the in-coming memberships V_ik.
        W_mu_prior: Mean of the prior distributions for the affinity tensor W.
        W_std_prior: Standard deviation of the prior distributions for the affinity tensor W.
        Hcategorical_mu_prior: Mean of the prior distributions for the
            community-covariate matrix Hcategorical, related to the categorical attribute.
        Hcategorical_std_prior: Standard deviation of the prior distributions for the
            community-covariate matrix Hcategorical, related to the categorical attribute.
        Hpoisson_mu_prior: Mean of the prior distributions for the
            community-covariate matrix Hpoisson, related to the poisson attributes.
        Hpoisson_std_prior: Standard deviation of the prior distributions for
            the community-covariate matrix Hpoisson, related to the Poisson attributes.
        Hgaussian_mu_prior: Mean of the prior distribution for the
            community-covariate matrix Hgaussian, related to the Gaussian attributes.
        Hgaussian_std_prior: Standard deviation of the prior distribution for the
            community-covariate matrix Hgaussian, related to the Gaussian attributes.
        K: Number of communities.
        N: Number of nodes.
        L: Number of layers.
        Z_categorical: Number of categories for the categorical attribute.
        P_poisson: Number of Poisson attributes.
        P_gaussian: Number of Gaussian attributes.
        configuration: Dictionary containing the configuration of the model.
        """

        super().__init__()
        self.U_mu_prior = U_mu_prior
        self.U_std_prior = U_std_prior
        self.V_mu_prior = V_mu_prior
        self.V_std_prior = V_std_prior
        self.W_mu_prior = W_mu_prior
        self.W_std_prior = W_std_prior
        self.Hcategorical_mu_prior = Hcategorical_mu_prior
        self.Hcategorical_std_prior = Hcategorical_std_prior
        self.Hpoisson_mu_prior = Hpoisson_mu_prior
        self.Hpoisson_std_prior = Hpoisson_std_prior
        self.Hgaussian_mu_prior = Hgaussian_mu_prior
        self.Hgaussian_std_prior = Hgaussian_std_prior
        self.K = K
        self.N = N
        self.L = L
        self.Z_categorical = Z_categorical
        self.P_poisson = P_poisson
        self.P_gaussian = P_gaussian
        self.device = configuration["device"]
        self.configuration = configuration

    def initialize(self, seed: int) -> None:
        """Initialize latent variables."""

        torch.manual_seed(seed)
        U_INIT = torch.normal(
            self.configuration["U_init_mu"] * torch.ones((self.N, self.K)),
            self.configuration["U_init_std"] * torch.ones((self.N, self.K)),
        ).to(self.device)
        V_INIT = torch.normal(
            self.configuration["V_init_mu"] * torch.ones((self.N, self.K)),
            self.configuration["V_init_std"] * torch.ones((self.N, self.K)),
        ).to(self.device)
        W_INIT = torch.normal(
            self.configuration["W_init_mu"] * torch.ones((self.L, self.K, self.K)),
            self.configuration["W_init_std"] * torch.ones((self.L, self.K, self.K)),
        ).to(self.device)
        Hcategorical_INIT = torch.normal(
            self.configuration["Hcategorical_init_mu"]
            * torch.ones((self.K, self.Z_categorical)),
            self.configuration["Hcategorical_init_std"]
            * torch.ones((self.K, self.Z_categorical)),
        ).to(self.device)
        Hpoisson_INIT = torch.normal(
            self.configuration["Hpoisson_init_mu"]
            * torch.ones((self.K, self.P_poisson)),
            self.configuration["Hpoisson_init_std"]
            * torch.ones((self.K, self.P_poisson)),
        ).to(self.device)
        Hgaussian_INIT = torch.normal(
            self.configuration["Hgaussian_init_mu"]
            * torch.ones((self.K, self.P_gaussian)),
            self.configuration["Hgaussian_init_std"]
            * torch.ones((self.K, self.P_gaussian)),
        ).to(self.device)

        self.U = nn.Parameter(U_INIT.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.V = nn.Parameter(V_INIT.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.W = nn.Parameter(W_INIT.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.Hcategorical = nn.Parameter(
            Hcategorical_INIT.clone().detach(), requires_grad=True
        ).to(self.device)
        self.Hpoisson = nn.Parameter(
            Hpoisson_INIT.clone().detach(), requires_grad=True
        ).to(self.device)
        self.Hgaussian = nn.Parameter(
            Hgaussian_INIT.clone().detach(), requires_grad=True
        ).to(self.device)

    def get_UVWH(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get latent variables."""

        return (
            self.U,
            self.V,
            self.W,
            self.Hcategorical,
            self.Hpoisson,
            self.Hgaussian,
        )

    def forward(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass to compute the expected values of the likelihoods."""

        W_bernoulli = self.W[0].reshape(1, self.K, self.K)
        W_poisson = self.W[1].reshape(1, self.K, self.K)
        W_gaussian = self.W[2].reshape(1, self.K, self.K)
        Lambda_bernoulli = tools.forward_bernoulli_layer(self.U, self.V, W_bernoulli)
        Lambda_poisson = tools.forward_poisson_layer(self.U, self.V, W_poisson)
        Lambda_gaussian = tools.forward_gaussian_layer(self.U, self.V, W_gaussian)
        pi_categorical = tools.forward_categorical_covariate(
            self.U, self.V, self.Hcategorical
        )
        pi_poisson = tools.forward_poisson_covariate(self.U, self.V, self.Hpoisson)
        pi_gaussian = tools.forward_gaussian_covariate(self.U, self.V, self.Hgaussian)

        return (
            Lambda_bernoulli,
            Lambda_poisson,
            Lambda_gaussian,
            pi_categorical,
            pi_poisson,
            pi_gaussian,
        )

    def get_neg_log_posterior(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> float:
        """Compute the negative log posterior."""

        # Likelihood
        (
            Lambda_bernoulli,
            Lambda_poisson,
            Lambda_gaussian,
            pi_categorical,
            pi_poisson,
            pi_gaussian,
        ) = self.forward()
        A_bernoulli = A[0]
        A_poisson = A[1]
        A_gaussian = A[2]
        layer_bernoulli = torch.distributions.bernoulli.Bernoulli(
            Lambda_bernoulli
        ).log_prob(A_bernoulli)
        layer_poisson = torch.distributions.poisson.Poisson(Lambda_poisson).log_prob(
            A_poisson
        )
        layer_gaussian = torch.distributions.normal.Normal(Lambda_gaussian, 1).log_prob(
            A_gaussian
        )
        covariate_categorical = torch.distributions.multinomial.Multinomial(
            total_count=1, probs=pi_categorical
        ).log_prob(X_categorical)
        covariate_poisson = torch.distributions.poisson.Poisson(pi_poisson).log_prob(
            X_poisson
        )
        covariate_gaussian = torch.distributions.normal.Normal(pi_gaussian, 1).log_prob(
            X_gaussian
        )
        # Priors
        norm_U = torch.distributions.normal.Normal(
            self.U_mu_prior, self.U_std_prior
        ).log_prob(self.U)
        norm_V = torch.distributions.normal.Normal(
            self.V_mu_prior, self.V_std_prior
        ).log_prob(self.V)
        norm_W = torch.distributions.normal.Normal(
            self.W_mu_prior, self.W_std_prior
        ).log_prob(self.W)
        norm_Hcategorical = torch.distributions.normal.Normal(
            self.Hcategorical_mu_prior, self.Hcategorical_std_prior
        ).log_prob(self.Hcategorical)
        norm_Hpoisson = torch.distributions.normal.Normal(
            self.Hpoisson_mu_prior, self.Hpoisson_std_prior
        ).log_prob(self.Hpoisson)
        norm_Hgaussian = torch.distributions.normal.Normal(
            self.Hgaussian_mu_prior, self.Hgaussian_std_prior
        ).log_prob(self.Hgaussian)

        return (
            -likelihood_weight
            * (
                layer_bernoulli.sum()
                + layer_poisson.sum()
                + layer_gaussian.sum()
                + covariate_categorical.sum()
                + covariate_poisson.sum()
                + covariate_gaussian.sum()
            )
            - norm_U.sum()
            - norm_V.sum()
            - norm_W.sum()
            - norm_Hcategorical.sum()
            - norm_Hpoisson.sum()
            - norm_Hgaussian.sum()
        )

    def fit(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        gamma: float = 0.1,
        tolerance: float = 1e-8,
        num_iter: int = 2000,
        likelihood_weight: float = 1.0,
        learning_rate: float = 1e-2,
        verbose: bool = True,
        N_seeds: int = 10,
        print_likelihoods=True,
    ) -> List:
        """Fit the model using Adam optimization.

        Parameters:
        -----------
        A: Adjacency tensor.
        X_categorical: Design matrix for categorical attribute.
        X_poisson: Design matrix for Poisson attributes.
        X_gaussian: Design matrix for Gaussian attributes.
        gamma: Multiplicative factor of learning rate decay
        tolerance: Tolerance for the stopping criterion.
        num_iter: Maximum number of iterations.
        likelihood_weight: Weight of the likelihood term.
        learning_rate: Learning rate.
        verbose: Flag to print details.
        N_seeds: Number of realization, each with a different random initialization.
        print_likelihoods: Flag to print the likelihood for each seed.
        """

        # Initialize optimal parameters
        best_likelihood = -float("inf")
        best_likelihoods = []
        self.initialize(0)
        best_U, best_V, best_W, best_Hcategorical, best_Hpoisson, best_Hgaussian = (
            self.U.clone(),
            self.V.clone(),
            self.W.clone(),
            self.Hcategorical.clone(),
            self.Hpoisson.clone(),
            self.Hgaussian.clone(),
        )

        # Set random seed
        np.random.seed(N_seeds)

        # Execute algorithm for different realizations
        for s in np.random.choice(np.arange(100), size=N_seeds, replace=False):
            # Initialize parameters
            self.initialize(s)

            # Optimize
            likelihoods = []
            optimizer = optim.Adam(
                [
                    self.U,
                    self.V,
                    self.W,
                    self.Hcategorical,
                    self.Hpoisson,
                    self.Hgaussian,
                ],
                lr=learning_rate,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[num_iter // 2, (num_iter // 4) * 3], gamma=gamma
            )
            for i in range(num_iter):
                optimizer.zero_grad()
                loss = self.get_neg_log_posterior(
                    A, X_categorical, X_poisson, X_gaussian, likelihood_weight
                ).double()
                if i % 50 == 0 and verbose:
                    print("Iteration: {}; log-likelihood: {}".format(i, -loss.item()))
                if i > 200:
                    if (
                        torch.abs(torch.tensor(-loss.item() - likelihoods[-50]))
                        < tolerance
                    ):
                        print("likelihood: ", -loss.item())
                        print(
                            "loss didn't change more than {} over the last 50 steps; breaking".format(
                                tolerance
                            )
                        )
                        break
                likelihoods.append(-loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
            if print_likelihoods:
                print("likelihood for seed {}: {}".format(s, likelihoods[-1]))
            if likelihoods[-1] > best_likelihood:
                best_likelihood = likelihoods[-1]
                best_likelihoods = likelihoods
                (
                    best_U,
                    best_V,
                    best_W,
                    best_Hcategorical,
                    best_Hpoisson,
                    best_Hgaussian,
                ) = (
                    self.U.clone(),
                    self.V.clone(),
                    self.W.clone(),
                    self.Hcategorical.clone(),
                    self.Hpoisson.clone(),
                    self.Hgaussian.clone(),
                )

        # Save results
        self.U = nn.Parameter(best_U.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.V = nn.Parameter(best_V.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.W = nn.Parameter(best_W.clone().detach(), requires_grad=True).to(
            self.device
        )
        self.Hcategorical = nn.Parameter(
            best_Hcategorical.clone().detach(), requires_grad=True
        ).to(self.device)
        self.Hpoisson = nn.Parameter(
            best_Hpoisson.clone().detach(), requires_grad=True
        ).to(self.device)
        self.Hgaussian = nn.Parameter(
            best_Hgaussian.clone().detach(), requires_grad=True
        ).to(self.device)

        return best_likelihoods

    def save_results(self, folder_name: str, file_name: str) -> None:
        """Save the inferred parameters in a compressed file."""

        outfile = folder_name + "theta" + file_name
        np.savez_compressed(
            outfile + ".npz",
            U=self.U.detach().cpu().numpy(),
            V=self.V.detach().cpu().numpy(),
            W=self.W.detach().cpu().numpy(),
            Hcategorical=self.Hcategorical.detach().cpu().numpy(),
            Hpoisson=self.Hpoisson.detach().cpu().numpy(),
            Hgaussian=self.Hgaussian.detach().cpu().numpy(),
            Cov=self.get_Covariance(diagonal_only=False),
            Cov_diag=self.get_Covariance(diagonal_only=True),
        )
        print(f'Inferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["U"]')

    def log_posterior(
        self,
        U: Tensor,
        V: Tensor,
        W: Tensor,
        Hcategorical: Tensor,
        Hpoisson: Tensor,
        Hgaussian: Tensor,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        Lambda_bernoulli: Tensor,
        Lambda_poisson: Tensor,
        Lambda_gaussian: Tensor,
        pi_categorical: Tensor,
        pi_poisson: Tensor,
        pi_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> float:
        """Compute the log posterior distribution."""

        A_bernoulli = A[0]
        A_poisson = A[1]
        A_gaussian = A[2]
        layer_bernoulli = torch.distributions.bernoulli.Bernoulli(
            Lambda_bernoulli
        ).log_prob(A_bernoulli)
        layer_poisson = torch.distributions.poisson.Poisson(Lambda_poisson).log_prob(
            A_poisson
        )
        layer_gaussian = torch.distributions.normal.Normal(Lambda_gaussian, 1).log_prob(
            A_gaussian
        )
        covariate_categorical = torch.distributions.multinomial.Multinomial(
            total_count=1, probs=pi_categorical
        ).log_prob(X_categorical)
        covariate_poisson = torch.distributions.poisson.Poisson(pi_poisson).log_prob(
            X_poisson
        )
        covariate_gaussian = torch.distributions.normal.Normal(pi_gaussian, 1).log_prob(
            X_gaussian
        )
        norm_U = torch.distributions.normal.Normal(
            self.U_mu_prior, self.U_std_prior
        ).log_prob(U)
        norm_V = torch.distributions.normal.Normal(
            self.V_mu_prior, self.V_std_prior
        ).log_prob(V)
        norm_W = torch.distributions.normal.Normal(
            self.W_mu_prior, self.W_std_prior
        ).log_prob(W)
        norm_Hcategorical = torch.distributions.normal.Normal(
            self.Hcategorical_mu_prior, self.Hcategorical_std_prior
        ).log_prob(Hcategorical)
        norm_Hpoisson = torch.distributions.normal.Normal(
            self.Hpoisson_mu_prior, self.Hpoisson_std_prior
        ).log_prob(Hpoisson)
        norm_Hgaussian = torch.distributions.normal.Normal(
            self.Hgaussian_mu_prior, self.Hgaussian_std_prior
        ).log_prob(Hgaussian)

        return (
            likelihood_weight
            * (
                layer_bernoulli.sum()
                + layer_poisson.sum()
                + layer_gaussian.sum()
                + covariate_categorical.sum()
                + covariate_poisson.sum()
                + covariate_gaussian.sum()
            )
            + norm_U.sum()
            + norm_V.sum()
            + norm_W.sum()
            + norm_Hcategorical.sum()
            + norm_Hpoisson.sum()
            + norm_Hgaussian.sum()
        )

    def log_post_for_Hessian(
        self,
        Theta: Tensor,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> float:
        """Compute log posterior distribution for the Hessian."""

        # Restore original parameters
        U = Theta[0 : self.N * self.K].reshape(self.N, self.K)
        V = Theta[self.N * self.K : 2 * self.N * self.K].reshape(self.N, self.K)
        W = Theta[
            2 * self.N * self.K : 2 * self.N * self.K + self.L * self.K * self.K
        ].reshape(self.L, self.K, self.K)
        Hcategorical = Theta[
            2 * self.N * self.K
            + self.L * self.K * self.K : 2 * self.N * self.K
            + self.L * self.K * self.K
            + self.K * self.Z_categorical
        ].reshape(self.K, self.Z_categorical)
        Hpoisson = Theta[
            2 * self.N * self.K
            + self.L * self.K * self.K
            + self.K * self.Z_categorical : 2 * self.N * self.K
            + self.L * self.K * self.K
            + self.K * (self.Z_categorical + self.P_poisson)
        ].reshape(self.K, self.P_poisson)
        Hgaussian = Theta[
            2 * self.N * self.K
            + self.L * self.K * self.K
            + self.K * (self.Z_categorical + self.P_poisson) : 2 * self.N * self.K
            + self.L * self.K * self.K
            + self.K * (self.Z_categorical + self.P_poisson + self.P_gaussian)
        ].reshape(self.K, self.P_gaussian)

        # Forward pass
        W_bernoulli = W[0].reshape(1, self.K, self.K)
        W_poisson = W[1].reshape(1, self.K, self.K)
        W_gaussian = W[2].reshape(1, self.K, self.K)
        Lambda_bernoulli = tools.forward_bernoulli_layer(U, V, W_bernoulli)
        Lambda_poisson = tools.forward_poisson_layer(U, V, W_poisson)
        Lambda_gaussian = tools.forward_gaussian_layer(U, V, W_gaussian)
        pi_categorical = tools.forward_categorical_covariate(U, V, Hcategorical)
        pi_poisson = tools.forward_poisson_covariate(U, V, Hpoisson)
        pi_gaussian = tools.forward_gaussian_covariate(U, V, Hgaussian)

        # Compute log posterior
        log_posterior = self.log_posterior(
            U,
            V,
            W,
            Hcategorical,
            Hpoisson,
            Hgaussian,
            A,
            X_categorical,
            X_poisson,
            X_gaussian,
            Lambda_bernoulli,
            Lambda_poisson,
            Lambda_gaussian,
            pi_categorical,
            pi_poisson,
            pi_gaussian,
            likelihood_weight,
        )

        return log_posterior

    def compute_Hessian(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> None:
        """Compute the Hessian around the MAP estimates."""

        Theta = torch.cat(
            [
                self.U.ravel(),
                self.V.ravel(),
                self.W.ravel(),
                self.Hcategorical.ravel(),
                self.Hpoisson.ravel(),
                self.Hgaussian.ravel(),
            ]
        )
        H = hessian(
            lambda x: self.log_post_for_Hessian(
                x, A, X_categorical, X_poisson, X_gaussian, likelihood_weight
            ),
            Theta,
        )
        self.Hessian = H

    def is_neg_Hessian_pos_def(self, eps: float = 0.0) -> None:
        """Check if the negative Hessian is positive definite."""

        eigv = torch.linalg.eigvals(
            -self.Hessian - eps * torch.eye(self.Hessian.size(0)).to(self.device)
        )
        all_pos = (eigv.real >= 0).all()
        if all_pos:
            print("All eigenvalues positive")
        else:
            print("Not all eigenvalues positive: ", torch.sort(eigv.real))

    def get_Hessian(self) -> Tensor:
        """Get the Hessian matrix."""

        return self.Hessian

    def compute_Covariance(self, eps: float = 1e-6, make_psd: bool = False) -> None:
        """Compute the covariance matrix."""

        if make_psd:
            eigv = torch.linalg.eigvals(-self.Hessian)
            all_pos = (eigv.real >= 0).all()
            if not all_pos:
                eps = -eps + eigv.real.min()
                print("eps after psd correction: ", eps)
        self.Covariance = torch.linalg.inv(
            -self.Hessian - eps * torch.eye(self.Hessian.size(0)).to(self.device)
        )

    def get_Covariance(self, diagonal_only: bool = False) -> Tensor:
        """Get the covariance matrix."""

        if diagonal_only:
            return self.Covariance.diag()
        else:
            return self.Covariance


def assign_priors(
    K: int,
    N: int,
    L: int,
    Z_categorical: int,
    P_poisson: int,
    P_gaussian: int,
    configuration,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """Set parameters mu and sigma of prior distributions."""

    device = configuration["device"]

    # Means
    U_mu_prior = torch.zeros((N, K)).to(device) + configuration["U_mu"]
    V_mu_prior = torch.zeros((N, K)).to(device) + configuration["V_mu"]
    W_mu_prior = torch.zeros((L, K, K)).to(device) + configuration["W_mu"]
    Hcategorical_mu_prior = (
        torch.zeros((K, Z_categorical)).to(device) + configuration["Hcategorical_mu"]
    )
    Hpoisson_mu_prior = (
        torch.zeros((K, P_poisson)).to(device) + configuration["Hpoisson_mu"]
    )
    Hgaussian_mu_prior = (
        torch.zeros((K, P_gaussian)).to(device) + configuration["Hgaussian_mu"]
    )

    # Standard deviations
    U_std_prior = torch.ones((N, K)).to(device) * configuration["U_std"]
    V_std_prior = torch.ones((N, K)).to(device) * configuration["V_std"]
    W_std_prior = torch.ones((L, K, K)).to(device) * configuration["W_std"]
    Hcategorical_std_prior = (
        torch.ones((K, Z_categorical)).to(device) * configuration["Hcategorical_std"]
    )
    Hpoisson_std_prior = (
        torch.ones((K, P_poisson)).to(device) * configuration["Hpoisson_std"]
    )
    Hgaussian_std_prior = (
        torch.ones((K, P_gaussian)).to(device) * configuration["Hgaussian_std"]
    )

    return (
        U_mu_prior,
        V_mu_prior,
        W_mu_prior,
        Hcategorical_mu_prior,
        Hpoisson_mu_prior,
        Hgaussian_mu_prior,
        U_std_prior,
        V_std_prior,
        W_std_prior,
        Hcategorical_std_prior,
        Hpoisson_std_prior,
        Hgaussian_std_prior,
    )
