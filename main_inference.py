import os
import time
import yaml
import numpy as np
import torch
from argparse import ArgumentParser
from src.model import PIHAM, assign_priors


def main():
    p = ArgumentParser()
    p.add_argument(
        "-f", "--in_folder", type=str, default="data/input/"
    )  # path of the input
    p.add_argument(
        "-d", "--data_file", type=str, default="synthetic_data.pt"
    )  # name of the data file
    p.add_argument("-K", "--K", type=int, default=5)  # number of communities
    args = p.parse_args()

    # Folder to store the results
    output_folder = "data/output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Setting to run the algorithm
    with open("src/setting_inference.yaml") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    device = configuration["device"]

    # Save setting in the output folder
    with open(output_folder + f"setting_inference.yaml", "w") as file:
        _ = yaml.dump(configuration, file)

    """
    Import data
    """
    # We analyze a heterogeneous and attributed multilayer network with L = 3 layers: one with binary interactions,
    # the second with nonnegative discrete weights, and the third with real values.
    # Additionally, each node is associated with three covariates: one categorical with Z = 4 categories,
    # one with nonnegative discrete values, and the last with real values.
    data = torch.load(args.in_folder + args.data_file)
    A = data["A"].to(device)  # adjacency tensor
    X_categorical = data["X_categorical"].to(device)  # categorical covariates
    X_poisson = data["X_poisson"].to(device)  # Poisson covariates
    X_gaussian = data["X_gaussian"].to(device)  # Gaussian covariates

    # Save variables
    K = args.K  # number of communities
    L = len(A)  # number of layers
    N = A.size(1)  # number of nodes
    Z_categorical = X_categorical.size(
        1
    )  # number of categories for the categorical attribute
    P_poisson = X_poisson.size(1)  # number of Poisson attributes
    P_gaussian = X_gaussian.size(1)  # number of Gaussian attributes

    """
    Run model
    """
    tic = time.time()

    # Assign priors
    (
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
    ) = assign_priors(N, L, K, Z_categorical, P_poisson, P_gaussian, configuration)

    # Initialize model
    model = PIHAM(
        U_mu_prior,
        U_std_prior,
        V_mu_prior,
        V_std_prior,
        W_mu_prior,
        W_std_prior,
        Hcategorical_mu_prior,
        Hcategorical_std_prior,
        Hpoisson_mu_prior,
        Hpoisson_std_prior,
        Hgaussian_mu_prior,
        Hgaussian_std_prior,
        K,
        N,
        L,
        Z_categorical,
        P_poisson,
        P_gaussian,
        configuration,
    )

    # Fit model
    model.fit(
        A,
        X_categorical,
        X_poisson,
        X_gaussian,
        gamma=configuration["gamma"],
        tolerance=configuration["tolerance"],
        num_iter=configuration["num_iter"],
        likelihood_weight=configuration["lik_weight"],
        learning_rate=configuration["learning_rate"],
        verbose=configuration["verbose"],
        N_seeds=configuration["N_seeds"],
    )

    # Compute the Hessian
    model.compute_Hessian(
        A,
        X_categorical,
        X_poisson,
        X_gaussian,
        likelihood_weight=configuration["lik_weight"],
    )
    # Test if the negative Hessian is positive definite
    model.is_neg_Hessian_pos_def()
    # Invert the Hessian
    model.compute_Covariance(eps=1e-6)

    # Save results
    model.save_results(
        folder_name=output_folder,
        file_name=f"_{args.data_file.replace('.pt', '')}_K{K}",
    )

    toc = time.time()
    print(f"\n ---- Time elapsed: {np.round(toc - tic, 4)} seconds ----")


if __name__ == "__main__":
    main()
