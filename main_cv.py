import os
import time
import yaml
import csv
import numpy as np
import torch
from sklearn import metrics
from argparse import ArgumentParser
from src import tools
from src import functions_cv


def main():
    p = ArgumentParser()
    p.add_argument(
        "-f", "--in_folder", type=str, default="data/input/"
    )  # path of the input
    p.add_argument(
        "-d", "--data_file", type=str, default="synthetic_data.pt"
    )  # name of the data file
    p.add_argument("-K", "--K", type=int, default=3)  # number of communities
    p.add_argument(
        "-F", "--NFold", type=int, default=5
    )  # number of folds to perform cv
    p.add_argument(
        "-v", "--cv_type", type=str, choices=["kfold", "random"], default="kfold"
    )  # type of cv routine
    p.add_argument(
        "-x", "--out_results", type=bool, default=True
    )  # flag to save the prediction performance
    p.add_argument(
        "-m", "--out_mask", type=bool, default=False
    )  # flag to save the cv masks
    p.add_argument(
        "-i", "--out_inference", type=bool, default=False
    )  # flag to save the inferred parameters during cv routine
    args = p.parse_args()

    tic = time.time()

    # Folder to store the results
    output_folder = "data/output/cv/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Setting to run the algorithm
    with open("src/setting_inference.yaml") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    device = configuration["device"]

    # Save setting in the output folder
    with open(output_folder + f"setting_inference.yaml", "w") as file:
        _ = yaml.dump(configuration, file)

    # Set the random seed
    seed = configuration["rseed"]
    rng = np.random.RandomState(seed)

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

    """
    Setup cross-validation routine
    """
    print("\n### Cross-validation routine ###")
    columns = [
        "K",
        "fold",
        "rseed",
        "auc_train_AB",
        "auc_test_AB",
        "mean_mae_AP",
        "mae_train_AP",
        "mae_test_AP",
        "mean_rmse_AG",
        "rmse_train_AG",
        "rmse_test_AG",
        "rp_XB",
        "mrf_XB",
        "acc_train_XB",
        "acc_test_XB",
        "mean_mae_XP",
        "mae_train_XP",
        "mae_test_XP",
        "mean_rmse_XG",
        "rmse_train_XG",
        "rmse_test_XG",
    ]
    prediction_results = [0 for _ in range(len(columns))]
    prediction_results[0] = K

    # Save prediction results
    if args.out_results:
        out_file = output_folder + f"cv_{args.data_file.replace('.pt', '')}.csv"
        if not os.path.isfile(out_file):  # write header
            with open(out_file, "w") as outfile:
                wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
                wrtr.writerow(columns)
        outfile = open(out_file, "a")
        wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
        print(f"Results will be saved in: {out_file}")

    # Shuffle indices
    if args.cv_type == "kfold":
        idxA = functions_cv.shuffle_indicesA(N, L, rng)
        idxX = functions_cv.shuffle_indicesX(N, rng)
    else:
        idxA = None
        idxX = None

    # Iterate over the folds
    for fold in range(args.NFold):
        print("FOLD ", fold)

        prediction_results[1], prediction_results[2] = fold, seed

        # Set cross-validation masks
        maskG, maskX = functions_cv.extract_masks(
            N,
            L,
            idxA=idxA,
            idxX=idxX,
            cv_type=args.cv_type,
            NFold=args.NFold,
            fold=fold,
            rng=rng,
            out_mask=args.out_mask,
            output_folder=output_folder,
            data_file=args.data_file.replace(".pt", ""),
        )

        A_train = A.clone()
        A_train[torch.Tensor(maskG).bool() > 0] = 0
        X_categorical_train = X_categorical.clone()
        X_poisson_train = X_poisson.clone()
        X_gaussian_train = X_gaussian.clone()
        X_categorical_train[torch.Tensor(maskX).bool() > 0] = 0
        X_poisson_train[torch.Tensor(maskX).bool() > 0] = 0
        X_gaussian_train[torch.Tensor(maskX).bool() > 0] = 0

        # Fit model on the training set
        U, V, W, Hcategorical, Hpoisson, Hgaussian, algo_obj = functions_cv.fit_model(
            K,
            N,
            L,
            A_train,
            X_categorical_train,
            X_poisson_train,
            X_gaussian_train,
            **configuration,
        )

        # Save inferred parameters during cv routine
        if args.out_inference:
            out_inference = (
                output_folder + f"theta_{args.data_file.replace('.pt', '')}_f{fold}.npz"
            )
            np.savez_compressed(
                out_inference,
                U=U.detach().cpu().numpy(),
                V=V.detach().cpu().numpy(),
                W=W.detach().cpu().numpy(),
                Hcategorical=Hcategorical.detach().cpu().numpy(),
                Hpoisson=Hpoisson.detach().cpu().numpy(),
                Hgaussian=Hgaussian.detach().cpu().numpy(),
            )
            print(f"Inferred parameters saved in: {out_inference}")
            print('To load: theta=np.load(filename), then e.g. theta["U"]')

        # Compute expected values
        Lambda_bernoulli, Lambda_poisson, Lambda_gaussian, _, _, _ = algo_obj.forward()
        # Account for label switching
        P = tools.calculate_permutation(U.detach().numpy(), V.detach().numpy())
        V_permut = torch.matmul(V, torch.Tensor(P))
        pi_categorical = tools.forward_categorical_covariate(U, V_permut, Hcategorical)
        pi_poisson = tools.forward_poisson_covariate(U, V_permut, Hpoisson)
        pi_gaussian = tools.forward_gaussian_covariate(U, V_permut, Hgaussian)

        # Compute prediction performance
        # A Bernoulli
        prediction_results[3] = functions_cv.AUC(
            A.detach().cpu().numpy()[0].reshape(1, N, N),
            Lambda_bernoulli.detach().cpu().numpy().reshape(1, N, N),
            mask=np.logical_not(maskG)[0].reshape(1, N, N),
        )
        prediction_results[4] = functions_cv.AUC(
            A.detach().cpu().numpy()[0].reshape(1, N, N),
            Lambda_bernoulli.detach().cpu().numpy().reshape(1, N, N),
            mask=maskG[0].reshape(1, N, N),
        )
        # A Poisson
        prediction_results[6] = functions_cv.MAE(
            A.detach().cpu().numpy()[1].reshape(1, N, N),
            Lambda_poisson.detach().cpu().numpy().reshape(1, N, N),
            mask=np.logical_not(maskG)[1].reshape(1, N, N),
        )
        prediction_results[7] = functions_cv.MAE(
            A.detach().cpu().numpy()[1].reshape(1, N, N),
            Lambda_poisson.detach().cpu().numpy().reshape(1, N, N),
            mask=maskG[1].reshape(1, N, N),
        )
        # A Gaussian
        prediction_results[9] = functions_cv.RMSE(
            A.detach().cpu().numpy()[2].reshape(1, N, N),
            Lambda_gaussian.detach().cpu().numpy().reshape(1, N, N),
            mask=np.logical_not(maskG)[2].reshape(1, N, N),
        )
        prediction_results[10] = functions_cv.RMSE(
            A.detach().cpu().numpy()[2].reshape(1, N, N),
            Lambda_gaussian.detach().cpu().numpy().reshape(1, N, N),
            mask=maskG[2].reshape(1, N, N),
        )
        # Baselines
        M_poisson_mean_train = (
            (Lambda_poisson[0][np.logical_not(maskG)[1]]).mean().detach().cpu().numpy()
        )
        prediction_results[5] = functions_cv.MAE(
            A.detach().cpu().numpy()[1][maskG[1]],
            np.tile(M_poisson_mean_train, A.detach().cpu().numpy()[1][maskG[1]].shape),
        )
        M_gauss_mean_train = (
            (Lambda_gaussian[0][np.logical_not(maskG)[2]]).mean().detach().cpu().numpy()
        )
        prediction_results[8] = functions_cv.RMSE(
            A.detach().cpu().numpy()[2][maskG[2]],
            np.tile(M_gauss_mean_train, A.detach().cpu().numpy()[2][maskG[2]].shape),
        )
        # X Bernoulli
        prediction_results[13] = functions_cv.accuracy(
            X_categorical.detach().cpu().numpy(),
            pi_categorical.detach().cpu().numpy(),
            mask=np.logical_not(maskX),
        )
        prediction_results[14] = functions_cv.accuracy(
            X_categorical.detach().cpu().numpy(),
            pi_categorical.detach().cpu().numpy(),
            mask=maskX,
        )
        # Baselines
        prediction_results[11] = 1 / X_categorical.shape[1]
        true_label = np.argmax(X_categorical[maskX], axis=1)
        mrf_label = np.tile(
            np.argmax(sum(X_categorical[np.logical_not(maskX)])), true_label.shape
        )
        prediction_results[12] = metrics.accuracy_score(true_label, mrf_label)
        # X Poisson
        prediction_results[16] = functions_cv.MAE(
            X_poisson.detach().cpu().numpy(),
            pi_poisson.detach().cpu().numpy(),
            mask=np.logical_not(maskX),
        )
        prediction_results[17] = functions_cv.MAE(
            X_poisson.detach().cpu().numpy(),
            pi_poisson.detach().cpu().numpy(),
            mask=maskX,
        )
        # X Gaussian
        prediction_results[19] = functions_cv.RMSE(
            X_gaussian.detach().cpu().numpy(),
            pi_gaussian.detach().cpu().numpy(),
            mask=np.logical_not(maskX),
        )
        prediction_results[20] = functions_cv.RMSE(
            X_gaussian.detach().cpu().numpy(),
            pi_gaussian.detach().cpu().numpy(),
            mask=maskX,
        )
        # Baselines
        X_poisson_mean_train = (
            (X_poisson[np.logical_not(maskX)]).mean(axis=0).detach().cpu().numpy()
        )
        prediction_results[15] = functions_cv.MAE(
            X_poisson.detach().cpu().numpy(),
            np.tile(X_poisson_mean_train, (X_poisson.shape[0], 1)),
            mask=maskX,
        )
        X_gaussian_mean_train = (
            (X_gaussian[np.logical_not(maskX)]).mean(axis=0).detach().cpu().numpy()
        )
        prediction_results[18] = functions_cv.RMSE(
            X_gaussian.detach().cpu().numpy(),
            np.tile(X_gaussian_mean_train, (X_gaussian.shape[0], 1)),
            mask=maskX,
        )

        # Save prediction results
        if args.out_results:
            wrtr.writerow(prediction_results)
            outfile.flush()

    if args.out_results:
        outfile.close()

    toc = time.time()
    print(f"\n ---- Time elapsed: {np.round(toc-tic, 4)} seconds ----")


if __name__ == "__main__":
    main()
