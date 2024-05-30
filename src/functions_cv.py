""" Functions used in the cross-validation routine."""
import pickle
import numpy as np
from sklearn import metrics
from src.model import PIHAM, assign_priors


def shuffle_indicesA(N, L, rng):
    """Shuffle indices of adjacency tensor"""
    n_samples = int(N * N)
    idxG = [np.arange(n_samples) for _ in range(L)]
    for l in range(L):
        rng.shuffle(idxG[l])
    return idxG


def shuffle_indicesX(N, rng):
    """Shuffle row indices of design matrix"""
    idxX = np.arange(N)
    rng.shuffle(idxX)
    return idxX


def extract_masks(
    N, L, idxA, idxX, cv_type, NFold, fold, rng, out_mask, output_folder, data_file
):
    """Extract masks to use during the cross-validation routine to hide entries of A and X"""
    if cv_type == "kfold":
        assert L == len(idxA)
        maskA = np.zeros((L, N, N), dtype=bool)
        for l in range(L):
            n_samples = len(idxA[l])
            test = idxA[l][
                fold * (n_samples // NFold) : (fold + 1) * (n_samples // NFold)
            ]
            mask0 = np.zeros(n_samples, dtype=bool)
            mask0[test] = 1
            maskA[l] = mask0.reshape((N, N))

        maskX = np.zeros(N, dtype=bool)
        testcov = idxX[fold * (N // NFold) : (fold + 1) * (N // NFold)]
        maskX[testcov] = 1

    elif cv_type == "random":
        maskA = rng.binomial(1, 1.0 / float(NFold), size=(L, N, N))
        maskX = rng.binomial(1, 1.0 / float(NFold), size=N)

    if out_mask:
        outmaskA = output_folder + f"maskA_{data_file}_f{fold}.pkl"
        outmaskX = output_folder + f"maskX_{data_file}_f{fold}.pkl"
        print("Mask saved in ", outmaskA, outmaskX)
        with open(outmaskA, "wb") as f:
            pickle.dump(np.where(maskA > 0), f)
        with open(outmaskX, "wb") as f:
            pickle.dump(np.where(maskX > 0), f)

    return maskA, maskX


def fit_model(N, L, K, A, X_categorical, X_poisson, X_gaussian, **configuration):
    """Fit PIHAM model on the training set"""
    Z_categorical = X_categorical.size(
        1
    )  # number of categories for the categorical attribute
    P_poisson = X_poisson.size(1)  # number of Poisson attributes
    P_gaussian = X_gaussian.size(1)  # number of Gaussian attributes

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
        N,
        K,
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

    U, V, W, Hcategorical, Hpoisson, Hgaussian = model.get_UVWH()

    return U, V, W, Hcategorical, Hpoisson, Hgaussian, model


def AUC(Y, Yhat, mask=None):
    """Compute the AUC score"""
    Y = (Y > 0).astype("int")
    if mask is None:
        fpr, tpr, thresholds = metrics.roc_curve(Y.flatten(), Yhat.flatten())
    else:
        fpr, tpr, thresholds = metrics.roc_curve(Y[mask > 0], Yhat[mask > 0])
    return metrics.auc(fpr, tpr)


def accuracy(Y, Yhat, mask=None):
    """Compute the accuracy score"""
    if mask is None:
        true_label = np.argmax(Y, axis=1)
        pred_label = np.argmax(Yhat, axis=1)
    else:
        true_label = np.argmax(Y[mask > 0], axis=1)
        pred_label = np.argmax(Yhat[mask > 0], axis=1)
    acc = metrics.accuracy_score(true_label, pred_label)
    return acc


def RMSE(Y, Yhat, mask=None):
    """Compute the root mean square error"""
    if mask is None:
        Y = Y.flatten()
        Yhat = Yhat.flatten()
    else:
        Y = Y[mask > 0]
        Yhat = Yhat[mask > 0]

    if len(Y.shape) == 1:
        rmse = np.sqrt(np.mean((Y - Yhat) ** 2))
    else:
        rmse = []
        for j in range(Y.shape[1]):
            Yj = Y[:, j]
            Yhatj = Yhat[:, j]
            rmse.append(np.sqrt(np.mean((Yj - Yhatj) ** 2)))
    return rmse


def compute_mae(Y, Yhat, mask=None):
    """Compute the mean absolute error"""
    if mask is None:
        Y = Y.flatten()
        Yhat = Yhat.flatten()
    else:
        Y = Y[mask > 0]
        Yhat = Yhat[mask > 0]

    if len(Y.shape) == 1:
        mae = np.mean(np.abs(Y - Yhat))
    else:
        mae = []
        for j in range(Y.shape[1]):
            Yj = Y[:, j]
            Yhatj = Yhat[:, j]
            mae.append(np.mean(np.abs(Yj - Yhatj)))
    return mae
