"""Functions used in the forward pass and helpers."""
import numpy as np
import torch
from torch import Tensor
from functorch import vmap


# Helpers
def row_wise_softmax(Y: Tensor) -> Tensor:
    """Compute the softmax function for every row of a 2D Tensor"""
    exp_Y = torch.exp(Y)
    return exp_Y / exp_Y.sum(1).reshape(-1, 1)


def logistic(Y: Tensor) -> Tensor:
    """Compute the logistic function for every entry of a Tensor"""
    return 1 / (1 + torch.exp(-Y))


def calculate_permutation(u_reference: np.ndarray, u_permut: np.ndarray) -> np.ndarray:
    """Permute membership matrices to account for label switching"""
    N, RANK = u_reference.shape
    M = np.dot(np.transpose(u_permut), u_reference) / N
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining submatrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.0
        c_index = 0
        r_index = 0
        for i in range(RANK):
            if columns[i] == 0:
                for j in range(RANK):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i]
                            c_index = i
                            r_index = j
        if max_entry > 0:
            P[r_index, c_index] = 1
            columns[c_index] = 1
            rows[r_index] = 1
    if (np.sum(P, axis=1) == 0).any():
        row = np.where(np.sum(P, axis=1) == 0)[0]
        if (np.sum(P, axis=0) == 0).any():
            col = np.where(np.sum(P, axis=0) == 0)[0]
            P[row, col] = 1
    return P


# Functions used in the forward pass of the adjacency tensor
def forward_gaussian_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Gaussian layers"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_bernoulli_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Bernoulli layers"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = logistic(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_poisson_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Poisson layers"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = torch.exp(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


# Functions used in the forward pass of the design matrix
def forward_categorical_covariate(U, V, H):
    """Compute expected values for categorical covariates"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = row_wise_softmax(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_gaussian_covariate(U, V, H):
    """Compute expected values for Gaussian covariates"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_poisson_covariate(U, V, H):
    """Compute expected values for Poisson covariates"""
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = torch.exp(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi
