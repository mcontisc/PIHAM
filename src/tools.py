"""Functions used in the forward pass and helpers."""
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


def feature_accuracy(Y, pi):
    """Compute the feature accuracy of a given prediction"""
    Y_argmax = Y.argmax(1)
    pi_argmax = pi.argmax(1)
    assert len(Y_argmax) == len(Y)
    acc = torch.sum(torch.tensor(Y_argmax == pi_argmax)) / len(Y_argmax)
    return acc


def rmse(Yhat, Y):
    """Compute root mean square error"""
    return torch.sqrt(torch.mean((Yhat - Y) ** 2))


# Functions used in the forward pass of the adjacency tensor
def forward_gaussian_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    # equivalent to U @ (W @ V.T)
    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_bernoulli_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = logistic(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_poisson_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = torch.exp(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


# Functions used in the forward pass of the design matrix
def forward_categorical_covariate(U, V, H):
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = row_wise_softmax(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_gaussian_covariate(U, V, H):
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_poisson_covariate(U, V, H):
    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = torch.exp(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi
