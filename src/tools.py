"""Functions used in the forward pass and helpers."""
import math
import numpy as np
import torch
from torch import Tensor
from functorch import vmap
from scipy.stats import norm
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Tuple


# Helpers
def row_wise_softmax(Y: Tensor) -> Tensor:
    """Compute the softmax function for every row of a 2D Tensor."""

    exp_Y = torch.exp(Y)
    return exp_Y / exp_Y.sum(1).reshape(-1, 1)


def logistic(Y: Tensor) -> Tensor:
    """Compute the logistic function for every entry of a Tensor."""

    return 1 / (1 + torch.exp(-Y))


def calculate_permutation(u_reference: np.ndarray, u_permut: np.ndarray) -> np.ndarray:
    """Permute membership matrices to account for label switching."""

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


def calculate_Overlap(mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Compute the area of overlap between every pair of distributions for a given node."""

    def normal_abs_diff(x, mean1, std1, mean2, std2):
        func = np.abs(norm.pdf(x, mean1, std1) - norm.pdf(x, mean2, std2))
        return func

    K = mu.shape[0]  # number of communities

    results = []
    for k in np.arange(K):
        for q in np.arange(k + 1, K):
            overlap = (
                1
                - 0.5
                * integrate.quad(
                    normal_abs_diff,
                    -np.inf,
                    np.inf,
                    args=(mu[k], std[k], mu[q], std[q]),
                )[0]
            )
            results.append(overlap)

    return np.array(results)


def compute_L2barycenter(
    x: np.ndarray, mu: np.ndarray, std: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """Compute the L2-barycenter distribution, which represents a weighted average
    of the node-community distributions. It also computes its mean and variance
    using the trapezoidal rule to approximate the integral.

    source: https://pythonot.github.io/auto_examples/unbalanced-partial/plot_UOT_barycenter_1D.html
    """

    # Gaussian distributions
    a = norm(loc=mu[0], scale=std[0]).pdf(x)
    b = norm(loc=mu[1], scale=std[1]).pdf(x)
    c = norm(loc=mu[2], scale=std[2]).pdf(x)

    # Create matrix A containing all distributions
    A = np.vstack((a, b, c)).T
    # Normalize weights
    weights = np.ones(A.shape[1])
    weights /= weights.sum()

    # Compute L2-barycenter distribution
    bary_l2 = A.dot(weights)

    # Approximate its mean and variance
    delta = (x[-1] - x[0]) / len(x)
    mean = (
        delta
        / 2
        * np.sum(
            [
                bary_l2[i - 1] * x[i - 1] + bary_l2[i] * x[i]
                for i in np.arange(1, len(x))
            ]
        )
    )
    var = (
        delta
        / 2
        * np.sum(
            [
                bary_l2[i - 1] * x[i - 1] ** 2 + bary_l2[i] * x[i] ** 2
                for i in np.arange(1, len(x))
            ]
        )
        - mean**2
    )

    return bary_l2, mean, var


def compute_alpha_from_normal(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Compute the alpha parameter of a Dirichlet distribution from parameters
    of a Normal distribution, using the Laplace Matching technique.
    """

    K = mu.shape[0]
    sum_exp = np.sum(np.exp(-1 * mu)).reshape(-1, 1)
    alpha = 1 / var * (1 - 2 / K + np.exp(mu) / K**2 * sum_exp)

    return alpha


def draw_Dirichlet(dist, ax, border=True, nlevels=200) -> None:
    """Functions for drawing contours of Dirichlet distributions over an equilateral triangle (2-simplex).

    source: https://gist.github.com/agitter/46b2169a035ad25b5d2b024a00344d54
    """

    _corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    _AREA = 0.5 * 1 * 0.75**0.5
    _triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

    # For each corner of the triangle, the pair of other corners
    _pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
    # The area of the triangle formed by point xy and another pair or points
    tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

    def xy2bc(xy, tol=1.0e-4):
        """Converts 2D Cartesian coordinates to barycentric."""
        coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
        return np.clip(coords, tol, 1.0 - tol)

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=8)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    ax.tricontour(trimesh, pvals, levels=10, linewidths=1, alpha=0.7, colors="darkgray")
    cnt = ax.tricontourf(trimesh, pvals, nlevels, antialiased=True, cmap="Greys")

    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)

    ax.set_aspect("equal", adjustable="box")
    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)
    if border is True:
        ax.triplot(_triangle, color="black", linewidth=1)


# Functions used in the forward pass of the adjacency tensor
def forward_gaussian_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Gaussian layers."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_bernoulli_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Bernoulli layers."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = logistic(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


def forward_poisson_layer(U: Tensor, V: Tensor, W: Tensor) -> Tensor:
    """Compute expected values for Poisson layers."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    W = torch.exp(W)

    def matmul_U(X):
        return torch.matmul(U, X)

    Lambda = vmap(matmul_U)(torch.matmul(W, V.T))

    return Lambda


# Functions used in the forward pass of the design matrix
def forward_categorical_covariate(U, V, H):
    """Compute expected values for categorical covariates."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = row_wise_softmax(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_gaussian_covariate(U, V, H):
    """Compute expected values for Gaussian covariates."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi


def forward_poisson_covariate(U, V, H):
    """Compute expected values for Poisson covariates."""

    U = row_wise_softmax(U)
    V = row_wise_softmax(V)
    H = torch.exp(H)

    pi = 0.5 * torch.matmul((U + V), H)

    return pi
